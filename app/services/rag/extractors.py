"""
파일 텍스트 추출 서비스

포맷별(PDF / Polaris / MarkItDown) 텍스트 추출 로직을 담당합니다.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from html import unescape as html_unescape
from concurrent.futures import ProcessPoolExecutor
from typing import Generator, List, Optional, Tuple

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import pdf4llm
except ImportError:
    pdf4llm = None

try:
    import win32com.client
    import pythoncom
except ImportError:
    win32com = None
    pythoncom = None

logger = logging.getLogger(__name__)

# PDF 처리용 프로세스 풀 (8 workers)
_pdf_process_pool: Optional[ProcessPoolExecutor] = None


def get_pdf_process_pool() -> ProcessPoolExecutor:
    """PDF 처리용 프로세스 풀 (lazy init)"""
    global _pdf_process_pool
    if _pdf_process_pool is None:
        _pdf_process_pool = ProcessPoolExecutor(max_workers=8)
    return _pdf_process_pool


# ----------------------------------------
# 멀티프로세싱용 모듈 레벨 함수 (ProcessPoolExecutor pickle 필요)
# ----------------------------------------

def _estimate_body_font_size(doc, sample_pages: int = 10) -> float:
    """전체 문서에서 본문 폰트 크기 추정 (가장 많이 등장하는 크기)"""
    from collections import Counter
    sizes = []
    for i in range(min(sample_pages, len(doc))):
        page = doc[i]
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = round(span.get("size", 0), 1)
                    if text and size > 0:
                        sizes.append(size)
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def _is_in_table(bbox, table_bboxes) -> bool:
    """블록이 표 영역 내에 있는지 확인"""
    bx0, by0, bx1, by1 = bbox
    for tx0, ty0, tx1, ty1 in table_bboxes:
        if bx0 >= tx0 - 2 and by0 >= ty0 - 2 and bx1 <= tx1 + 2 and by1 <= ty1 + 2:
            return True
    return False


def _table_to_markdown(table) -> str:
    """PyMuPDF 표를 마크다운 형식으로 변환"""
    try:
        data = table.extract()
        if not data:
            return ""
        rows = []
        for i, row in enumerate(data):
            cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
            rows.append("| " + " | ".join(cells) + " |")
            if i == 0:
                rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
        return "\n".join(rows)
    except Exception:
        return ""


def _table_to_text(table) -> str:
    """PyMuPDF 표 셀 내용을 plain text로 추출 (테두리 없는 표용)"""
    try:
        data = table.extract()
        if not data:
            return ""
        cells = []
        for row in data:
            for cell in row:
                text = str(cell or "").replace("\n", " ").strip()
                if text:
                    cells.append(text)
        return " ".join(cells)
    except Exception:
        return ""


def _bbox_overlaps(a, b) -> bool:
    """두 bbox가 겹치는지 확인"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _detect_borderless_tables(page, existing_bboxes: list, page_height: float) -> list:
    """y clustering + x histogram으로 테두리 없는 표 감지

    Returns: [(y_top, bbox_tuple, plain_text), ...]
    """
    Y_TOL = 4    # 같은 행으로 볼 y좌표 허용 오차 (px)
    X_TOL = 6    # 같은 열로 볼 x좌표 허용 오차 (px)
    MIN_COLS = 2  # 최소 열 수
    MIN_ROWS = 2  # 최소 행 수
    ROW_GAP = 20  # 같은 표로 묶을 최대 행 간격 (px)

    # 1. span 수집 (헤더/푸터·기존 표 영역 제외)
    spans = []
    for block in page.get_text("rawdict").get("blocks", []):
        if block.get("type") != 0:
            continue
        bx = block.get("bbox", [0, 0, 0, 0])
        if bx[1] < page_height * 0.05 or bx[3] > page_height * 0.95:
            continue
        if _is_in_table(bx, existing_bboxes):
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                sb = span.get("bbox", [0, 0, 0, 0])
                spans.append((sb[0], sb[1], sb[2], sb[3], text))

    if not spans:
        return []

    # 2. y clustering → rows
    rows = []
    for span in sorted(spans, key=lambda s: s[1]):
        placed = False
        for row in rows:
            row_y = sum(s[1] for s in row) / len(row)
            if abs(span[1] - row_y) <= Y_TOL:
                row.append(span)
                placed = True
                break
        if not placed:
            rows.append([span])

    # 3. x histogram → column positions
    all_x0 = [s[0] for row in rows for s in row]
    x_clusters = []
    for x in sorted(all_x0):
        placed = False
        for cluster in x_clusters:
            if abs(x - sum(cluster) / len(cluster)) <= X_TOL:
                cluster.append(x)
                placed = True
                break
        if not placed:
            x_clusters.append([x])

    col_positions = [sum(c) / len(c) for c in x_clusters if len(c) >= MIN_ROWS]
    if len(col_positions) < MIN_COLS:
        return []

    # 4. 표 구조 검증: 각 행이 2개 이상의 column에 align되는지 확인
    table_rows = []
    for row in rows:
        aligned = {col_x for span in row for col_x in col_positions if abs(span[0] - col_x) <= X_TOL}
        if len(aligned) >= MIN_COLS:
            table_rows.append(row)

    if len(table_rows) < MIN_ROWS:
        return []

    # 5. 연속된 행을 그룹으로 묶기
    groups = []
    current = [table_rows[0]]
    for row in table_rows[1:]:
        prev_y1 = max(s[3] for s in current[-1])
        curr_y0 = min(s[1] for s in row)
        if curr_y0 - prev_y1 <= ROW_GAP:
            current.append(row)
        else:
            groups.append(current)
            current = [row]
    groups.append(current)

    # 6. 그룹별 plain text + bbox 생성
    results = []
    for group in groups:
        all_spans = [s for row in group for s in row]
        x0 = min(s[0] for s in all_spans)
        y0 = min(s[1] for s in all_spans)
        x1 = max(s[2] for s in all_spans)
        y1 = max(s[3] for s in all_spans)
        bbox = (x0, y0, x1, y1)

        text_parts = []
        for row in group:
            row_text = " ".join(s[4] for s in sorted(row, key=lambda s: s[0]))
            text_parts.append(row_text)
        plain_text = " ".join(text_parts).strip()

        if plain_text:
            results.append((y0, bbox, plain_text))

    return results


def _classify_block(dominant_size: float, body_size: float, is_bold: bool) -> str:
    """폰트 크기 비율로 블록 타입 분류"""
    ratio = dominant_size / body_size if body_size > 0 else 1.0
    if ratio >= 1.8:
        return "heading1"
    elif ratio >= 1.4:
        return "heading2"
    elif ratio >= 1.1 or (abs(ratio - 1.0) < 0.05 and is_bold):
        return "heading3"
    return "body"


def _extract_structured_page(page, body_size: float) -> str:
    """PyMuPDF 페이지에서 구조화된 텍스트 추출"""
    from collections import Counter

    page_height = page.rect.height

    # 1. 표 감지
    table_bboxes = []
    table_items = []
    try:
        for table in page.find_tables():
            bbox = tuple(table.bbox)
            md = _table_to_markdown(table)
            if md:
                table_bboxes.append(bbox)
                table_items.append((bbox[1], "table", md))
    except Exception:
        pass

    # 2. 텍스트 블록 추출 및 분류
    raw = page.get_text("dict")
    text_items = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        bbox = block.get("bbox", [0, 0, 0, 0])
        y_top, y_bottom = bbox[1], bbox[3]

        # 헤더/푸터 제거 (상위 5%, 하위 5%)
        if y_top < page_height * 0.05 or y_bottom > page_height * 0.95:
            continue

        # 표 영역 내 블록 제거
        if _is_in_table(bbox, table_bboxes):
            continue

        spans = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                size = round(span.get("size", 0), 1)
                bold = "Bold" in span.get("font", "") or "bold" in span.get("font", "")
                if text and size > 0:
                    spans.append({"text": text, "size": size, "bold": bold})

        if not spans:
            continue

        sizes = [s["size"] for s in spans]
        dominant_size = Counter(sizes).most_common(1)[0][0]
        is_bold = any(s["bold"] for s in spans)
        full_text = " ".join(s["text"] for s in spans).strip()

        if not full_text:
            continue

        # 너무 작은 폰트 제거 (페이지 번호 등)
        if dominant_size < body_size * 0.7:
            continue

        block_type = _classify_block(dominant_size, body_size, is_bold)
        text_items.append((y_top, block_type, full_text))

    # 3. y좌표 순으로 정렬 후 마크다운 텍스트 생성
    all_items = text_items + [(y, t, m) for y, t, m in table_items]
    all_items.sort(key=lambda x: x[0])

    lines = []
    for _, block_type, text in all_items:
        if block_type == "heading1":
            lines.append(f"\n# {text}")
        elif block_type == "heading2":
            lines.append(f"\n## {text}")
        elif block_type == "heading3":
            lines.append(f"\n### {text}")
        elif block_type == "table":
            lines.append(f"\n{text}\n")
        else:
            lines.append(text)

    return "\n".join(lines).strip()


def _parse_pdf_in_process(pdf_path: str) -> List[Tuple[int, str]]:
    """별도 프로세스에서 PDF 파싱 (PyMuPDF 블록 기반)

    Returns: [(page_num, content), ...]
    """
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open(pdf_path)
        body_size = _estimate_body_font_size(doc)

        results = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
            page_text = _extract_structured_page(page, body_size)
            if page_text.strip():
                results.append((page_num, page_text))

        doc.close()
        return results
    except Exception:
        return []


def _convert_polaris_in_process(file_path: str, temp_output_dir: str) -> Optional[str]:
    """별도 프로세스에서 Polaris 변환 (멀티프로세싱용)

    Returns: markdown 문자열 또는 None
    """
    try:
        from app.services.api_clients.polaris_client import polaris_converter, PolarisJsonParser
        polaris_data = polaris_converter.convert(file_path, temp_output_dir, True)
        if polaris_data:
            parser = PolarisJsonParser(polaris_data)
            return parser.parse_to_markdown()
        return None
    except Exception:
        return None


class FileTextExtractor:
    """파일에서 텍스트를 추출하는 서비스"""

    def __init__(self):
        self._markitdown = MarkItDown() if MarkItDown else None

    # ----------------------------------------
    # 동기 추출 (asyncio.to_thread 용)
    # ----------------------------------------

    def iter_pdf_pages(self, pdf_path: str) -> Generator[Tuple[int, str], None, None]:
        """PDF를 페이지 단위로 yield하는 제너레이터 (PyMuPDF 블록 기반)"""
        try:
            try:
                import pymupdf as fitz
            except ImportError:
                import fitz

            doc = fitz.open(pdf_path)
            body_size = _estimate_body_font_size(doc)
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_num = page_idx + 1
                page_text = _extract_structured_page(page, body_size)
                if page_text.strip():
                    yield page_num, page_text.strip()
            doc.close()
        except Exception as e:
            logger.exception(f"[Extractor] PyMuPDF 변환 중 예외 발생: {e}")

    def _try_hwp_save(self, hwp, output_pdf_path: str) -> bool:
        """HWP → PDF 저장을 3가지 방법으로 순차 시도. 성공 시 True 반환."""
        try:
            hwp.HAction.GetDefault("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
            hwp.HParameterSet.HFileOpenSave.filename = output_pdf_path
            hwp.HParameterSet.HFileOpenSave.Format = "PDF"
            return hwp.HAction.Execute("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
        except Exception as e1:
            logger.warning(f"[Extractor] HWP 저장 방법 1 (FileSaveAsPdf) 실패: {e1}")

        try:
            hwp.SaveAs(output_pdf_path, "PDF")
            return os.path.exists(output_pdf_path)
        except Exception as e2:
            logger.warning(f"[Extractor] HWP 저장 방법 2 (SaveAs) 실패: {e2}")

        try:
            hwp.XHwpDocuments.Active.SaveAs(output_pdf_path, "PDF", "")
            return os.path.exists(output_pdf_path)
        except Exception as e3:
            logger.warning(f"[Extractor] HWP 저장 방법 3 (XHwpDocuments) 실패: {e3}")

        return False

    def convert_hwp_to_pdf(self, input_path: str, output_dir: str) -> Optional[str]:
        """win32com을 사용하여 HWP 파일을 PDF로 변환"""
        if not win32com or not pythoncom:
            logger.error("[Extractor] win32com 또는 pythoncom 라이브러리가 없습니다.")
            return None

        input_abs_path = os.path.abspath(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pdf_path = os.path.join(os.path.abspath(output_dir), f"{base_name}.pdf")

        hwp = None
        com_initialized = False
        try:
            pythoncom.CoInitialize()
            com_initialized = True

            try:
                hwp = win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
            except Exception:
                # gen_py 캐시 손상 시 자동 삭제 후 재시도
                gen_py_path = os.path.join(tempfile.gettempdir(), "gen_py")
                if os.path.exists(gen_py_path):
                    shutil.rmtree(gen_py_path, ignore_errors=True)
                    logger.warning("[Extractor] win32com gen_py 캐시 삭제 후 재시도")
                hwp = win32com.client.Dispatch("HWPFrame.HwpObject")

            hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
            hwp.Open(input_abs_path, "HWP", "forceopen:true")

            success = self._try_hwp_save(hwp, output_pdf_path)

            if success and os.path.exists(output_pdf_path):
                logger.info(f"[Extractor] HWP -> PDF 변환 성공: {output_pdf_path}")
                return output_pdf_path

            logger.error("[Extractor] HWP -> PDF 변환: 모든 방법 실패")
            return None

        except Exception as e:
            logger.exception(f"[Extractor] win32com HWP 변환 중 오류 발생: {e}")
            return None
        finally:
            if hwp:
                try:
                    hwp.Quit()
                except Exception:
                    pass
            if com_initialized:
                pythoncom.CoUninitialize()

    def convert_sync(self, file_path: str) -> Optional[str]:
        """MarkItDown 동기 변환 (asyncio.to_thread 사용 시)"""
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".txt":
            return self._read_text_file(file_path)
        if self._markitdown is None:
            return None
        try:
            result = self._markitdown.convert(file_path)
            return result.text_content if result and result.text_content else None
        except Exception as e:
            logger.error(f"[Extractor] MarkItDown 변환 오류: {e}")
            return None

    @staticmethod
    def _read_text_file(file_path: str) -> Optional[str]:
        """TXT 파일을 인코딩 자동 감지하여 읽기"""
        for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                if text.strip():
                    return text
            except (UnicodeDecodeError, LookupError):
                continue
        logger.error(f"[Extractor] TXT 인코딩 감지 실패: {file_path}")
        return None

    # ----------------------------------------
    # 비동기 추출
    # ----------------------------------------

    async def extract_with_polaris(self, file_path: str, on_progress=None) -> Optional[str]:
        """Polaris 변환으로 텍스트 추출 (임시 디렉토리 자체 관리)"""
        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")
        temp_output_dir = tempfile.mkdtemp()
        try:
            loop = asyncio.get_event_loop()
            parse_start = time.time()
            markdown_content = await loop.run_in_executor(
                get_pdf_process_pool(),
                _convert_polaris_in_process,
                file_path,
                temp_output_dir
            )
            parse_elapsed = time.time() - parse_start
            if not markdown_content:
                logger.error("[Extractor] Polaris 변환 실패")
                return None
            logger.info(f"[Extractor] Polaris 텍스트 추출 완료: {len(markdown_content)}자, {parse_elapsed:.2f}초")
            return markdown_content
        finally:
            shutil.rmtree(temp_output_dir, ignore_errors=True)

    async def extract_pdf_pages(self, file_path: str, on_progress=None) -> List[Tuple[int, str]]:
        """PDF 멀티프로세싱 페이지별 텍스트 추출"""
        if on_progress:
            await on_progress(10, "페이지 파싱 중...")
        loop = asyncio.get_event_loop()
        parse_start = time.time()
        page_results = await loop.run_in_executor(
            get_pdf_process_pool(),
            _parse_pdf_in_process,
            file_path
        )
        parse_elapsed = time.time() - parse_start
        logger.info(f"[Extractor] PDF 텍스트 추출 완료: {len(page_results)}페이지, {parse_elapsed:.2f}초")
        return page_results

    async def extract_with_markitdown(self, file_path: str, on_progress=None) -> Optional[str]:
        """MarkItDown으로 텍스트 추출"""
        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".txt":
            return await asyncio.to_thread(self._read_text_file, file_path)
        if self._markitdown is None:
            logger.error("[Extractor] MarkItDown 미설치")
            return None
        try:
            parse_start = time.time()
            result = await asyncio.to_thread(self._markitdown.convert, file_path)
            parse_elapsed = time.time() - parse_start
            if result and result.text_content:
                text = html_unescape(result.text_content)
                logger.info(f"[Extractor] MarkItDown 텍스트 추출 완료: {len(text)}자, {parse_elapsed:.2f}초")
                return text
            return None
        except Exception as e:
            logger.error(f"[Extractor] MarkItDown 오류: {e}")
            return None

    @staticmethod
    def extract_hwpx_direct(file_path: str) -> Optional[str]:
        """HWPX(ZIP+XML)에서 win32com 없이 텍스트 직접 추출."""
        import re
        import zipfile
        from xml.etree import ElementTree as ET
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                if "Preview/PrvText.txt" in names:
                    raw = zf.read("Preview/PrvText.txt")
                    for enc in ("utf-8", "utf-16", "utf-16-le", "cp949"):
                        try:
                            text = raw.decode(enc).strip()
                        except UnicodeDecodeError:
                            continue
                        if text:
                            return text
                texts: List[str] = []
                section_names = sorted(
                    n for n in names
                    if n.startswith("Contents/section") and n.endswith(".xml")
                )
                for name in section_names:
                    try:
                        root = ET.fromstring(zf.read(name))
                    except (KeyError, ET.ParseError):
                        continue
                    for elem in root.iter():
                        tag = elem.tag.split("}", 1)[-1]
                        if tag in ("t", "char") and elem.text:
                            texts.append(elem.text)
                    texts.append("\n")
            joined = "".join(texts)
            joined = re.sub(r"[ \t]+", " ", joined)
            joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
            return joined or None
        except Exception as e:
            logger.error(f"[Extractor] HWPX 직접 파싱 실패: {e}")
            return None

    async def extract_with_win32hwp(self, file_path: str, on_progress=None) -> Optional[List[Tuple[int, str]]]:
        """HWP → PDF (win32com) → 텍스트 추출. .hwpx는 win32com 없이 직접 파싱 우선."""
        if file_path.lower().endswith(".hwpx"):
            if on_progress:
                await on_progress(5, "HWPX 직접 파싱 중...")
            direct_text = await asyncio.to_thread(self.extract_hwpx_direct, file_path)
            if direct_text:
                return [(1, direct_text)]
            logger.warning("[Extractor] HWPX 직접 파싱 결과 비어있음 → win32com 폴백")

        if on_progress:
            await on_progress(5, "HWP 문서 변환 중...")
        temp_dir = tempfile.mkdtemp()
        try:
            loop = asyncio.get_event_loop()
            pdf_path = await loop.run_in_executor(
                None,  # thread pool (COM 객체는 프로세스 풀 불가)
                self.convert_hwp_to_pdf,
                file_path,
                temp_dir
            )
            if not pdf_path:
                logger.error("[Extractor] HWP → PDF 변환 실패")
                return None
            return await self.extract_pdf_pages(pdf_path, on_progress)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def extract_text(
        self, file_path: str, file_type: str, on_progress=None
    ) -> Tuple[Optional[List[Tuple[int, str]]], Optional[str]]:
        """파일 타입별 추출 함수로 분기

        Returns:
            (page_results, markdown_content)
            - pdf:    ([(page_num, text), ...], None)
            - 나머지: (None, "markdown string")
        """
        if file_type == "pdf":
            return await self.extract_pdf_pages(file_path, on_progress), None
        if file_type == "polaris":
            return None, await self.extract_with_polaris(file_path, on_progress)
        if file_type == "hwp_win32":
            return await self.extract_with_win32hwp(file_path, on_progress), None
        return None, await self.extract_with_markitdown(file_path, on_progress)


# 싱글톤 인스턴스
file_text_extractor = FileTextExtractor()
