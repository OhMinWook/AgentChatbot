import asyncio
import tempfile
import logging
import os
import re
import time
from typing import List, Dict, Optional

# Polaris 비활성화 시 사용할 대체 라이브러리들
try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import pdf4llm
except ImportError:
    pdf4llm = None

# Windows 환경에서 HWP 제어를 위한 win32com
try:
    import win32com.client
    import pythoncom
except ImportError:
    win32com = None
    pythoncom = None

from app.core.config import settings
from app.services.api_clients.model_server_client import model_server_client
from app.services.api_clients.polaris_client import polaris_converter, PolarisJsonParser

logger = logging.getLogger(__name__)


class RagIngestionService:
    def __init__(self):
        self._markitdown = MarkItDown() if MarkItDown else None

    # ========================================
    # 청킹 파이프라인
    # ========================================

    def _parse_page_texts(self, markdown_content: str) -> List[tuple]:
        """마크다운에서 페이지별 텍스트 추출"""
        page_pattern = re.compile(r'\n--- Page (\d+) ---\n')
        parts = page_pattern.split(markdown_content)

        page_texts = []  # [(page_num, text), ...]
        if parts[0].strip():
            page_texts.append((1, parts[0].strip()))

        for i in range(1, len(parts), 2):
            try:
                page_num = int(parts[i])
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if content:
                    page_texts.append((page_num, content))
            except (IndexError, ValueError):
                continue

        return page_texts

    def _create_chunks(self, markdown_content: str, file_name: str) -> List[Dict]:
        """
        마크다운을 청크로 분할합니다.

        Returns:
            [{
                "id": "manual.pdf_p1_0",
                "content": "청크 내용",
                "metadata": {"source": "manual.pdf", "page": 1}
            }, ...]
        """
        if not markdown_content.strip():
            return []

        page_texts = self._parse_page_texts(markdown_content)
        if not page_texts:
            return []

        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP

        # 전체 텍스트를 페이지 정보와 함께 하나로 합침
        full_text = ""
        page_ranges = []  # [(text_start, text_end, page_num), ...]

        for page_num, page_text in page_texts:
            start_idx = len(full_text)
            full_text += page_text + "\n\n"
            end_idx = len(full_text)
            page_ranges.append((start_idx, end_idx, page_num))

        # 청크 분할
        chunks = []
        start = 0
        text_len = len(full_text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = full_text[start:end]

            # 문장 경계에서 자르기
            if end < text_len:
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                cut_point = max(last_period, last_newline)
                if cut_point > chunk_size * 0.7:
                    chunk_text = chunk_text[:cut_point + 1]
                    end = start + cut_point + 1

            if chunk_text.strip():
                # 청크가 포함하는 페이지 계산
                chunk_start = start
                chunk_end = start + len(chunk_text)
                page = 1

                for (ps, pe, pn) in page_ranges:
                    if ps < chunk_end and pe > chunk_start:
                        page = pn
                        break

                chunk_id = f"{file_name}_p{page}_{chunk_idx}"

                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text.strip(),
                    "metadata": {
                        "source": file_name,
                        "page": page
                    }
                })
                chunk_idx += 1

            start = end - chunk_overlap if end < text_len else text_len

        logger.info(f"[Chunking] {file_name}: {len(chunks)}개 청크 생성")
        return chunks

    def _process_pdf_with_pdf4llm(self, pdf_path: str, display_name: str, invoke_id: str) -> str:
        """pdf4llm을 사용하여 PDF를 마크다운으로 변환합니다. (페이지 구분자 포함)"""
        if pdf4llm is None:
            logger.error("pdf4llm 라이브러리가 설치되지 않았습니다.")
            return ""

        try:
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            markdown_parts = []
            for page_idx, page_data in enumerate(pages_data):
                real_page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                if content_text.strip():
                    markdown_parts.append(f"\n--- Page {real_page_num} ---\n{content_text.strip()}")

            full_markdown = "\n".join(markdown_parts)
            logger.info(f"[PDF] {display_name}: {len(full_markdown)}자, {len(markdown_parts)}페이지")
            return full_markdown
        except Exception as e:
            logger.exception(f"pdf4llm 변환 중 예외 발생: {e}")
            return ""

    def _convert_hwp_to_pdf_with_win32com(self, input_path: str, output_dir: str) -> Optional[str]:
        """win32com을 사용하여 한글(HWP) 파일을 PDF로 변환합니다."""
        if not win32com or not pythoncom:
            logger.error("win32com 또는 pythoncom 라이브러리가 없습니다.")
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
                hwp = win32com.client.Dispatch("HWPFrame.HwpObject")

            hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
            hwp.Open(input_abs_path, "HWP", "forceopen:true")

            success = False
            try:
                hwp.HAction.GetDefault("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
                hwp.HParameterSet.HFileOpenSave.filename = output_pdf_path
                hwp.HParameterSet.HFileOpenSave.Format = "PDF"
                success = hwp.HAction.Execute("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
            except Exception as e1:
                logger.warning(f"방법 1 (FileSaveAsPdf) 실패: {e1}")
                try:
                    hwp.SaveAs(output_pdf_path, "PDF")
                    success = os.path.exists(output_pdf_path)
                except Exception as e2:
                    logger.warning(f"방법 2 (SaveAs) 실패: {e2}")
                    try:
                        hwp.XHwpDocuments.Active.SaveAs(output_pdf_path, "PDF", "")
                        success = os.path.exists(output_pdf_path)
                    except Exception as e3:
                        logger.warning(f"방법 3 (XHwpDocuments) 실패: {e3}")

            if success and os.path.exists(output_pdf_path):
                logger.info(f"HWP -> PDF 변환 성공: {output_pdf_path}")
                return output_pdf_path
            else:
                logger.error("HWP -> PDF 변환: 모든 방법 실패")
                return None

        except Exception as e:
            logger.exception(f"win32com HWP 변환 중 오류 발생: {e}")
            return None
        finally:
            if hwp:
                try:
                    hwp.Quit()
                except Exception:
                    pass
            if com_initialized:
                pythoncom.CoUninitialize()

    async def _store_chunks_to_colbert(
        self,
        chunks: List[Dict],
        invoke_id: str,
        file_name: str,
        total_pages: int
    ):
        """
        청크 목록을 모델 서버의 ColBERT 인덱스에 저장합니다.
        """
        if not chunks:
            return

        total_chunks = len(chunks)
        logger.info(f"[Ingestion] ColBERT에 저장할 총 청크 수: {total_chunks}")

        # 모델 서버에 인덱싱 요청
        result = await model_server_client.colbert_index_documents(invoke_id, chunks)
        indexed_count = result.get("indexed_count", 0)
        logger.info(f"[Ingestion] ColBERT 인덱싱 완료: {indexed_count} / {total_chunks} 청크")

        # 문서 메타데이터 저장
        await model_server_client.colbert_store_document_metadata(
            invoke_id=invoke_id,
            file_name=file_name,
            total_pages=total_pages,
            total_chunks=total_chunks
        )
        logger.info(f"[Ingestion] 문서 메타데이터 저장 완료: {file_name} ({total_pages}페이지, {total_chunks}청크)")

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, on_progress=None, on_markdown=None):
        """
        파일을 처리하고, 결과를 ColBERT 인덱스에 저장합니다.

        Args:
            file_name: 파일명 (확장자 포함, 예: 'report.pdf')
            invoke_id: 세션 ID
            file_path: 파일 경로
            on_progress: 진행률 콜백 (async def func(percent, message))
            on_markdown: 마크다운 변환 결과 콜백 (async def func(markdown_content))
        """
        logger.info(f"[Ingestion] 파일 처리 시작: {file_name} (Room: {invoke_id})")

        if on_progress:
            await on_progress(0, "파일 처리 시작")

        display_name = file_name
        _, ext = os.path.splitext(display_name)
        ext_to_use = ext.lower().strip()

        markdown_content = ""

        # === 1. 문서 파싱 (0% -> 20%) ===
        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")

        if settings.POLARIS_ENABLED:
            logger.info("[Ingestion] Polaris 엔진 사용")
            with tempfile.TemporaryDirectory() as temp_output_dir:
                try:
                    polaris_start = time.time()
                    polaris_data = await asyncio.to_thread(
                        polaris_converter.convert, file_path, temp_output_dir, True
                    )
                    polaris_elapsed = time.time() - polaris_start
                    if polaris_data:
                        parser = PolarisJsonParser(polaris_data)
                        markdown_content = parser.parse_to_markdown()
                        logger.info(f"[Polaris] {display_name}: {len(markdown_content)}자 ({polaris_elapsed:.2f}s)")
                    else:
                        logger.error("[Ingestion] Polaris 변환 실패")
                        return
                except Exception as e:
                    logger.error(f"[Ingestion] Polaris 오류: {e}")
                    return
        else:
            logger.info(f"[Ingestion] 확장자: '{ext_to_use}' (파일명: {display_name})")

            # 1. HWP/HWPX 파일 처리
            if ext_to_use in ['.hwp', '.hwpx']:
                logger.info("[Ingestion] 한글 문서 -> PDF 변환 -> 마크다운 추출")
                with tempfile.TemporaryDirectory() as temp_pdf_dir:
                    converted_pdf = await asyncio.to_thread(
                        self._convert_hwp_to_pdf_with_win32com, file_path, temp_pdf_dir
                    )
                    if converted_pdf and os.path.exists(converted_pdf):
                        markdown_content = await asyncio.to_thread(
                            self._process_pdf_with_pdf4llm, converted_pdf, display_name, invoke_id
                        )
                    else:
                        logger.error("[Ingestion] HWP -> PDF 변환 실패")
                        return

            # 2. PDF 파일
            elif ext_to_use == '.pdf':
                logger.info("[Ingestion] PDF -> 마크다운 추출")
                markdown_content = await asyncio.to_thread(
                    self._process_pdf_with_pdf4llm, file_path, display_name, invoke_id
                )

            # 3. 기타 문서 (MarkItDown)
            else:
                logger.info("[Ingestion] 일반 문서 -> MarkItDown 변환")
                if self._markitdown is None:
                    logger.error("[Ingestion] MarkItDown 미설치")
                    return

                try:
                    result = await asyncio.to_thread(self._markitdown.convert, file_path)
                    if result and result.text_content:
                        markdown_content = result.text_content
                except Exception as e:
                    logger.error(f"[Ingestion] MarkItDown 오류: {e}")
                    return

        if on_progress:
            await on_progress(20, "문서 파싱 완료")

        if not markdown_content:
            logger.error("[Ingestion] 마크다운 없음")
            return

        # 마크다운 결과 콜백 (디버깅용)
        if on_markdown:
            await on_markdown(markdown_content)

        # === 2. 청킹 (20% -> 40%) ===
        if on_progress:
            await on_progress(25, "청크 생성 중...")

        chunks = self._create_chunks(markdown_content, display_name)

        if not chunks:
            logger.error("[Ingestion] 청크 생성 실패")
            return

        if on_progress:
            await on_progress(40, f"{len(chunks)}개 청크 생성 완료")

        # 총 페이지 수 계산
        total_pages = max((c["metadata"].get("page", 1) for c in chunks), default=1)

        # === 3. ColBERT 인덱싱 (40% -> 95%) ===
        if on_progress:
            await on_progress(50, "ColBERT 인덱싱 중...")

        try:
            indexing_task = asyncio.create_task(
                self._store_chunks_to_colbert(chunks, invoke_id, display_name, total_pages)
            )
            current_percent = 55
            while not indexing_task.done() and current_percent <= 90:
                await asyncio.sleep(3)
                if indexing_task.done():
                    break
                if on_progress:
                    await on_progress(current_percent, "ColBERT 인덱싱 중...")
                current_percent += 10

            await indexing_task
            logger.info(f"[Ingestion] ColBERT 인덱싱 완료 ({len(chunks)}개 청크)")
            if on_progress:
                await on_progress(95, "ColBERT 인덱싱 완료")
        except Exception as e:
            logger.error(f"[Ingestion] ColBERT 인덱싱 실패: {e}")
            raise

        # === 4. 완료 (100%) ===
        if on_progress:
            await on_progress(100, "인덱싱 완료!")

        logger.info(f"[Ingestion] {file_name} 완료 ({len(chunks)}개 청크)")


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
