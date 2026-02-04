import asyncio
import tempfile
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional, Generator, Tuple

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
from app.services.rag.qdrant_service import qdrant_service

logger = logging.getLogger(__name__)

# PDF 처리용 프로세스 풀 (8 workers)
_pdf_process_pool: Optional[ProcessPoolExecutor] = None


def get_pdf_process_pool() -> ProcessPoolExecutor:
    """PDF 처리용 프로세스 풀 (lazy init)"""
    global _pdf_process_pool
    if _pdf_process_pool is None:
        _pdf_process_pool = ProcessPoolExecutor(max_workers=8)
    return _pdf_process_pool


def _parse_pdf_in_process(pdf_path: str) -> List[Tuple[int, str]]:
    """
    별도 프로세스에서 PDF 파싱 (멀티프로세싱용)

    Returns: [(page_num, content), ...]
    """
    try:
        import pdf4llm as _pdf4llm
        pages_data = _pdf4llm.to_markdown(pdf_path, page_chunks=True)
        results = []
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
            if content_text.strip():
                results.append((page_num, content_text.strip()))
        return results
    except Exception as e:
        # 프로세스 내 로깅은 메인으로 전달 안됨, 빈 리스트 반환
        return []


def _convert_polaris_in_process(file_path: str, temp_output_dir: str) -> Optional[str]:
    """
    별도 프로세스에서 Polaris 변환 (멀티프로세싱용)

    Returns: markdown 문자열 또는 None
    """
    try:
        from app.services.api_clients.polaris_client import polaris_converter, PolarisJsonParser
        polaris_data = polaris_converter.convert(file_path, temp_output_dir, True)
        if polaris_data:
            parser = PolarisJsonParser(polaris_data)
            return parser.parse_to_markdown()
        return None
    except Exception as e:
        return None


class IncrementalChunker:
    """
    점진적 청킹을 위한 클래스.
    페이지 단위로 텍스트를 추가하면 청크를 생성합니다.
    """

    def __init__(self, file_name: str, chunk_size: int, chunk_overlap: int):
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.buffer = ""
        self.chunk_idx = 0
        self.current_page = 1
        self.page_ranges: List[Tuple[int, int, int]] = []  # (start, end, page_num)

    def add_page(self, page_num: int, text: str) -> List[Dict]:
        """페이지 텍스트 추가 후 생성 가능한 청크 반환"""
        if not text.strip():
            return []

        start_idx = len(self.buffer)
        self.buffer += text + "\n\n"
        end_idx = len(self.buffer)
        self.page_ranges.append((start_idx, end_idx, page_num))
        self.current_page = page_num

        return self._extract_chunks()

    def _extract_chunks(self) -> List[Dict]:
        """버퍼에서 완성된 청크 추출"""
        chunks = []

        while len(self.buffer) >= self.chunk_size:
            chunk_text = self.buffer[:self.chunk_size]

            # 문장 경계에서 자르기
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            cut_point = max(last_period, last_newline)

            if cut_point > self.chunk_size * 0.7:
                chunk_text = chunk_text[:cut_point + 1]
                actual_end = cut_point + 1
            else:
                actual_end = self.chunk_size

            if chunk_text.strip():
                page = self._get_page_for_position(0, actual_end)
                chunk_id = f"{self.file_name}_p{page}_{self.chunk_idx}"

                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text.strip(),
                    "metadata": {
                        "source": self.file_name,
                        "page": page
                    }
                })
                self.chunk_idx += 1

            # 버퍼에서 청크 제거 (오버랩 유지)
            remove_len = actual_end - self.chunk_overlap
            if remove_len > 0:
                self.buffer = self.buffer[remove_len:]
                # page_ranges 업데이트
                self.page_ranges = [
                    (max(0, s - remove_len), max(0, e - remove_len), p)
                    for s, e, p in self.page_ranges
                    if e > remove_len
                ]
            else:
                break

        return chunks

    def _get_page_for_position(self, start: int, end: int) -> int:
        """위치에 해당하는 페이지 번호 반환"""
        for ps, pe, pn in self.page_ranges:
            if ps < end and pe > start:
                return pn
        return self.current_page

    def flush(self) -> List[Dict]:
        """남은 버퍼를 청크로 변환"""
        chunks = []

        if self.buffer.strip():
            page = self._get_page_for_position(0, len(self.buffer))
            chunk_id = f"{self.file_name}_p{page}_{self.chunk_idx}"

            chunks.append({
                "id": chunk_id,
                "content": self.buffer.strip(),
                "metadata": {
                    "source": self.file_name,
                    "page": page
                }
            })
            self.chunk_idx += 1
            self.buffer = ""

        return chunks


class RagIngestionService:
    def __init__(self):
        self._markitdown = MarkItDown() if MarkItDown else None

    # ========================================
    # 페이지 단위 PDF 처리 (제너레이터)
    # ========================================

    def _iter_pdf_pages(self, pdf_path: str) -> Generator[Tuple[int, str], None, None]:
        """PDF를 페이지 단위로 yield하는 제너레이터"""
        if pdf4llm is None:
            logger.error("pdf4llm 라이브러리가 설치되지 않았습니다.")
            return

        try:
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            for page_idx, page_data in enumerate(pages_data):
                page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                if content_text.strip():
                    yield page_num, content_text.strip()
        except Exception as e:
            logger.exception(f"pdf4llm 변환 중 예외 발생: {e}")

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

    # ========================================
    # 배치 처리 (임베딩 + 인덱싱)
    # ========================================

    async def _process_batch(
        self,
        chunks: List[Dict],
        invoke_id: str,
        on_progress=None,
    ) -> int:
        """
        청크를 처리합니다 (임베딩 + 인덱싱).

        Args:
            chunks: 청크 리스트
            invoke_id: 인덱스 ID
            on_progress: 진행률 콜백 (percent, message)

        Returns: 처리된 청크 수
        """
        if not chunks:
            return 0

        # 임베딩 요청 (vLLM이 자체 배칭/메모리 관리)
        total_chunks = len(chunks)
        texts = [c["content"] for c in chunks]

        logger.info(f"[Ingestion] 임베딩 요청: {total_chunks}개")
        if on_progress:
            await on_progress(40, f"임베딩 중... ({total_chunks}개)")

        all_embeddings = await model_server_client.embed_texts(texts, is_query=False)

        # 임베딩 개수 검증
        if len(all_embeddings) != total_chunks:
            raise ValueError(
                f"Embedding count mismatch: expected {total_chunks}, got {len(all_embeddings)}"
            )

        # 임베딩 결과 할당
        for chunk, emb in zip(chunks, all_embeddings):
            chunk["embedding"] = emb

        # Qdrant에 저장
        indexed_count = await qdrant_service.upsert_documents(invoke_id, chunks)

        logger.info(f"[Ingestion] {len(chunks)}개 처리 완료 (Qdrant: {indexed_count})")
        return indexed_count

    # ========================================
    # 메인 Ingestion (스트리밍 방식)
    # ========================================

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, on_progress=None, on_markdown=None):
        """
        파일을 처리합니다 (청킹 → 임베딩 → 인덱싱).
        """
        logger.info(f"[Ingestion] 파일 처리 시작: {file_name} (Room: {invoke_id})")

        if on_progress:
            await on_progress(0, "파일 처리 시작")

        display_name = file_name
        _, ext = os.path.splitext(display_name)
        ext_to_use = ext.lower().strip()

        # PDF 경로 결정 (HWP는 먼저 PDF로 변환)
        pdf_path = None
        temp_pdf_dir = None

        try:
            # === 1. 파일 타입별 전처리 ===
            # HWP는 Polaris 설정과 무관하게 항상 Polaris로 처리
            if ext_to_use in ['.hwp', '.hwpx']:
                logger.info(f"[Ingestion] HWP 파일은 Polaris로 처리: {file_name}")
                await self._ingest_file_legacy(file_name, invoke_id, file_path, on_progress, on_markdown)
                return

            if settings.POLARIS_ENABLED:
                # Polaris 활성화 시 모든 문서를 Polaris로 처리
                await self._ingest_file_legacy(file_name, invoke_id, file_path, on_progress, on_markdown)
                return

            if ext_to_use == '.pdf':
                pdf_path = file_path

            else:
                # 기타 문서는 기존 방식 (MarkItDown 등)
                await self._ingest_file_legacy(file_name, invoke_id, file_path, on_progress, on_markdown)
                return

            # === 2. PDF 처리 ===
            if on_progress:
                await on_progress(10, "페이지 파싱 중...")

            chunker = IncrementalChunker(
                file_name=display_name,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            all_chunks: List[Dict] = []
            max_page = 0
            markdown_preview_parts = []

            # 페이지 단위로 청킹 (멀티프로세싱)
            loop = asyncio.get_event_loop()
            parse_start = time.time()
            page_generator = await loop.run_in_executor(
                get_pdf_process_pool(),
                _parse_pdf_in_process,
                pdf_path
            )
            parse_elapsed = time.time() - parse_start
            total_pages = len(page_generator)
            logger.info(f"[Ingestion] PDF 텍스트 추출 완료: {total_pages}페이지, {parse_elapsed:.2f}초")

            for page_num, page_text in page_generator:
                max_page = max(max_page, page_num)

                # 마크다운 프리뷰 (처음 10페이지만)
                if page_num <= 10:
                    markdown_preview_parts.append(f"\n--- Page {page_num} ---\n{page_text}")

                # 청킹
                new_chunks = chunker.add_page(page_num, page_text)
                all_chunks.extend(new_chunks)

            # 남은 청크 처리
            remaining = chunker.flush()
            all_chunks.extend(remaining)

            if on_progress:
                await on_progress(30, f"{len(all_chunks)}개 청크 생성 완료")

            # 마크다운 콜백 (처음 10페이지만)
            if on_markdown and markdown_preview_parts:
                preview = "\n".join(markdown_preview_parts)
                if total_pages > 10:
                    preview += f"\n\n... ({total_pages - 10}페이지 생략)"
                await on_markdown(preview)

            # prev/next 청크 ID 연결
            for i, chunk in enumerate(all_chunks):
                chunk["metadata"]["prev_chunk_id"] = all_chunks[i - 1]["id"] if i > 0 else None
                chunk["metadata"]["next_chunk_id"] = all_chunks[i + 1]["id"] if i < len(all_chunks) - 1 else None

            # 모든 청크 한 번에 처리
            if all_chunks:
                await self._process_batch(all_chunks, invoke_id, on_progress)

            total_chunks = len(all_chunks)

            # === 3. 완료 처리 ===
            if total_chunks == 0:
                logger.warning(f"[Ingestion] 텍스트 추출 실패 (스캔 문서?): {display_name}")
                if on_progress:
                    await on_progress(100, "텍스트 추출 실패 (스캔 문서일 수 있음)")
                return

            logger.info(f"[Ingestion] 완료: {display_name} ({max_page}페이지, {total_chunks}청크)")

            if on_progress:
                await on_progress(100, "인덱싱 완료!")

        finally:
            # 임시 PDF 디렉토리 정리
            if temp_pdf_dir and os.path.exists(temp_pdf_dir):
                import shutil
                shutil.rmtree(temp_pdf_dir, ignore_errors=True)

    async def _ingest_file_legacy(self, file_name: str, invoke_id: str, file_path: str, on_progress=None, on_markdown=None):
        """
        기존 방식의 파일 처리 (Polaris, 기타 문서용).
        전체 파일을 한 번에 처리합니다.
        """
        display_name = file_name
        _, ext = os.path.splitext(display_name)
        ext_to_use = ext.lower().strip()

        markdown_content = ""

        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")

        # HWP는 Polaris 설정과 무관하게 항상 Polaris 사용
        use_polaris = settings.POLARIS_ENABLED or ext_to_use in ['.hwp', '.hwpx']

        if use_polaris:
            temp_output_dir = tempfile.mkdtemp()
            try:
                # 멀티프로세싱으로 Polaris 변환
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
                    logger.error("[Ingestion] Polaris 변환 실패")
                    return
                logger.info(f"[Ingestion] Polaris 텍스트 추출 완료: {len(markdown_content)}자, {parse_elapsed:.2f}초")
            except Exception as e:
                logger.error(f"[Ingestion] Polaris 오류: {e}")
                return
            finally:
                # 임시 디렉토리 정리
                import shutil
                shutil.rmtree(temp_output_dir, ignore_errors=True)
        else:
            # MarkItDown 사용
            if self._markitdown is None:
                logger.error("[Ingestion] MarkItDown 미설치")
                return

            try:
                parse_start = time.time()
                result = await asyncio.to_thread(self._markitdown.convert, file_path)
                parse_elapsed = time.time() - parse_start
                if result and result.text_content:
                    markdown_content = result.text_content
                    logger.info(f"[Ingestion] MarkItDown 텍스트 추출 완료: {len(markdown_content)}자, {parse_elapsed:.2f}초")
            except Exception as e:
                logger.error(f"[Ingestion] MarkItDown 오류: {e}")
                return

        if not markdown_content:
            logger.error("[Ingestion] 마크다운 없음")
            return

        if on_markdown:
            await on_markdown(markdown_content)

        if on_progress:
            await on_progress(20, "문서 파싱 완료")

        # 청킹
        chunks = self._create_chunks_simple(markdown_content, display_name)

        if not chunks:
            logger.error("[Ingestion] 청크 생성 실패")
            return

        if on_progress:
            await on_progress(40, f"{len(chunks)}개 청크 생성 완료")

        # prev/next 청크 ID 연결
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["prev_chunk_id"] = chunks[i - 1]["id"] if i > 0 else None
            chunk["metadata"]["next_chunk_id"] = chunks[i + 1]["id"] if i < len(chunks) - 1 else None

        # 모든 청크 한 번에 처리
        total_chunks = len(chunks)
        await self._process_batch(chunks, invoke_id, on_progress)

        if on_progress:
            await on_progress(100, "인덱싱 완료!")

        logger.info(f"[Ingestion] 완료 (Legacy): {display_name} ({total_chunks}청크)")

    def _create_chunks_simple(self, markdown_content: str, file_name: str) -> List[Dict]:
        """단순 청킹 (기존 방식)"""
        if not markdown_content.strip():
            return []

        chunk_size = settings.CHUNK_SIZE
        chunk_overlap = settings.CHUNK_OVERLAP

        chunks = []
        start = 0
        text_len = len(markdown_content)
        chunk_idx = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = markdown_content[start:end]

            if end < text_len:
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                cut_point = max(last_period, last_newline)
                if cut_point > chunk_size * 0.7:
                    chunk_text = chunk_text[:cut_point + 1]
                    end = start + cut_point + 1

            if chunk_text.strip():
                chunk_id = f"{file_name}_p1_{chunk_idx}"
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text.strip(),
                    "metadata": {
                        "source": file_name,
                        "page": 1
                    }
                })
                chunk_idx += 1

            start = end - chunk_overlap if end < text_len else text_len

        return chunks


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
