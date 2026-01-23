import asyncio
import tempfile
import logging
import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple

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
from app.services.clients.model_server_client import model_server_client
from app.services.clients.polaris_client import polaris_converter, PolarisJsonParser
from app.services.rag.parent_chunk_store import parent_chunk_store

logger = logging.getLogger(__name__)


class RagIngestionService:
    def __init__(self):
        # MarkItDown 인스턴스 재사용 (매번 생성하지 않음)
        self._markitdown = MarkItDown() if MarkItDown else None

    def _create_parent_child_chunks(
        self,
        markdown_content: str,
        file_name: str,
        invoke_id: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Parent-Child 청킹 전략으로 청크를 생성합니다.

        Parent 청크: 큰 컨텍스트 (LLM 답변 생성용) - Redis 저장
        Child 청크: 작은 청크 (ColBERT 검색용) - ColBERT 인덱싱

        Args:
            markdown_content: 마크다운 텍스트
            file_name: 파일명
            invoke_id: 사용자/세션 ID

        Returns:
            (parent_chunks, child_chunks) 튜플
        """
        if not markdown_content.strip():
            return [], []

        # 설정값 로드
        parent_min = settings.PARENT_MIN_SIZE
        parent_max = settings.PARENT_MAX_SIZE
        child_size = settings.CHILD_CHUNK_SIZE
        child_overlap = settings.CHILD_CHUNK_OVERLAP

        # 페이지 구분자로 페이지별 텍스트와 페이지 번호 매핑 생성
        page_pattern = re.compile(r'\n--- Page (\d+) ---\n')
        parts = page_pattern.split(markdown_content)

        # 페이지별 텍스트 수집
        page_texts = []
        current_page = 1

        if parts[0].strip():
            page_texts.append((current_page, parts[0].strip()))

        for i in range(1, len(parts), 2):
            try:
                page_num = int(parts[i])
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if content:
                    page_texts.append((page_num, content))
            except (IndexError, ValueError):
                continue

        if not page_texts:
            return [], []

        # === Parent 청크 생성 ===
        # 페이지들을 parent_min ~ parent_max 범위로 그룹화
        parent_chunks = []
        current_parent_text = ""
        current_parent_pages = []

        for page_num, page_text in page_texts:
            test_text = current_parent_text + "\n\n" + page_text if current_parent_text else page_text

            if len(test_text) > parent_max and current_parent_text:
                # 현재 Parent 청크 저장
                parent_id = str(uuid.uuid4())
                start_page = current_parent_pages[0] if current_parent_pages else page_num
                end_page = current_parent_pages[-1] if current_parent_pages else page_num

                parent_chunks.append({
                    "parent_id": parent_id,
                    "content": f"[문서: {file_name}]\n{current_parent_text.strip()}",
                    "metadata": {
                        "source": file_name,
                        "page_start": start_page,
                        "page_end": end_page
                    }
                })

                # 새 Parent 시작
                current_parent_text = page_text
                current_parent_pages = [page_num]
            else:
                current_parent_text = test_text
                current_parent_pages.append(page_num)

        # 마지막 Parent 청크
        if current_parent_text.strip():
            parent_id = str(uuid.uuid4())
            start_page = current_parent_pages[0] if current_parent_pages else 1
            end_page = current_parent_pages[-1] if current_parent_pages else 1

            parent_chunks.append({
                "parent_id": parent_id,
                "content": f"[문서: {file_name}]\n{current_parent_text.strip()}",
                "metadata": {
                    "source": file_name,
                    "page_start": start_page,
                    "page_end": end_page
                }
            })

        # === Child 청크 생성 ===
        # 각 Parent를 작은 Child 청크로 분할
        child_chunks = []

        for parent in parent_chunks:
            parent_id = parent["parent_id"]
            parent_content = parent["content"]
            parent_meta = parent["metadata"]

            # [문서: ...] 접두사 제거 후 분할
            clean_content = parent_content
            if parent_content.startswith("[문서:"):
                newline_idx = parent_content.find("\n")
                if newline_idx != -1:
                    clean_content = parent_content[newline_idx + 1:].strip()

            # 고정 크기로 분할
            start = 0
            text_length = len(clean_content)

            while start < text_length:
                end = start + child_size
                chunk_text = clean_content[start:end]

                # 단어 중간에서 자르지 않도록 조정
                if end < text_length:
                    last_space = chunk_text.rfind(' ')
                    if last_space > child_size * 0.5:
                        chunk_text = chunk_text[:last_space]
                        end = start + last_space

                if chunk_text.strip():
                    content_with_title = f"[문서: {file_name}]\n{chunk_text.strip()}"
                    child_chunks.append({
                        "id": str(uuid.uuid4()),
                        "content": content_with_title,
                        "metadata": {
                            "source": file_name,
                            "page": parent_meta.get("page_start", 1),
                            "parent_id": parent_id  # Parent 연결
                        }
                    })

                start = end - child_overlap if end < text_length else text_length

        logger.info(
            f"[Parent-Child Chunking] {file_name}: "
            f"Parent {len(parent_chunks)}개, Child {len(child_chunks)}개 생성"
        )

        return parent_chunks, child_chunks

    def _create_chunks_with_splitting(self, markdown_content: str, file_name: str, invoke_id: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
        """
        마크다운 텍스트를 고정 크기로 분할하여 청크를 생성합니다.
        페이지 정보가 있으면 메타데이터에 포함합니다.

        Args:
            chunk_size: 청크 크기 (기본 800자)
            chunk_overlap: 청크 간 겹침 (기본 100자)
        Returns: List of dicts with 'id', 'content', 'metadata'
        """
        if not markdown_content.strip():
            logger.warning("추출된 마크다운 내용이 없습니다.")
            return []

        # 페이지 구분자로 페이지별 텍스트와 페이지 번호 매핑 생성
        page_pattern = re.compile(r'\n--- Page (\d+) ---\n')
        parts = page_pattern.split(markdown_content)

        # 페이지별 텍스트 수집 (페이지 번호 -> 텍스트)
        page_texts = []
        current_page = 1

        if parts[0].strip():
            page_texts.append((current_page, parts[0].strip()))

        for i in range(1, len(parts), 2):
            try:
                page_num = int(parts[i])
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if content:
                    page_texts.append((page_num, content))
            except (IndexError, ValueError):
                continue

        # 전체 텍스트 합치기 (페이지 구분 없이)
        full_text = "\n\n".join([text for _, text in page_texts])

        if not full_text.strip():
            return []

        # 고정 크기로 청크 분할
        chunks = []
        start = 0
        text_length = len(full_text)

        while start < text_length:
            end = start + chunk_size
            chunk_text = full_text[start:end]

            # 단어 중간에서 자르지 않도록 조정 (마지막 청크가 아닌 경우)
            if end < text_length:
                # 마지막 공백 위치 찾기
                last_space = chunk_text.rfind(' ')
                if last_space > chunk_size * 0.5:  # 청크의 50% 이상이면 그 위치에서 자름
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space

            if chunk_text.strip():
                # 해당 청크가 어느 페이지에 속하는지 추정
                chunk_start_pos = start
                estimated_page = 1
                cumulative_length = 0
                for page_num, page_text in page_texts:
                    cumulative_length += len(page_text) + 2  # +2 for "\n\n"
                    if chunk_start_pos < cumulative_length:
                        estimated_page = page_num
                        break

                content_with_title = f"[문서: {file_name}]\n{chunk_text.strip()}"
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "content": content_with_title,
                    "metadata": {
                        "source": file_name,
                        "page": estimated_page
                    }
                })

            # 다음 청크 시작 위치 (overlap 적용)
            start = end - chunk_overlap if end < text_length else text_length

        logger.info(f"[Chunking] {file_name}: {len(chunks)}개 청크 생성 (크기: {chunk_size}, 겹침: {chunk_overlap})")
        return chunks

    def _process_polaris_json(self, polaris_data: Dict[str, Any], file_name: str, invoke_id: str) -> List[Dict]:
        """Polaris에서 추출한 JSON 데이터를 청크 목록으로 변환합니다."""
        parser = PolarisJsonParser(polaris_data)
        markdown_content = parser.parse_to_markdown()
        return self._create_chunks_with_splitting(markdown_content, file_name, invoke_id)

    def _process_pdf_with_pdf4llm(self, pdf_path: str, display_name: str, invoke_id: str) -> List[Dict]:
        """pdf4llm을 사용하여 PDF 파일을 청크 목록으로 변환합니다."""
        if pdf4llm is None:
            logger.error("pdf4llm 라이브러리가 설치되지 않았습니다.")
            return []

        try:
            # 페이지별로 추출 후 페이지 구분자와 함께 합침
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            markdown_parts = []
            for page_idx, page_data in enumerate(pages_data):
                real_page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                if content_text.strip():
                    markdown_parts.append(f"\n--- Page {real_page_num} ---\n{content_text.strip()}")

            full_markdown = "\n".join(markdown_parts)
            return self._create_chunks_with_splitting(full_markdown, display_name, invoke_id)
        except Exception as e:
            logger.exception(f"pdf4llm 변환 중 예외 발생: {e}")
            return []

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

    async def _store_chunks_to_colbert(self, chunks: List[Dict], invoke_id: str):
        """
        청크 목록을 ColBERT 인덱스에 저장합니다.
        모델 서버의 /index API를 호출합니다.
        """
        if not chunks:
            return

        batch_size = 50
        total_chunks = len(chunks)
        logger.info(f"[Ingestion] ColBERT에 저장할 총 청크 수: {total_chunks}")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]

            indexed = await model_server_client.colbert_index(batch, invoke_id)
            logger.info(f"[Ingestion] 진행률: {min(i + batch_size, total_chunks)} / {total_chunks} 청크 저장 완료.")

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, file_extension: Optional[str] = None):
        """
        파일을 처리하고, 결과를 ColBERT 인덱스에 저장합니다.
        설정에 따라 Polaris 또는 대체 라이브러리(pdf4llm, MarkItDown, win32com)를 사용합니다.
        """
        logger.info(f"[Ingestion] 파일 처리 시작: {file_name} (Room: {invoke_id})")

        # 확장자 결정 및 파일명 보정
        ext_to_use = ""
        if file_extension:
            ext_to_use = file_extension.lower().strip()
            if not ext_to_use.startswith('.'):
                ext_to_use = '.' + ext_to_use
        else:
            _, ext_from_path = os.path.splitext(file_name)
            ext_to_use = ext_from_path.lower().strip()

        # 메타데이터에 저장할 파일명 (확장자가 없으면 붙여줌)
        if ext_to_use and not file_name.lower().endswith(ext_to_use):
            display_name = file_name + ext_to_use
        else:
            display_name = file_name

        chunks = []

        if settings.POLARIS_ENABLED:
            logger.info("Polaris 엔진을 사용하여 문서 변환을 시도합니다.")
            with tempfile.TemporaryDirectory() as temp_output_dir:
                try:
                    polaris_data = await asyncio.to_thread(
                        polaris_converter.convert, file_path, temp_output_dir, True
                    )
                    if polaris_data:
                        logger.info("Polaris JSON 데이터 처리 중...")
                        chunks = self._process_polaris_json(polaris_data, display_name, invoke_id)
                    else:
                        logger.error("Polaris 변환 실패 또는 반환 데이터 없음.")
                        return
                except Exception as e:
                    logger.exception(f"Polaris 변환 중 예외 발생: {e}")
                    return
        else:
            logger.info(f"사용할 확장자: '{ext_to_use}' (표시 파일명: {display_name})")

            # 1. HWP/HWPX 파일 처리
            if ext_to_use in ['.hwp', '.hwpx']:
                logger.info(f"한글 문서({ext_to_use}) 감지. win32com을 통해 PDF로 변환 후 pdf4llm을 사용합니다.")
                with tempfile.TemporaryDirectory() as temp_pdf_dir:
                    converted_pdf = await asyncio.to_thread(
                        self._convert_hwp_to_pdf_with_win32com, file_path, temp_pdf_dir
                    )
                    if converted_pdf and os.path.exists(converted_pdf):
                        logger.info(f"PDF 변환 완료: {converted_pdf}. pdf4llm으로 텍스트를 추출합니다.")
                        chunks = await asyncio.to_thread(
                            self._process_pdf_with_pdf4llm, converted_pdf, display_name, invoke_id
                        )
                        if not chunks:
                            return
                    else:
                        logger.error("HWP -> PDF 변환 실패.")
                        return

            # 2. 원래 PDF 파일
            elif ext_to_use == '.pdf':
                logger.info("PDF 파일 감지. pdf4llm을 사용하여 변환합니다.")
                chunks = await asyncio.to_thread(
                    self._process_pdf_with_pdf4llm, file_path, display_name, invoke_id
                )
                if not chunks:
                    return

            # 3. 기타 문서 (MarkItDown)
            else:
                logger.info(f"일반 문서({ext_to_use}) 감지. MarkItDown을 사용하여 변환합니다.")
                if self._markitdown is None:
                    logger.error("MarkItDown 라이브러리가 설치되지 않았습니다.")
                    return

                try:
                    result = await asyncio.to_thread(self._markitdown.convert, file_path)
                    if result and result.text_content:
                        chunks = self._create_chunks_with_splitting(result.text_content, display_name, invoke_id)
                    else:
                        logger.warning("MarkItDown 변환 결과가 비어있습니다.")
                except Exception as e:
                    logger.exception(f"MarkItDown 변환 중 예외 발생: {e}")
                    return

        if not chunks:
            logger.warning("[Ingestion] 처리 후 생성된 청크(chunk)가 없습니다.")
            return

        # Parent-Child 청킹 후 저장
        combined_markdown = "\n\n".join([c.get("content", "") for c in chunks])
        parent_chunks, child_chunks = self._create_parent_child_chunks(
            combined_markdown, display_name, invoke_id
        )

        if parent_chunks:
            await parent_chunk_store.save_batch(invoke_id, parent_chunks)
            logger.info(f"[Ingestion] Parent {len(parent_chunks)}개 Redis 저장 완료")

        if child_chunks:
            await self._store_chunks_to_colbert(child_chunks, invoke_id)
            logger.info(f"[Ingestion] Child {len(child_chunks)}개 ColBERT 저장 완료")

        logger.info(
            f"[Ingestion] {file_name} Parent-Child 청킹 완료 "
            f"(Parent: {len(parent_chunks)}, Child: {len(child_chunks)}, Room: {invoke_id})"
        )


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
