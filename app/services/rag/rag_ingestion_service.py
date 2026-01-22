import asyncio
import tempfile
import logging
import os
import re
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance,
    PointStruct, SparseVector,
    NamedVector, NamedSparseVector
)

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

logger = logging.getLogger(__name__)


class RagIngestionService:
    def __init__(self):
        # Qdrant 클라이언트 초기화
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # MarkItDown 인스턴스 재사용 (매번 생성하지 않음)
        self._markitdown = MarkItDown() if MarkItDown else None

        # Collection 초기화
        self._ensure_collection()

    def _ensure_collection(self):
        """Qdrant Collection이 없으면 생성합니다."""
        collections = self.qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.EMBEDDING_DIMS,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                }
            )
            logger.info(f"Collection '{self.collection_name}' created successfully.")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def _create_chunks_with_page_tracking(self, markdown_content: str, file_name: str, invoke_id: str, start_page: int = 1) -> List[Dict]:
        """
        마크다운 텍스트 내의 페이지 구분자(--- Page N ---)를 감지하여
        페이지 단위로 청크를 생성합니다.
        Returns: List of dicts with 'content', 'source', 'invoke_id', 'page'
        """
        if not markdown_content.strip():
            logger.warning("추출된 마크다운 내용이 없습니다.")
            return []

        # 페이지 구분자로 텍스트 분할
        page_pattern = re.compile(r'\n--- Page (\d+) ---\n')
        parts = page_pattern.split(markdown_content)

        chunks = []

        # 첫 번째 파트 처리 (구분자 이전 내용)
        current_page = start_page
        if parts[0].strip():
            content_with_title = f"[문서: {file_name}]\n{parts[0].strip()}"
            chunks.append({
                "content": content_with_title,
                "source": file_name,
                "invoke_id": invoke_id,
                "page": current_page
            })

        # 이후 페이지들 처리
        for i in range(1, len(parts), 2):
            try:
                page_num_str = parts[i]
                content = parts[i+1]
                current_page = int(page_num_str)
            except (IndexError, ValueError):
                continue

            if content.strip():
                content_with_title = f"[문서: {file_name}]\n{content.strip()}"
                chunks.append({
                    "content": content_with_title,
                    "source": file_name,
                    "invoke_id": invoke_id,
                    "page": current_page
                })

        return chunks

    def _process_polaris_json(self, polaris_data: Dict[str, Any], file_name: str, invoke_id: str) -> List[Dict]:
        """Polaris에서 추출한 JSON 데이터를 청크 목록으로 변환합니다."""
        parser = PolarisJsonParser(polaris_data)
        markdown_content = parser.parse_to_markdown()
        return self._create_chunks_with_page_tracking(markdown_content, file_name, invoke_id)

    def _process_pdf_with_pdf4llm(self, pdf_path: str, display_name: str, invoke_id: str) -> List[Dict]:
        """pdf4llm을 사용하여 PDF 파일을 페이지 단위로 청크 목록으로 변환합니다."""
        if pdf4llm is None:
            logger.error("pdf4llm 라이브러리가 설치되지 않았습니다.")
            return []

        chunks = []
        try:
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            for page_idx, page_data in enumerate(pages_data):
                real_page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                if content_text.strip():
                    content_with_title = f"[문서: {display_name}]\n{content_text.strip()}"
                    chunks.append({
                        "content": content_with_title,
                        "source": display_name,
                        "invoke_id": invoke_id,
                        "page": real_page_num
                    })
        except Exception as e:
            logger.exception(f"pdf4llm 변환 중 예외 발생: {e}")
        return chunks

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

    async def _store_chunks_to_qdrant(self, chunks: List[Dict]):
        """
        청크 목록을 Qdrant에 저장합니다.
        각 청크에 대해 dense + sparse embedding을 생성하여 저장합니다.
        """
        if not chunks:
            return

        batch_size = 50  # 임베딩 API 호출 배치 크기
        total_chunks = len(chunks)
        logger.info(f"[Ingestion] Qdrant에 저장할 총 청크 수: {total_chunks}")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [chunk["content"] for chunk in batch]

            # Dense + Sparse embedding 동시 획득
            dense_embeddings, sparse_embeddings = await model_server_client.get_hybrid_embeddings(texts)

            # Qdrant Point 생성
            points = []
            for j, chunk in enumerate(batch):
                point_id = str(uuid.uuid4())

                # Sparse vector 변환
                sparse_data = sparse_embeddings[j]
                sparse_vector = SparseVector(
                    indices=sparse_data["indices"],
                    values=sparse_data["values"]
                )

                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_embeddings[j],
                        "sparse": sparse_vector
                    },
                    payload={
                        "content": chunk["content"],
                        "source": chunk["source"],
                        "invoke_id": chunk["invoke_id"],
                        "page": chunk["page"]
                    }
                )
                points.append(point)

            # Qdrant에 upsert
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"[Ingestion] 진행률: {min(i + batch_size, total_chunks)} / {total_chunks} 청크 저장 완료.")

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, file_extension: Optional[str] = None):
        """
        파일을 처리하고, 결과를 Qdrant에 저장합니다.
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
                        chunks = self._create_chunks_with_page_tracking(result.text_content, display_name, invoke_id)
                    else:
                        logger.warning("MarkItDown 변환 결과가 비어있습니다.")
                except Exception as e:
                    logger.exception(f"MarkItDown 변환 중 예외 발생: {e}")
                    return

        if not chunks:
            logger.warning("[Ingestion] 처리 후 생성된 청크(chunk)가 없습니다.")
            return

        # Qdrant에 저장
        await self._store_chunks_to_qdrant(chunks)
        logger.info(f"[Ingestion] {file_name}에 대한 모든 {len(chunks)}개 청크 저장 성공 (Room: {invoke_id})")


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
