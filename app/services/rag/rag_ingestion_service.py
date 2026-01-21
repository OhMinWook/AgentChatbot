import asyncio
import tempfile
import logging
import os
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

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

class ExternalServiceEmbeddings(Embeddings):
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await model_server_client.get_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await model_server_client.get_embeddings([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return model_server_client.get_embeddings_sync(texts)

    def embed_query(self, text: str) -> List[float]:
        embeddings = model_server_client.get_embeddings_sync([text])
        return embeddings[0]


class RagIngestionService:
    def __init__(self):
        self.embeddings = ExternalServiceEmbeddings()
        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        self.index_name = settings.RAG_INDEX_NAME
        self.config = RedisConfig.with_metadata_schema(
            [
                {"name": "invoke_id", "type": "tag"},
                {"name": "source", "type": "text"},
                {"name": "page", "type": "numeric"}, # 실제 문서 페이지 번호
            ],
            index_name=self.index_name,
            redis_url=self.redis_url,
            embedding_dimensions=settings.EMBEDDING_DIMS,
            distance_metric="COSINE",
        )
        # MarkItDown 인스턴스 재사용 (매번 생성하지 않음)
        self._markitdown = MarkItDown() if MarkItDown else None

    def _split_text_recursive(self, text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """단순 텍스트를 재귀적으로 분할하여 텍스트 리스트로 반환"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    def _create_chunks_with_page_tracking(self, markdown_content: str, file_name: str, invoke_id: str, start_page: int = 1) -> List[Document]:
        """
        마크다운 텍스트 내의 페이지 구분자(--- Page N ---)를 감지하여
        청크마다 올바른 페이지 번호를 부여합니다.
        구분자가 없으면 전체를 start_page로 간주합니다.
        """
        if not markdown_content.strip():
            logger.warning("추출된 마크다운 내용이 없습니다.")
            return []

        # 1. 페이지 구분자로 텍스트 분할
        page_pattern = re.compile(r'\n--- Page (\d+) ---\n')
        
        parts = page_pattern.split(markdown_content)
        
        final_docs = []
        
        # 첫 번째 파트 처리
        current_page = start_page
        if parts[0].strip():
            chunks = self._split_text_recursive(parts[0])
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_name,
                        "invoke_id": [invoke_id],
                        "page": current_page
                    }
                )
                final_docs.append(doc)

        for i in range(1, len(parts), 2):
            try:
                page_num_str = parts[i]
                content = parts[i+1]
                current_page = int(page_num_str)
            except (IndexError, ValueError):
                continue

            if content.strip():
                chunks = self._split_text_recursive(content)
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "invoke_id": [invoke_id],
                            "page": current_page
                        }
                    )
                    final_docs.append(doc)
        
        return final_docs

    def _process_polaris_json(self, polaris_data: Dict[str, Any], file_name: str, invoke_id: str) -> List[Document]:
        """Polaris에서 추출한 JSON 데이터를 LangChain Document 목록으로 변환합니다."""
        parser = PolarisJsonParser(polaris_data)
        markdown_content = parser.parse_to_markdown()
        return self._create_chunks_with_page_tracking(markdown_content, file_name, invoke_id)

    def _process_pdf_with_pdf4llm(self, pdf_path: str, display_name: str, invoke_id: str) -> List[Document]:
        """pdf4llm을 사용하여 PDF 파일을 Document 목록으로 변환합니다."""
        if pdf4llm is None:
            logger.error("pdf4llm 라이브러리가 설치되지 않았습니다.")
            return []

        docs = []
        try:
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            for page_idx, page_data in enumerate(pages_data):
                real_page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                chunks = self._split_text_recursive(content_text)
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": display_name,
                            "invoke_id": [invoke_id],
                            "page": real_page_num
                        }
                    ))
        except Exception as e:
            logger.exception(f"pdf4llm 변환 중 예외 발생: {e}")
        return docs

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

            # gencache 사용 시도, 실패하면 일반 Dispatch 사용
            try:
                hwp = win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
            except Exception:
                hwp = win32com.client.Dispatch("HWPFrame.HwpObject")

            # 보안 모듈 등록 (파일 접근 허용)
            hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")

            # HWP 파일 열기 (Format과 arg 파라미터 필요)
            # arg: "forceopen:true" - 경고 무시하고 열기
            hwp.Open(input_abs_path, "HWP", "forceopen:true")

            # PDF로 저장 (SaveAs 메서드 사용)
            # Format: "PDF" 또는 "EXPORT:PDF"
            # 한글 버전에 따라 다른 방식 시도
            success = False

            # 방법 1: HAction.Run 사용
            try:
                hwp.HAction.GetDefault("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
                hwp.HParameterSet.HFileOpenSave.filename = output_pdf_path
                hwp.HParameterSet.HFileOpenSave.Format = "PDF"
                success = hwp.HAction.Execute("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
            except Exception as e1:
                logger.warning(f"방법 1 (FileSaveAsPdf) 실패: {e1}")

                # 방법 2: SaveAs 직접 사용
                try:
                    hwp.SaveAs(output_pdf_path, "PDF")
                    success = os.path.exists(output_pdf_path)
                except Exception as e2:
                    logger.warning(f"방법 2 (SaveAs) 실패: {e2}")

                    # 방법 3: XHwpDocuments 인터페이스 사용
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

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, file_extension: Optional[str] = None):
        """
        파일을 처리하고, 결과를 Redis에 저장합니다.
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

        final_docs = []

        if settings.POLARIS_ENABLED:
            logger.info("Polaris 엔진을 사용하여 문서 변환을 시도합니다.")
            with tempfile.TemporaryDirectory() as temp_output_dir:
                try:
                    polaris_data = await asyncio.to_thread(
                        polaris_converter.convert, file_path, temp_output_dir, True
                    )
                    if polaris_data:
                        logger.info("Polaris JSON 데이터 처리 중...")
                        final_docs = self._process_polaris_json(polaris_data, display_name, invoke_id)
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
                        final_docs = await asyncio.to_thread(
                            self._process_pdf_with_pdf4llm, converted_pdf, display_name, invoke_id
                        )
                        if not final_docs:
                            return
                    else:
                        logger.error("HWP -> PDF 변환 실패.")
                        return

            # 2. 원래 PDF 파일
            elif ext_to_use == '.pdf':
                logger.info("PDF 파일 감지. pdf4llm을 사용하여 변환합니다.")
                final_docs = await asyncio.to_thread(
                    self._process_pdf_with_pdf4llm, file_path, display_name, invoke_id
                )
                if not final_docs:
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
                        final_docs = self._create_chunks_with_page_tracking(result.text_content, display_name, invoke_id)
                    else:
                        logger.warning("MarkItDown 변환 결과가 비어있습니다.")
                except Exception as e:
                    logger.exception(f"MarkItDown 변환 중 예외 발생: {e}")
                    return

        if not final_docs:
            logger.warning("[Ingestion] 처리 후 생성된 청크(chunk)가 없습니다.")
            return

        batch_size = 100
        total_chunks = len(final_docs)
        logger.info(f"[Ingestion] Redis에 저장할 총 청크 수: {total_chunks}")

        for i in range(0, total_chunks, batch_size):
            batch = final_docs[i:i + batch_size]
            await RedisVectorStore.afrom_documents(
                documents=batch,
                embedding=self.embeddings,
                config=self.config,
            )
            logger.info(f"[Ingestion] 진행률: {min(i + batch_size, total_chunks)} / {total_chunks} 청크 저장 완료.")

        logger.info(f"[Ingestion] {file_name}에 대한 모든 {total_chunks}개 청크 저장 성공 (Room: {invoke_id})")

# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()