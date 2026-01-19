import os
import pymupdf4llm
from markitdown import MarkItDown
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_core.documents import Document
from app.core.config import settings


class RagIngestionService:
    # 지원하는 파일 확장자
    PDF_EXTENSIONS = {".pdf"}
    MARKITDOWN_EXTENSIONS = {
        ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
        ".html", ".htm", ".txt", ".csv", ".json", ".xml"
    }

    def __init__(self):
        self.markitdown = MarkItDown()

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        self.index_name = "otinus_knowledge_index"

        # ✅ 임베딩 차원(필수) 계산
        dims = len(self.embeddings.embed_query("dim_probe"))

        # ✅ 메타데이터 스키마 명시 (invoke_id는 TAG)
        self.config = RedisConfig.with_metadata_schema(
            [
                {"name": "invoke_id", "type": "tag"},
                {"name": "source", "type": "text"},
                {"name": "page", "type": "numeric"},
            ],
            index_name=self.index_name,
            redis_url=self.redis_url,
            embedding_dimensions=dims,
            distance_metric="COSINE",
        )

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str):
        print(f"[Ingestion] Processing for Room: {invoke_id} | File: {file_name}")

        # 1. 파일 확장자에 따라 적절한 파서로 마크다운 추출
        markdown_content = self._extract_markdown(file_path)
        if not markdown_content or not markdown_content.strip():
            print("[Ingestion] No content extracted from file.")
            return

        # 2. 마크다운 헤더 기반 1차 분할 (구조 보존)
        headers_to_split = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
        md_docs = md_splitter.split_text(markdown_content)

        # 3. 큰 청크는 추가로 분할 (chunk_size 초과 시)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        final_docs = []

        for doc in md_docs:
            if len(doc.page_content) > 500:
                # 큰 청크는 추가 분할
                sub_chunks = text_splitter.split_text(doc.page_content)
                for chunk in sub_chunks:
                    final_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": file_name,
                                "invoke_id": [invoke_id],
                                **doc.metadata  # 헤더 정보 유지
                            }
                        )
                    )
            else:
                final_docs.append(
                    Document(
                        page_content=doc.page_content,
                        metadata={
                            "source": file_name,
                            "invoke_id": [invoke_id],
                            **doc.metadata
                        }
                    )
                )

        if not final_docs:
            print("[Ingestion] No chunks generated after splitting.")
            return

        batch_size = 100
        total_chunks = len(final_docs)
        print(f"[Ingestion] Total chunks to save: {total_chunks}")

        for i in range(0, total_chunks, batch_size):
            batch = final_docs[i: i + batch_size]
            await RedisVectorStore.afrom_documents(
                documents=batch,
                embedding=self.embeddings,
                config=self.config,
            )
            print(f"[Ingestion] Progress: {min(i + batch_size, total_chunks)} / {total_chunks} chunks saved.")

        print(f"[Ingestion] Successfully saved all {total_chunks} chunks for {invoke_id}")

    def _extract_markdown(self, file_path: str) -> str:
        """파일 확장자에 따라 적절한 파서로 마크다운 추출"""
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext in self.PDF_EXTENSIONS:
                # PDF는 PyMuPDF4LLM으로 처리 (테이블, 헤더, 리스트 구조 보존)
                return pymupdf4llm.to_markdown(file_path)

            elif ext in self.MARKITDOWN_EXTENSIONS:
                # 나머지 문서는 MarkItDown으로 처리
                result = self.markitdown.convert(file_path)
                return result.text_content

            else:
                # 지원하지 않는 확장자는 일반 텍스트로 시도
                print(f"[Ingestion] Unknown extension {ext}, trying as plain text")
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()

        except Exception as e:
            print(f"[Ingestion] Error extracting markdown from {file_path}: {e}")
            return ""


rag_ingestion_service = RagIngestionService()
