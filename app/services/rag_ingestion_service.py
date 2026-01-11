import httpx
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_redis import RedisVectorStore, RedisConfig
from langchain_core.documents import Document
from app.core.config import settings


class RagIngestionService:
    def __init__(self):
        self.parse_api_url = os.getenv("UNSTRUCTURED_URL", "http://localhost:8001/general/v0/general")

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
        print(f"🚀 [Ingestion] Processing for Room: {invoke_id} | File: {file_name}")
        elements = await self._call_unstructured_api(file_path)
        if not elements:
            print("❌ [Ingestion] No elements parsed from Unstructured.")
            return

        # 1. 모든 엘리먼트의 텍스트를 하나로 합침 (줄바꿈 두 번으로 구분하여 문맥 유지)
        full_text = "\n\n".join([el.get("text", "") for el in elements if el.get("text", "").strip()])

        # 2. Splitter 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

        # 3. 합쳐진 텍스트를 분할 (문자열 리스트 반환)
        chunks = text_splitter.split_text(full_text)

        # 4. 분할된 텍스트 조각들을 Document 객체로 변환
        # 이 부분에서 docs 대신 chunks를 참조하도록 수정했습니다.
        final_docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": file_name,
                    "invoke_id": [invoke_id],
                    "page": 1  # 전체를 합쳤으므로 기본값 설정
                }
            )
            for chunk in chunks
        ]

        batch_size = 100
        total_chunks = len(final_docs)
        print(f"📦 Total chunks to save (Optimized): {total_chunks}")

        for i in range(0, total_chunks, batch_size):
            batch = final_docs[i: i + batch_size]

            # ✅ 기존 데이터와 충돌을 피하기 위해 afrom_documents 대신
            # 클래스 레벨에서 인스턴스화하여 사용하거나 아래와 같이 호출
            await RedisVectorStore.afrom_documents(
                documents=batch,
                embedding=self.embeddings,
                config=self.config,
            )
            print(f"✅ [Ingestion] Progress: {min(i + batch_size, total_chunks)} / {total_chunks} chunks saved.")

        print(f"🎉 [Ingestion] Successfully saved all {total_chunks} chunks for {invoke_id}")

    async def _call_unstructured_api(self, file_path: str):
        async with httpx.AsyncClient(timeout=600.0) as client:
            filename = os.path.basename(file_path)
            with open(file_path, "rb") as f:
                files = {"files": (filename, f)}
                data = {"strategy": "fast"}
                resp = await client.post(self.parse_api_url, files=files, data=data)
                if resp.status_code != 200:
                    return []
                return resp.json()


rag_ingestion_service = RagIngestionService()
