import time
import asyncio
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

from app.services.clients.llm_client import llm_client
from app.services.clients.model_server_client import model_server_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class ExternalServiceEmbeddings(Embeddings):
    """
    외부 모델 서버를 통해 임베딩을 수행하는 LangChain 호환 Embeddings 클래스.
    """
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await model_server_client.get_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await model_server_client.get_embeddings([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """ [수정] 동기 문서 임베딩 시 동기 클라이언트 사용 """
        return model_server_client.get_embeddings_sync(texts)

    def embed_query(self, text: str) -> List[float]:
        """ [수정] 동기 질의 임베딩 시 동기 클라이언트 사용 """
        embeddings = model_server_client.get_embeddings_sync([text])
        return embeddings[0]


class RagSearchService:
    def __init__(self):
        # 1) 임베딩 모델 (외부 API 사용)
        self.embedding_model = ExternalServiceEmbeddings()

        # 2) 리랭커 모델 (제거, 외부 API 사용)

        self.vector_db = None
        self.retriever = None

        try:
            db_root = Path("Wiki_Full_Index")
            part_paths = sorted(db_root.glob("index_part_*"))

            if not part_paths:
                raise FileNotFoundError(f"FAISS index_part_* not found under: {db_root}")

            # 첫 파트 로드 (외부 임베딩 모델 사용)
            self.vector_db = FAISS.load_local(
                str(part_paths[0]),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )

            # 나머지 병합
            for path in part_paths[1:]:
                try:
                    temp_db = FAISS.load_local(
                        str(path),
                        self.embedding_model,
                        allow_dangerous_deserialization=True,
                    )
                    self.vector_db.merge_from(temp_db)
                except Exception as e:
                    logger.warning("Failed to merge FAISS part: %s (%s)", path, e)

            # Retriever 설정
            self.retriever = self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 50, "fetch_k": 200, "lambda_mult": 0.3},
            )

            logger.info("✅ RAG system ready (External API for models)")

        except Exception as e:
            logger.exception("❌ RAG init failed: %s", e)
            self.vector_db = None
            self.retriever = None

    async def _retrieve(self, query: str) -> List[Document]:
        """Retriever를 사용하여 초기 문서 목록을 가져옵니다."""
        if self.retriever is None:
            return []

        if hasattr(self.retriever, "ainvoke"):
            return await self.retriever.ainvoke(query)
        
        return await asyncio.to_thread(self.retriever.invoke, query)

    async def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """외부 모델 서버를 사용하여 Reranking 수행"""
        if not docs:
            return []

        doc_contents = [doc.page_content for doc in docs]
        doc_map = {doc.page_content: doc for doc in docs}

        reranked_results = await model_server_client.rerank(query, doc_contents)

        # 재랭킹된 순서대로 Document 객체 목록 재구성
        final_docs = []
        for result in reranked_results:
            content = result["text"] # 'document' 대신 'text' 키 사용
            if content in doc_map:
                final_docs.append(doc_map[content])
        
        return final_docs[:5] # 상위 5개 반환

    async def search(self, query: str) -> 'AsyncGenerator[bytes, None]':
        """
        사용자 질문(query)과 관련된 문서를 검색하고, LLM의 답변을 스트리밍으로 반환합니다.
        """
        if self.retriever is None:
            # 스트리밍 환경에서 오류를 어떻게 처리할지 고려해야 함.
            # 여기서는 간단한 텍스트를 스트림으로 보내는 것으로 임시 처리.
            async def error_stream():
                yield b'data: {"error": "RAG retriever is not ready."}\n\n'
            return error_stream()

        initial_docs = await self._retrieve(query)
        if not initial_docs:
            async def error_stream():
                yield b'data: {"error": "No relevant documents found."}\n\n'
            return error_stream()

        final_docs = await self._rerank(query, initial_docs)
        if not final_docs:
            async def error_stream():
                yield b'data: {"error": "No relevant documents found after reranking."}\n\n'
            return error_stream()

        # 컨텍스트 구성
        context_chunks = [f"[문서 {i}]: {doc.page_content}" for i, doc in enumerate(final_docs, start=1)]
        context = "\n\n".join(context_chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    "너는 (주)울타리정보통신에서 만든 업무보조형 AI 비서 챗봇이야.\n"
                    "너의 주 업무는 사람들의 요청에 전문적인 어투로 예의바르게 대응하는거야.\n"
                    "다음은 사용자의 질문과 관련된 문서 내용이야. 각 내용 상단에 [문서]가 명시되어 있어.\n"
                    f"[참고 자료]\n{context}\n"
                ),
            },
            {"role": "user", "content": query},
        ]

        payload = {
            "model": settings.VLLM_MODEL,
            "messages": messages,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }
        
        # LLM 스트리밍 호출
        return await llm_client.chat_completions_stream(payload)


# 인스턴스 생성
rag_service = RagSearchService()