import time
import asyncio
import logging
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_community.vectorstores import FAISS

from app.services.clients.llm_client import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class RagSearchService:
    def __init__(self):
        # 1) 임베딩 모델 (CPU)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # 2) 리랭커 모델 (CPU)
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device="cpu",
        )

        self.vector_db = None
        self.retriever = None

        try:
            # ✅ Windows 경로 안전 처리 (원하면 settings로 빼세요)
            db_root = Path("Wiki_Full_Index")  # 또는 Path.cwd() / "Wiki_Full_Index"
            part_paths = sorted(db_root.glob("index_part_*"))

            if not part_paths:
                raise FileNotFoundError(f"FAISS index_part_* not found under: {db_root}")

            # 첫 파트 로드
            self.vector_db = FAISS.load_local(
                str(part_paths[0]),
                self.embedding_model,
                allow_dangerous_deserialization=True,  # 신뢰된 인덱스일 때만 유지 권장
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

            # Retriever 설정 (MMR)
            self.retriever = self.vector_db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 50, "fetch_k": 200, "lambda_mult": 0.3},
            )

            logger.info("✅ RAG system ready (CPU embedding + CPU rerank)")

        except Exception as e:
            logger.exception("❌ RAG init failed: %s", e)
            self.vector_db = None
            self.retriever = None

    async def _retrieve(self, query: str):
        """Retriever가 async 지원하면 ainvoke, 아니면 thread로 오프로딩"""
        if self.retriever is None:
            return []

        if hasattr(self.retriever, "ainvoke"):
            return await self.retriever.ainvoke(query)

        # fallback: sync invoke를 thread로
        return await asyncio.to_thread(self.retriever.invoke, query)

    async def _rerank(self, query: str, docs):
        """CrossEncoder.predict는 sync이므로 thread로 오프로딩"""
        if not docs:
            return []

        pairs = [[query, d.page_content] for d in docs]
        scores = await asyncio.to_thread(self.reranker.predict, pairs)

        top_k = 5
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [docs[i] for i in top_indices]

    async def search(self, query: str) -> str:
        """
        사용자 질문(query)과 관련된 문서를 검색하고 LLM으로 최종 답변 생성
        """
        if self.retriever is None:
            return "RAG 검색기가 준비되지 않았습니다. (인덱스 로드 실패)"
        initial_docs = await self._retrieve(query)
        if not initial_docs:
            return "관련 문서 없음"
        final_docs = await self._rerank(query, initial_docs)
        end_time = time.time()
        if not final_docs:
            return "관련 문서 없음"

        # 컨텍스트 구성
        context_chunks = []
        for i, doc in enumerate(final_docs, start=1):
            context_chunks.append(f"[문서 {i}]: {doc.page_content}")

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
        start_time = time.time()
        response_json = await llm_client.chat_completions(payload)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"실행 시간: {elapsed_time:.5f}초")
        return response_json


# 인스턴스 생성 (FastAPI라면 보통 app startup에서 1회만 생성 권장)
rag_service = RagSearchService()