"""
RAG Ingestion Service

파일 인제스트 오케스트레이션 (추출 → 청킹 → 임베딩 → 인덱싱)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.api_clients.llm_client import llm_client
from app.services.api_clients.model_server_client import model_server_client
from app.services.rag.chunker import build_chunks
from app.services.rag.extractors import file_text_extractor
from app.services.rag.qdrant_service import qdrant_service
from app.services.rag.sparse_encoder import sparse_encoder
from app.services.utils.llm_payload import build_chat_payload

logger = logging.getLogger(__name__)


class RagIngestionService:
    EMBED_BATCH_SIZE = 64

    def _detect_file_type(self, file_name: str) -> str:
        """파일 확장자 기반 처리 타입 결정 (polaris / pdf / markitdown)"""
        _, ext = os.path.splitext(file_name)
        ext_lower = ext.lower().strip()
        if settings.POLARIS_ENABLED:
            return "polaris"
        if ext_lower in ['.hwp', '.hwpx']:
            return "hwp_win32"
        if ext_lower == '.pdf':
            return "pdf"
        return "markitdown"

    async def _emit_markdown_preview(
        self,
        page_results: Optional[List[Tuple[int, str]]],
        markdown_content: Optional[str],
        on_markdown,
    ) -> None:
        """마크다운 미리보기 콜백 호출"""
        if markdown_content:
            await on_markdown(markdown_content)
        elif page_results:
            preview_parts = [f"\n--- Page {pn} ---\n{pt}" for pn, pt in page_results if pn <= 10]
            preview = "\n".join(preview_parts)
            if len(page_results) > 10:
                preview += f"\n\n... ({len(page_results) - 10}페이지 생략)"
            await on_markdown(preview)

    async def _detect_document_type(self, text: str) -> str:
        """LLM으로 문서 유형을 감지한다.

        Returns:
            "structured"   — 헤더/목차/표 등 명확한 구조가 있는 문서 (매뉴얼, 규정집 등)
            "unstructured" — 연속된 문장 위주의 문서 (계약서 본문, 에세이 등)
        """
        sample = text[:1000]
        messages = [
            {
                "role": "system",
                "content": (
                    "문서의 첫 부분을 보고 유형을 판단하세요.\n"
                    "- structured: 목차, 헤더(#), 번호 체계, 표 등 명확한 구조가 있는 문서\n"
                    "- unstructured: 연속된 문장 위주로 구성된 문서\n"
                    "반드시 'structured' 또는 'unstructured' 중 하나만 출력하세요. 다른 설명은 하지 마세요."
                ),
            },
            {
                "role": "user",
                "content": f"<document>\n{sample}\n</document>",
            },
        ]
        payload = build_chat_payload(messages, max_tokens=10, temperature=0.0)
        try:
            response = await llm_client.chat_completions(payload)
            result = llm_client.extract_content(response).strip().lower()
            doc_type = "unstructured" if "unstructured" in result else "structured"
            logger.info(f"[Ingestion] 문서 유형 감지: {doc_type} (응답: {result!r})")
            return doc_type
        except Exception as e:
            logger.warning(f"[Ingestion] 문서 유형 감지 실패, structured 기본값 사용: {e}")
            return "structured"

    async def _generate_chunk_context(self, doc_text: str, chunk_content: str) -> str:
        """LLM으로 청크의 문서 내 맥락을 1~2문장으로 생성"""
        messages = [
            {
                "role": "system",
                "content": "문서 청크의 검색 품질을 높이기 위한 짧은 맥락을 1~2문장으로 작성하세요. 다른 설명 없이 맥락만 출력하세요."
            },
            {
                "role": "user",
                "content": f"<document>\n{doc_text}\n</document>\n\n<chunk>\n{chunk_content}\n</chunk>"
            }
        ]
        payload = build_chat_payload(messages, max_tokens=150)
        response = await llm_client.chat_completions(payload)
        return llm_client.extract_content(response).strip()

    async def _add_context_to_chunks(self, doc_text: str, chunks: List[Dict], on_progress=None) -> List[Dict]:
        """청크에 LLM 생성 맥락을 붙임 (병렬 배치 처리)"""
        limited_doc = doc_text[:settings.CONTEXTUAL_RETRIEVAL_MAX_DOC_CHARS]
        batch_size = settings.CONTEXTUAL_RETRIEVAL_BATCH_SIZE
        total = len(chunks)

        logger.info(f"[Contextual] 컨텍스트 생성 시작: {total}개 청크")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            async def add_context(chunk):
                try:
                    context = await self._generate_chunk_context(limited_doc, chunk["content"])
                    if context:
                        chunk["content"] = f"{context}\n{chunk['content']}"
                except Exception as e:
                    logger.warning(f"[Contextual] 컨텍스트 생성 실패 (스킵): {e}")
                return chunk

            await asyncio.gather(*[add_context(c) for c in batch])

            processed = min(i + batch_size, total)
            logger.info(f"[Contextual] {processed}/{total}개 완료")
            if on_progress:
                pct = 30 + int(8 * processed / total)  # 30~38% 구간
                await on_progress(pct, f"컨텍스트 생성 중... ({processed}/{total})")

        return chunks

    async def _process_batch(
        self,
        chunks: List[Dict],
        invoke_id: str,
        on_progress=None,
    ) -> int:
        """청크를 처리합니다 (Dense + Sparse 임베딩 + 인덱싱).

        Returns: 처리된 청크 수
        """
        if not chunks:
            return 0

        total_chunks = len(chunks)
        texts = [c["content"] for c in chunks]

        logger.info(f"[Ingestion] 임베딩 요청: {total_chunks}개 (Dense + Sparse, batch={self.EMBED_BATCH_SIZE})")
        if on_progress:
            await on_progress(40, f"임베딩 중... ({total_chunks}개)")

        sparse_vectors = sparse_encoder.encode_batch(texts)

        all_embeddings = []
        for i in range(0, len(texts), self.EMBED_BATCH_SIZE):
            batch_texts = texts[i:i + self.EMBED_BATCH_SIZE]
            batch_embeddings = await model_server_client.embed_texts(batch_texts, is_query=False)
            all_embeddings.extend(batch_embeddings)

        if len(all_embeddings) != total_chunks:
            raise ValueError(
                f"Embedding count mismatch: expected {total_chunks}, got {len(all_embeddings)}"
            )

        for chunk, emb, sparse in zip(chunks, all_embeddings, sparse_vectors):
            chunk["embedding"] = emb
            chunk["sparse"] = sparse

        indexed_count = await qdrant_service.upsert_documents(invoke_id, chunks)

        logger.info(f"[Ingestion] {len(chunks)}개 처리 완료 (Hybrid, Qdrant: {indexed_count})")
        return indexed_count

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, on_progress=None, on_markdown=None):
        """파일을 처리합니다 (청킹 → 임베딩 → 인덱싱)."""
        logger.info(f"[Ingestion] 파일 처리 시작: {file_name} (Room: {invoke_id})")
        if on_progress:
            await on_progress(0, "파일 처리 시작")

        # 1. 파일 타입 결정
        file_type = self._detect_file_type(file_name)

        # 2. 텍스트 추출
        page_results, markdown_content = await file_text_extractor.extract_text(
            file_path, file_type, on_progress
        )

        # 3. 추출 결과 검증
        if not page_results and not markdown_content:
            logger.warning(f"[Ingestion] 텍스트 추출 실패 (스캔 문서?): {file_name}")
            if on_progress:
                await on_progress(100, "텍스트 추출 실패 (스캔 문서일 수 있음)")
            return

        # 4. 마크다운 미리보기 콜백
        if on_markdown:
            await self._emit_markdown_preview(page_results, markdown_content, on_markdown)
        if on_progress:
            await on_progress(20, "문서 파싱 완료")

        # 5. 문서 유형 감지 (LLM)
        raw_text = markdown_content or "\n".join(t for _, t in (page_results or []))
        doc_type = await self._detect_document_type(raw_text)

        # 6. 청킹 + prev/next 연결
        all_chunks, max_page = await build_chunks(file_name, page_results, markdown_content, doc_type)
        if on_progress:
            await on_progress(30, f"{len(all_chunks)}개 청크 생성 완료")

        if not all_chunks:
            logger.warning(f"[Ingestion] 청크 생성 실패: {file_name}")
            return

        # 7. Contextual Retrieval (설정 시 활성화)
        if settings.CONTEXTUAL_RETRIEVAL_ENABLED:
            all_chunks = await self._add_context_to_chunks(raw_text, all_chunks, on_progress)

        # 8. 임베딩 + 인덱싱
        await self._process_batch(all_chunks, invoke_id, on_progress)

        logger.info(f"[Ingestion] 완료: {file_name} ({max_page}페이지, {len(all_chunks)}청크)")
        if on_progress:
            await on_progress(100, "인덱싱 완료!")


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
