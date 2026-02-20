"""
RAG Ingestion Service

파일 인제스트 오케스트레이션 (추출 → 청킹 → 임베딩 → 인덱싱)
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.api_clients.model_server_client import model_server_client
from app.services.rag.chunker import build_chunks
from app.services.rag.extractors import file_text_extractor
from app.services.rag.qdrant_service import qdrant_service
from app.services.rag.sparse_encoder import sparse_encoder

logger = logging.getLogger(__name__)


class RagIngestionService:
    EMBED_BATCH_SIZE = 64

    def _detect_file_type(self, file_name: str) -> str:
        """파일 확장자 기반 처리 타입 결정 (polaris / pdf / markitdown)"""
        _, ext = os.path.splitext(file_name)
        ext_lower = ext.lower().strip()
        if settings.POLARIS_ENABLED or ext_lower in ['.hwp', '.hwpx']:
            return "polaris"
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

        # 5. 청킹 + prev/next 연결
        all_chunks, max_page = build_chunks(file_name, page_results, markdown_content)
        if on_progress:
            await on_progress(30, f"{len(all_chunks)}개 청크 생성 완료")

        if not all_chunks:
            logger.warning(f"[Ingestion] 청크 생성 실패: {file_name}")
            return

        # 6. 임베딩 + 인덱싱
        await self._process_batch(all_chunks, invoke_id, on_progress)

        logger.info(f"[Ingestion] 완료: {file_name} ({max_page}페이지, {len(all_chunks)}청크)")
        if on_progress:
            await on_progress(100, "인덱싱 완료!")


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
