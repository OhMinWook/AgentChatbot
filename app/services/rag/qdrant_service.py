"""
Qdrant Vector Database Service

문서 임베딩 저장 및 벡터 검색을 담당합니다.
"""

import logging
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과"""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class QdrantService:
    """
    Qdrant 벡터 DB 서비스

    - invoke_id별로 문서 관리 (payload filter 사용)
    - 단일 컬렉션에 모든 문서 저장
    """

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
        return self._client

    async def ensure_collection(self) -> None:
        """컬렉션이 없으면 생성 (Hybrid 지원: dense + sparse vectors)"""
        try:
            await self.client.get_collection(settings.QDRANT_COLLECTION)
            logger.info(f"[Qdrant] Collection '{settings.QDRANT_COLLECTION}' exists")
        except UnexpectedResponse:
            await self.client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,  # BM25 스타일
                    )
                },
            )
            # invoke_id 필드에 인덱스 생성 (필터 검색 성능 향상)
            await self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="invoke_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"[Qdrant] Collection '{settings.QDRANT_COLLECTION}' created (Hybrid)")

    async def upsert_documents(
        self,
        invoke_id: str,
        documents: List[Dict],
    ) -> int:
        """
        문서 upsert (doc_id 기준으로 덮어쓰기, Hybrid vectors)

        Args:
            invoke_id: 룸/세션 ID
            documents: [{"id": str, "content": str, "embedding": List[float], "sparse": dict, "metadata": dict}, ...]
                       sparse: {"indices": [int, ...], "values": [float, ...]}

        Returns:
            저장된 문서 수
        """
        if not documents:
            return 0

        await self.ensure_collection()

        points = []
        for doc in documents:
            # point_id는 invoke_id + doc_id 조합으로 결정적 UUID 생성
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{invoke_id}:{doc['id']}"))

            # Sparse vector 구성
            sparse_data = doc.get("sparse", {"indices": [], "values": []})

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": doc["embedding"],
                    "sparse": models.SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"],
                    )
                },
                payload={
                    "invoke_id": invoke_id,
                    "doc_id": doc["id"],
                    "content": doc["content"],
                    "source": doc.get("metadata", {}).get("source", ""),
                    "page": doc.get("metadata", {}).get("page", 0),
                    "prev_chunk_id": doc.get("metadata", {}).get("prev_chunk_id"),
                    "next_chunk_id": doc.get("metadata", {}).get("next_chunk_id"),
                }
            ))

        # Qdrant upsert (동일 ID면 덮어쓰기)
        await self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )

        logger.info(f"[Qdrant] Upserted {len(points)} documents (hybrid) for invoke_id={invoke_id}")
        return len(points)

    async def search(
        self,
        invoke_id: str,
        query_embedding: List[float],
        top_k: int = 30,
        filter_source: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        벡터 검색 (invoke_id로 필터링) - Dense only (fallback용)

        Args:
            invoke_id: 룸/세션 ID
            query_embedding: 쿼리 임베딩 벡터
            top_k: 반환할 결과 수
            filter_source: 특정 파일명으로 필터링 (Private chat용)

        Returns:
            SearchResult 리스트 (score 내림차순)
        """
        try:
            # 필터 조건 구성
            filter_conditions = [
                models.FieldCondition(
                    key="invoke_id",
                    match=models.MatchValue(value=invoke_id),
                )
            ]

            # 파일명 필터 (Private chat)
            if filter_source:
                filter_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=filter_source),
                    )
                )

            # 글로벌 문서: is_use=False만 제외 (True 또는 필드 없음은 통과)
            if invoke_id == settings.GLOBAL_INVOKE_ID:
                filter_conditions.append(
                    models.Filter(
                        must_not=[
                            models.FieldCondition(
                                key="is_use",
                                match=models.MatchValue(value=False),
                            )
                        ]
                    )
                )

            results = await self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=query_embedding,
                using="dense",
                query_filter=models.Filter(must=filter_conditions),
                limit=top_k,
            )

            return [
                SearchResult(
                    doc_id=hit.payload.get("doc_id", ""),
                    score=hit.score,
                    content=hit.payload.get("content", ""),
                    metadata={
                        "source": hit.payload.get("source", ""),
                        "page": hit.payload.get("page", 0),
                        "prev_chunk_id": hit.payload.get("prev_chunk_id"),
                        "next_chunk_id": hit.payload.get("next_chunk_id"),
                    }
                )
                for hit in results.points
            ]
        except UnexpectedResponse as e:
            logger.error(f"[Qdrant] Search failed: {e}")
            return []

    async def search_hybrid(
        self,
        invoke_id: str,
        dense_embedding: List[float],
        sparse_vector: Dict[str, List],
        top_k: int = 30,
        filter_source: Optional[str] = None,
        dense_weight: float = 0.8,
    ) -> List[SearchResult]:
        """
        Hybrid 검색 (Dense + Sparse 가중 합계, 정규화 적용)

        Args:
            invoke_id: 룸/세션 ID
            dense_embedding: Dense 쿼리 임베딩 벡터
            sparse_vector: Sparse 쿼리 벡터 {"indices": [...], "values": [...]}
            top_k: 반환할 결과 수
            filter_source: 특정 파일명으로 필터링 (Private chat용)
            dense_weight: Dense 가중치 (기본 0.8, Sparse는 1-dense_weight)

        Returns:
            SearchResult 리스트 (가중 합계 점수 내림차순)
        """
        sparse_weight = 1.0 - dense_weight

        try:
            # 필터 조건 구성
            filter_conditions = [
                models.FieldCondition(
                    key="invoke_id",
                    match=models.MatchValue(value=invoke_id),
                )
            ]

            # 파일명 필터 (Private chat)
            if filter_source:
                filter_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=filter_source),
                    )
                )

            # 글로벌 문서: is_use=False만 제외
            if invoke_id == settings.GLOBAL_INVOKE_ID:
                filter_conditions.append(
                    models.Filter(
                        must_not=[
                            models.FieldCondition(
                                key="is_use",
                                match=models.MatchValue(value=False),
                            )
                        ]
                    )
                )

            query_filter = models.Filter(must=filter_conditions)

            # 1. Dense 검색
            dense_results = await self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=dense_embedding,
                using="dense",
                query_filter=query_filter,
                limit=top_k,
            )

            # 2. Sparse 검색
            sparse_results = await self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=models.SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using="sparse",
                query_filter=query_filter,
                limit=top_k,
            )

            # 3. Dense는 코사인 유사도(0~1), Sparse만 Min-Max 정규화
            dense_scores = {p.id: p.score for p in dense_results.points}

            sparse_scores = {}
            if sparse_results.points:
                scores = [p.score for p in sparse_results.points]
                min_s, max_s = min(scores), max(scores)
                range_s = max_s - min_s if max_s > min_s else 1.0
                sparse_scores = {
                    p.id: (p.score - min_s) / range_s
                    for p in sparse_results.points
                }

            # 4. 가중 합계
            all_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
            combined_scores = {}
            for pid in all_ids:
                d_score = dense_scores.get(pid, 0.0)
                s_score = sparse_scores.get(pid, 0.0)
                combined_scores[pid] = dense_weight * d_score + sparse_weight * s_score

            # 5. 정렬 및 top_k
            sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]

            # 6. payload 매핑
            payload_map = {}
            for p in dense_results.points:
                payload_map[p.id] = p.payload
            for p in sparse_results.points:
                if p.id not in payload_map:
                    payload_map[p.id] = p.payload

            return [
                SearchResult(
                    doc_id=payload_map[pid].get("doc_id", ""),
                    score=combined_scores[pid],
                    content=payload_map[pid].get("content", ""),
                    metadata={
                        "source": payload_map[pid].get("source", ""),
                        "page": payload_map[pid].get("page", 0),
                        "prev_chunk_id": payload_map[pid].get("prev_chunk_id"),
                        "next_chunk_id": payload_map[pid].get("next_chunk_id"),
                    }
                )
                for pid in sorted_ids
                if pid in payload_map
            ]
        except UnexpectedResponse as e:
            logger.error(f"[Qdrant] Hybrid search failed: {e}")
            # Fallback to dense-only search
            logger.warning("[Qdrant] Falling back to dense-only search")
            return await self.search(invoke_id, dense_embedding, top_k, filter_source)

    async def get_chunks_by_ids(
        self,
        invoke_id: str,
        doc_ids: List[str],
    ) -> Dict[str, SearchResult]:
        """
        doc_id 리스트로 청크 조회

        Args:
            invoke_id: 룸/세션 ID
            doc_ids: 조회할 doc_id 리스트

        Returns:
            {doc_id: SearchResult} 딕셔너리
        """
        if not doc_ids:
            return {}

        try:
            # doc_id 필터로 조회
            results, _ = await self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="invoke_id",
                            match=models.MatchValue(value=invoke_id),
                        ),
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchAny(any=doc_ids),
                        ),
                    ]
                ),
                limit=len(doc_ids),
                with_payload=True,
                with_vectors=False,
            )

            return {
                point.payload.get("doc_id", ""): SearchResult(
                    doc_id=point.payload.get("doc_id", ""),
                    score=0.0,
                    content=point.payload.get("content", ""),
                    metadata={
                        "source": point.payload.get("source", ""),
                        "page": point.payload.get("page", 0),
                        "prev_chunk_id": point.payload.get("prev_chunk_id"),
                        "next_chunk_id": point.payload.get("next_chunk_id"),
                    }
                )
                for point in results
            }
        except UnexpectedResponse as e:
            logger.error(f"[Qdrant] Get chunks by IDs failed: {e}")
            return {}

    async def delete_by_invoke_id(self, invoke_id: str) -> int:
        """invoke_id에 해당하는 모든 문서 삭제"""
        try:
            result = await self.client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="invoke_id",
                                match=models.MatchValue(value=invoke_id),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"[Qdrant] Deleted documents for invoke_id={invoke_id}")
            return 1  # Qdrant doesn't return count
        except UnexpectedResponse as e:
            logger.error(f"[Qdrant] Delete failed: {e}")
            return 0

    async def delete_by_source(self, invoke_id: str, source: str) -> int:
        """특정 source 파일의 문서만 삭제"""
        try:
            await self.client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="invoke_id",
                                match=models.MatchValue(value=invoke_id),
                            ),
                            models.FieldCondition(
                                key="source",
                                match=models.MatchValue(value=source),
                            ),
                        ]
                    )
                ),
            )
            logger.info(f"[Qdrant] Deleted documents for invoke_id={invoke_id}, source={source}")
            return 1
        except UnexpectedResponse as e:
            logger.error(f"[Qdrant] Delete by source failed: {e}")
            return 0

    async def get_document_list(self, invoke_id: str) -> List[Dict]:
        """invoke_id에 해당하는 문서 목록 (source별 그룹화)"""
        try:
            # scroll로 모든 문서 조회
            results, _ = await self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="invoke_id",
                            match=models.MatchValue(value=invoke_id),
                        )
                    ]
                ),
                limit=10000,  # 충분히 큰 수
                with_payload=["source", "page"],
                with_vectors=False,
            )

            # source별 그룹화
            file_stats: Dict[str, Dict] = {}
            for point in results:
                source = point.payload.get("source", "unknown")
                if source not in file_stats:
                    file_stats[source] = {
                        "file_name": source,
                        "total_chunks": 0,
                    }
                file_stats[source]["total_chunks"] += 1

            return list(file_stats.values())
        except UnexpectedResponse as e:
            logger.error(f"[Qdrant] Get document list failed: {e}")
            return []

    async def close(self):
        """클라이언트 연결 종료"""
        if self._client is not None:
            await self._client.close()
            self._client = None


# 싱글톤 인스턴스
qdrant_service = QdrantService()
