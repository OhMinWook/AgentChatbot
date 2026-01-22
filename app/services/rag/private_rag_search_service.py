from typing import List, Dict, Set
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector, NamedSparseVector, SparseVector,
    Filter, FieldCondition, MatchValue,
    SearchRequest, ScoredPoint
)

from app.core.config import settings
from app.services.clients.model_server_client import model_server_client


class PrivateRagSearchService:
    def __init__(self):
        # Qdrant 클라이언트 초기화
        self.qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[ScoredPoint],
        sparse_results: List[ScoredPoint],
        k: int = 60
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion)을 사용하여 두 검색 결과를 통합합니다.

        RRF 공식: score = sum(1 / (k + rank_i)) for each list

        Args:
            dense_results: Dense 검색 결과
            sparse_results: Sparse 검색 결과
            k: RRF 상수 (기본값 60, 논문 권장값)

        Returns:
            RRF 점수로 정렬된 문서 리스트 (중복 제거됨)
        """
        # point_id -> (rrf_score, payload, content)
        fused_scores: Dict[str, Dict] = {}

        # Dense 결과 처리
        for rank, point in enumerate(dense_results, start=1):
            point_id = str(point.id)
            rrf_score = 1.0 / (k + rank)

            if point_id not in fused_scores:
                fused_scores[point_id] = {
                    "rrf_score": 0.0,
                    "payload": point.payload,
                    "dense_score": point.score,
                    "sparse_score": 0.0
                }
            fused_scores[point_id]["rrf_score"] += rrf_score
            fused_scores[point_id]["dense_score"] = point.score

        # Sparse 결과 처리
        for rank, point in enumerate(sparse_results, start=1):
            point_id = str(point.id)
            rrf_score = 1.0 / (k + rank)

            if point_id not in fused_scores:
                fused_scores[point_id] = {
                    "rrf_score": 0.0,
                    "payload": point.payload,
                    "dense_score": 0.0,
                    "sparse_score": point.score
                }
            fused_scores[point_id]["rrf_score"] += rrf_score
            fused_scores[point_id]["sparse_score"] = point.score

        # RRF 점수로 정렬
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1]["rrf_score"],
            reverse=True
        )

        return [
            {
                "id": point_id,
                "rrf_score": data["rrf_score"],
                "dense_score": data["dense_score"],
                "sparse_score": data["sparse_score"],
                "payload": data["payload"]
            }
            for point_id, data in sorted_results
        ]

    async def _rerank(self, query: str, docs: List[Dict]) -> List[Dict]:
        """외부 모델 서버를 사용하여 Reranking 수행"""
        if not docs:
            return []

        # API에 전달할 문서 내용 목록
        doc_contents = [doc["payload"]["content"] for doc in docs]

        # 외부 API 호출
        reranked_results = await model_server_client.rerank(query, doc_contents)

        # 결과 재구성: index 기반으로 원본 doc과 매칭
        final_docs = []
        for result in reranked_results:
            idx = result["index"]
            score = result["score"]

            doc = docs[idx].copy()
            doc["rerank_score"] = score
            final_docs.append(doc)

        # Rerank 점수로 정렬
        final_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_docs = final_docs[:settings.RERANK_TOP_K]

        # 디버그: rerank 결과 출력
        print(f"\n{'='*60}")
        print(f"🔍 [Rerank 결과] 상위 {len(top_docs)}개 문서")
        print(f"{'='*60}")
        for idx, doc in enumerate(top_docs):
            score = doc.get("rerank_score", 0)
            source = doc["payload"].get("source", "unknown")
            page = doc["payload"].get("page", 0)
            content_preview = doc["payload"]["content"][:200].replace('\n', ' ')
            print(f"\n[{idx+1}] Rerank: {score:.4f} | RRF: {doc['rrf_score']:.4f}")
            print(f"    출처: {source} (p.{page})")
            print(f"    내용: {content_preview}...")
        print(f"{'='*60}\n")

        return top_docs

    async def search(self, query: str, invoke_id: str) -> dict:
        """
        [Hybrid Search + RRF + Reranking]

        1. Dense 검색: Top 50
        2. Sparse 검색: Top 50
        3. RRF Fusion: 중복 제거 후 통합 (약 70~80개)
        4. Reranking: ko-reranker로 재정렬
        5. 최종: Top 5 반환

        Returns: {"context": "...", "references": [...]}
        """
        try:
            # 쿼리 임베딩 (dense + sparse)
            dense_embeddings, sparse_embeddings = await model_server_client.get_hybrid_embeddings([query])
            query_dense = dense_embeddings[0]
            query_sparse = sparse_embeddings[0]

            # invoke_id 필터
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="invoke_id",
                        match=MatchValue(value=invoke_id)
                    )
                ]
            )

            print(f"🔒 [Hybrid Search] invoke_id='{invoke_id}', query='{query[:50]}...'")

            # 1. Dense 검색 (Top 50)
            dense_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=NamedVector(
                    name="dense",
                    vector=query_dense
                ),
                query_filter=search_filter,
                limit=settings.DENSE_TOP_K,
                with_payload=True
            )
            print(f"   Dense 결과: {len(dense_results)}개")

            # 2. Sparse 검색 (Top 50)
            sparse_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=NamedSparseVector(
                    name="sparse",
                    vector=SparseVector(
                        indices=query_sparse["indices"],
                        values=query_sparse["values"]
                    )
                ),
                query_filter=search_filter,
                limit=settings.SPARSE_TOP_K,
                with_payload=True
            )
            print(f"   Sparse 결과: {len(sparse_results)}개")

            if not dense_results and not sparse_results:
                return {"context": None, "references": []}

            # 3. RRF Fusion
            fused_docs = self._reciprocal_rank_fusion(dense_results, sparse_results)
            print(f"   RRF 통합 결과: {len(fused_docs)}개 (중복 제거)")

            if not fused_docs:
                return {"context": None, "references": []}

            # 4. Reranking
            reranked_docs = await self._rerank(query, fused_docs)

            # 5. 결과 포맷팅
            valid_docs = []
            references = []
            seen_contents: Set[str] = set()

            for doc in reranked_docs:
                content = doc["payload"]["content"].strip()
                score = doc.get("rerank_score", 0)

                # 점수가 0.01 이상인 경우에만 사용
                if score < 0.01:
                    continue

                if content not in seen_contents:
                    source_file = doc["payload"].get("source", "unknown")
                    page_num = doc["payload"].get("page", 0)

                    # 임베딩용 [문서: ...] 접두사 제거
                    clean_content = content
                    if content.startswith("[문서:"):
                        newline_idx = content.find("\n")
                        if newline_idx != -1:
                            clean_content = content[newline_idx + 1:].strip()

                    # XML 태그 형식으로 문서 포맷팅
                    formatted_doc = f'<document source="{source_file}" page="{page_num}">\n{clean_content}\n</document>'
                    valid_docs.append(formatted_doc)

                    # 참조 정보 추가 (중복 제거)
                    ref_info = {"source": source_file, "page": page_num}
                    if ref_info not in references:
                        references.append(ref_info)

                    seen_contents.add(content)

            if not valid_docs:
                return {"context": None, "references": []}

            # XML 구조로 감싸서 반환
            xml_context = "<documents>\n" + "\n".join(valid_docs) + "\n</documents>"
            return {
                "context": xml_context,
                "references": references
            }

        except Exception as e:
            print(f"🔥 [Search Error]: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"context": "검색 중 오류가 발생했습니다.", "references": []}


# 싱글톤 인스턴스
private_rag_service = PrivateRagSearchService()
