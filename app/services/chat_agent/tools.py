"""
LangGraph 도구 정의

문서 검색을 LangGraph 도구로 래핑
"""

import logging
from typing import List, Dict, Any
from app.services.api_clients.model_server_client import model_server_client
from app.services.rag.qdrant_service import qdrant_service
from app.core.config import settings

logger = logging.getLogger(__name__)

# Reranker 후보 수 (Qdrant에서 가져올 개수)
RERANK_CANDIDATES = 30

# 인접 청크에서 가져올 추가 컨텍스트 크기 (글자 수)
CONTEXT_EXPAND_SIZE = 1500


class SearchTool:
    """문서 검색 도구"""

    def __init__(self, invoke_id: str):
        self.invoke_id = invoke_id

    async def search(self, query: str, top_k: int = None, filter_filename: str = None) -> Dict[str, Any]:
        """
        문서 검색 수행 (Qdrant + Reranker)

        Returns:
            {
                "context": "<documents>...</documents>" or None,
                "references": [{"source": "...", "page": N}, ...],
                "results": [raw search results]
            }
        """
        k = top_k or settings.SEARCH_TOP_K

        logger.info(f"[Search] query: {query[:50]}...")
        logger.debug(f"[Search] invoke_id: {self.invoke_id}, top_k: {k}, filter: {filter_filename}")

        # 1. 쿼리 임베딩 생성
        embeddings = await model_server_client.embed_texts([query], is_query=True)
        if not embeddings:
            return {"context": None, "references": [], "results": []}
        query_embedding = embeddings[0]

        # 2. Qdrant에서 후보 검색
        candidates = await qdrant_service.search(
            invoke_id=self.invoke_id,
            query_embedding=query_embedding,
            top_k=RERANK_CANDIDATES
        )

        logger.info(f"[Search] Qdrant 후보: {len(candidates)}개")

        if not candidates:
            return {"context": None, "references": [], "results": []}

        # 3. Reranking (인접 청크 필터링을 위해 더 많이 가져옴)
        documents = [c.content for c in candidates]
        reranked = await model_server_client.rerank(
            query=query,
            documents=documents,
            top_k=min(k * 3, len(candidates))  # 필터링 여유분
        )

        logger.info(f"[Search] Rerank 결과: {len(reranked)}개")

        # 4. 결과 조합 (인접 청크 제외)
        results = []
        selected_ids = set()  # 선택된 청크 ID
        adjacent_ids = set()  # 인접 청크 ID (제외 대상)

        for r in reranked:
            if len(results) >= k:
                break

            # # 점수 필터링 (0.6 미만은 제외)
            # score = r.get("score", 0.0)
            # if score < 0.6:
            #     logger.debug(f"[Search] 낮은 점수 스킵: {score:.3f}")
            #     continue

            orig_idx = r.get("index", 0)
            if orig_idx >= len(candidates):
                continue

            candidate = candidates[orig_idx]
            doc_id = candidate.doc_id

            # 이미 선택된 청크의 인접 청크면 스킵
            if doc_id in adjacent_ids:
                logger.debug(f"[Search] 인접 청크 스킵: {doc_id}")
                continue

            # 선택
            results.append({
                "doc_id": doc_id,
                "score": r.get("score", 0.0),
                "content": candidate.content,
                "metadata": candidate.metadata
            })
            selected_ids.add(doc_id)

            # 이 청크의 인접 청크를 제외 대상에 추가
            prev_id = candidate.metadata.get("prev_chunk_id")
            next_id = candidate.metadata.get("next_chunk_id")
            if prev_id:
                adjacent_ids.add(prev_id)
            if next_id:
                adjacent_ids.add(next_id)

        logger.info(f"[Search] 인접 청크 필터링 후: {len(results)}개")

        # 필터링
        filtered_results = []
        references = []

        for doc in results:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)

            # 파일명 필터링 (정확한 파일명 비교)
            if filter_filename and source != filter_filename:
                continue

            filtered_results.append(doc)

            ref = {"source": source, "page": page}
            if ref not in references:
                references.append(ref)

        if not filtered_results:
            return {"context": None, "references": [], "results": results}

        # 5. 인접 청크로 컨텍스트 확장
        filtered_results = await self._expand_with_adjacent_chunks(filtered_results)

        context = self._format_context(filtered_results)
        return {"context": context, "references": references, "results": filtered_results}

    async def _expand_with_adjacent_chunks(self, results: List[Dict]) -> List[Dict]:
        """
        검색 결과에 인접 청크 컨텍스트 추가

        각 청크의 앞/뒤 청크에서 OVERLAP 제외하고 1000자씩 가져와 확장
        """
        if not results:
            return results

        # 1. 필요한 인접 청크 ID 수집
        adjacent_ids = set()
        for doc in results:
            metadata = doc.get("metadata", {})
            prev_id = metadata.get("prev_chunk_id")
            next_id = metadata.get("next_chunk_id")
            if prev_id:
                adjacent_ids.add(prev_id)
            if next_id:
                adjacent_ids.add(next_id)

        if not adjacent_ids:
            return results

        # 2. 인접 청크 조회
        adjacent_chunks = await qdrant_service.get_chunks_by_ids(
            self.invoke_id, list(adjacent_ids)
        )

        logger.debug(f"[Search] 인접 청크 {len(adjacent_chunks)}개 조회")

        # 3. 각 결과에 컨텍스트 확장
        chunk_overlap = settings.CHUNK_OVERLAP
        expanded_results = []

        for doc in results:
            metadata = doc.get("metadata", {})
            prev_id = metadata.get("prev_chunk_id")
            next_id = metadata.get("next_chunk_id")
            content = doc.get("content", "")

            # 앞 청크에서 추가 (overlap 제외 후 마지막 1000자)
            prev_context = ""
            if prev_id and prev_id in adjacent_chunks:
                prev_content = adjacent_chunks[prev_id].content
                # overlap 부분 제외 (끝부분이 현재 청크와 중복)
                non_overlap = prev_content[:-chunk_overlap] if len(prev_content) > chunk_overlap else ""
                prev_context = non_overlap[-CONTEXT_EXPAND_SIZE:] if non_overlap else ""

            # 뒤 청크에서 추가 (overlap 제외 후 처음 1000자)
            next_context = ""
            if next_id and next_id in adjacent_chunks:
                next_content = adjacent_chunks[next_id].content
                # overlap 부분 제외 (앞부분이 현재 청크와 중복)
                non_overlap = next_content[chunk_overlap:] if len(next_content) > chunk_overlap else ""
                next_context = non_overlap[:CONTEXT_EXPAND_SIZE] if non_overlap else ""

            # 확장된 컨텐츠 생성
            expanded_content = ""
            if prev_context:
                expanded_content += prev_context + "\n\n"
            expanded_content += content
            if next_context:
                expanded_content += "\n\n" + next_context

            expanded_doc = doc.copy()
            expanded_doc["content"] = expanded_content
            expanded_results.append(expanded_doc)

        return expanded_results

    def _format_context(self, results: List[Dict]) -> str:
        """검색 결과를 XML 컨텍스트로 포맷팅"""
        if not results:
            return None

        valid_docs = []
        seen_contents = set()

        for doc in results:
            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)

            # 중복 제거
            content_key = content[:300]
            if content_key in seen_contents:
                continue
            seen_contents.add(content_key)

            formatted_doc = f'<document source="{source}" page="{page}">\n{content}\n</document>'
            valid_docs.append(formatted_doc)

        if not valid_docs:
            return None

        return "<documents>\n" + "\n".join(valid_docs) + "\n</documents>"

    async def search_batch(self, queries: List[str], top_k: int = None, filter_filename: str = None) -> List[Dict[str, Any]]:
        """
        여러 쿼리 배치 검색 (각 쿼리별로 search 호출)

        Returns:
            [
                {"query": "...", "context": "...", "references": [...], "results": [...]},
                ...
            ]
        """
        logger.info(f"[Batch Search] {len(queries)}개 쿼리, invoke_id: {self.invoke_id}, filter: {filter_filename}")

        all_search_results = []
        for query in queries:
            result = await self.search(query, top_k, filter_filename)
            result["query"] = query
            all_search_results.append(result)

        return all_search_results


def create_search_tool(invoke_id: str) -> SearchTool:
    """invoke_id가 바인딩된 검색 도구 인스턴스 생성"""
    return SearchTool(invoke_id)
