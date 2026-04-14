"""
LangGraph 도구 정의

문서 검색을 LangGraph 도구로 래핑
"""

import asyncio
import logging
from html import escape as html_escape
from typing import List, Dict, Any
from app.core.langfuse_client import observe  # langfuse 비활성화 스텁
from app.services.api_clients.model_server_client import model_server_client
from app.services.rag.qdrant_service import qdrant_service
from app.services.rag.sparse_encoder import sparse_encoder
from app.services.rag.text_utils import add_source_prefix
from app.core.config import settings
from app.services.agent_base.tool import BaseTool

logger = logging.getLogger(__name__)

# Reranker 후보 수 (Qdrant에서 가져올 개수)
RERANK_CANDIDATES = 64


class SearchTool(BaseTool):
    """문서 검색 도구 — Hybrid 검색 파이프라인 (Dense + Sparse → RRF → Reranker)"""

    def __init__(self, invoke_id: str):
        self.invoke_id = invoke_id

    @property
    def name(self) -> str:
        return "search_tool"

    async def execute(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """BaseTool 인터페이스 구현 — search()로 위임"""
        return await self.search(
            query=input_text,
            top_k=kwargs.get("top_k"),
            filter_filename=kwargs.get("filter_filename"),
        )

    async def execute_batch(self, inputs: List[str], **kwargs) -> List[Dict[str, Any]]:
        """BaseTool 인터페이스 구현 — search_batch()로 위임"""
        return await self.search_batch(
            queries=inputs,
            top_k=kwargs.get("top_k"),
            filter_filename=kwargs.get("filter_filename"),
        )

    @observe()
    async def search(self, query: str, top_k: int = None, filter_filename: str = None) -> Dict[str, Any]:
        """
        문서 검색 수행 (Hybrid: Dense + Sparse → RRF → Reranker)

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

        # 1. Dense 쿼리 임베딩 생성
        embeddings = await model_server_client.embed_texts([query], is_query=True)
        if not embeddings:
            return {"context": None, "references": [], "results": []}
        query_embedding = embeddings[0]

        # 2. Sparse 쿼리 임베딩 생성 (BM25 스타일)
        query_sparse = sparse_encoder.encode(query)

        # 3. Qdrant Hybrid 검색 (Dense + Sparse RRF Fusion)
        candidates = await qdrant_service.search_hybrid(
            invoke_id=self.invoke_id,
            dense_embedding=query_embedding,
            sparse_vector=query_sparse,
            top_k=RERANK_CANDIDATES,
            filter_source=filter_filename
        )

        logger.info(f"[Search] Qdrant 후보: {len(candidates)}개")

        if not candidates:
            return {"context": None, "references": [], "results": []}

        # 4. Reranking (인접 청크 필터링을 위해 더 많이 가져옴)
        documents = [
            add_source_prefix(c.content, c.metadata.get("source", ""))
            for c in candidates
        ]
        reranked = await model_server_client.rerank(
            query=query,
            documents=documents,
            top_k=min(k * 3, len(candidates))  # 필터링 여유분
        )

        logger.info(f"[Search] Rerank 결과: {len(reranked)}개")

        # 4. 결과 조합 (인접 청크 제외)
        results = self._rerank_and_filter(candidates, reranked, k)
        logger.info(f"[Search] 인접 청크 필터링 후: {len(results)}개")

        if not results:
            return {"context": None, "references": [], "results": []}

        # 5. 참조 목록 생성
        references = []
        for doc in results:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)
            # open chat은 페이지 정보 없이 제공
            ref = {"source": source, "page": page if filter_filename else None}
            if ref not in references:
                references.append(ref)

        # 6. 결과가 2개 이하면 다음(오른쪽) 청크 2개씩 추가 (문맥 보강, 점수 무관)
        if len(results) <= 2:
            results = await self._append_following_chunks(results, count=2)

        # 7. 인접 청크로 컨텍스트 확장
        results = await self._expand_with_adjacent_chunks(results)

        context = self._format_context(results)
        return {"context": context, "references": references, "results": results}

    def _rerank_and_filter(self, candidates, reranked, k: int) -> List[Dict]:
        """리랭크 결과에서 인접 청크 제외하며 top-k 선택"""
        results = []
        selected_ids = set()
        adjacent_ids = set()

        for r in reranked:
            if len(results) >= k:
                break

            # 점수 필터링 (최소 N개 보장, 이후 임계값 미만 제외)
            score = r.get("score", 0.0)
            if len(results) >= settings.RERANK_MIN_RESULTS and score < settings.RERANK_SCORE_THRESHOLD:
                logger.debug(f"[Search] 낮은 점수 스킵: {score:.3f}")
                continue

            orig_idx = r.get("index", 0)
            if orig_idx >= len(candidates):
                continue

            candidate = candidates[orig_idx]
            doc_id = candidate.doc_id

            # 이미 선택된 청크의 인접 청크면 스킵 (최소 결과 수 보장 후에만 적용)
            if doc_id in adjacent_ids and len(results) >= settings.RERANK_MIN_RESULTS:
                logger.debug(f"[Search] 인접 청크 스킵: {doc_id}")
                continue

            results.append({
                "doc_id": doc_id,
                "score": score,
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

        return results

    async def _append_following_chunks(self, results: List[Dict], count: int = 2) -> List[Dict]:
        """
        결과가 적을 때 다음(오른쪽) 청크 내용을 현재 청크에 덧붙임 (문맥 보강)

        각 청크의 next_chunk_id를 따라가며 count개 청크의 내용을 content에 추가
        """
        if not results:
            return results

        # 1. 모든 next_chunk_id 수집 (체인 따라가기 위해 일단 첫 번째만)
        chunks_to_fetch = set()
        for doc in results:
            next_id = doc.get("metadata", {}).get("next_chunk_id")
            if next_id:
                chunks_to_fetch.add(next_id)

        if not chunks_to_fetch:
            return results

        # 2. 첫 번째 다음 청크들 조회
        fetched_chunks = await qdrant_service.get_chunks_by_ids(
            self.invoke_id, list(chunks_to_fetch)
        )

        # 3. 두 번째 다음 청크들도 조회 (count=2인 경우)
        if count >= 2:
            second_level_ids = set()
            for chunk in fetched_chunks.values():
                next_next_id = chunk.metadata.get("next_chunk_id")
                if next_next_id and next_next_id not in fetched_chunks:
                    second_level_ids.add(next_next_id)

            if second_level_ids:
                second_chunks = await qdrant_service.get_chunks_by_ids(
                    self.invoke_id, list(second_level_ids)
                )
                fetched_chunks.update(second_chunks)

        logger.info(f"[Search] 다음 청크 {len(fetched_chunks)}개 조회 (문맥 보강)")

        # 4. 각 결과의 content에 다음 청크 내용 덧붙이기
        expanded_results = []
        chunk_overlap = settings.CHUNK_OVERLAP

        for doc in results:
            content = doc.get("content", "")
            next_id = doc.get("metadata", {}).get("next_chunk_id")

            # 다음 청크들 내용 수집
            appended_content = ""
            for _ in range(count):
                if next_id and next_id in fetched_chunks:
                    next_chunk = fetched_chunks[next_id]
                    # overlap 부분 제외 (앞부분이 현재 청크와 중복)
                    next_content = next_chunk.content
                    non_overlap = next_content[chunk_overlap:] if len(next_content) > chunk_overlap else next_content
                    appended_content += "\n\n" + non_overlap
                    next_id = next_chunk.metadata.get("next_chunk_id")
                else:
                    break

            expanded_doc = doc.copy()
            expanded_doc["content"] = content + appended_content
            expanded_results.append(expanded_doc)

        return expanded_results

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
                prev_context = non_overlap[-settings.CONTEXT_EXPAND_SIZE:] if non_overlap else ""

            # 뒤 청크에서 추가 (overlap 제외 후 처음 1000자)
            next_context = ""
            if next_id and next_id in adjacent_chunks:
                next_content = adjacent_chunks[next_id].content
                # overlap 부분 제외 (앞부분이 현재 청크와 중복)
                non_overlap = next_content[chunk_overlap:] if len(next_content) > chunk_overlap else ""
                next_context = non_overlap[:settings.CONTEXT_EXPAND_SIZE] if non_overlap else ""

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

            # XML 특수문자 이스케이프
            escaped_content = html_escape(content, quote=False)
            escaped_source = html_escape(str(source), quote=True)
            formatted_doc = f'<document source="{escaped_source}" page="{page}">\n{escaped_content}\n</document>'
            valid_docs.append(formatted_doc)

        if not valid_docs:
            return None

        return "<documents>\n" + "\n".join(valid_docs) + "\n</documents>"

    async def search_batch(self, queries: List[str], top_k: int = None, filter_filename: str = None) -> List[Dict[str, Any]]:
        """
        여러 쿼리 배치 검색 (병렬 처리)

        Returns:
            [
                {"query": "...", "context": "...", "references": [...], "results": [...]},
                ...
            ]
        """
        logger.info(f"[Batch Search] {len(queries)}개 쿼리, invoke_id: {self.invoke_id}, filter: {filter_filename}")

        async def search_with_query(query: str) -> Dict[str, Any]:
            result = await self.search(query, top_k, filter_filename)
            result["query"] = query
            return result

        all_search_results = await asyncio.gather(*[search_with_query(q) for q in queries])
        return list(all_search_results)


def create_search_tool(invoke_id: str) -> SearchTool:
    """invoke_id가 바인딩된 검색 도구 인스턴스 생성"""
    return SearchTool(invoke_id)
