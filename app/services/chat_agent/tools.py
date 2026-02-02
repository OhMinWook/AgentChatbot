"""
LangGraph 도구 정의

ColBERT 검색을 LangGraph 도구로 래핑
"""

import logging
from typing import List, Dict, Any
from app.services.api_clients.model_server_client import model_server_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class ColBERTSearchTool:
    """ColBERT 검색 도구"""

    def __init__(self, invoke_id: str):
        self.invoke_id = invoke_id

    async def search(self, query: str, top_k: int = None, filter_filename: str = None) -> Dict[str, Any]:
        """
        ColBERT 검색 수행

        Returns:
            {
                "context": "<documents>...</documents>" or None,
                "references": [{"source": "...", "page": N}, ...],
                "results": [raw search results]
            }
        """
        k = top_k or settings.COLBERT_TOP_K

        logger.info(f"[Search] query: {query[:50]}...")
        logger.debug(f"[Search] invoke_id: {self.invoke_id}, top_k: {k}, filter: {filter_filename}")

        search_results = await model_server_client.colbert_search(
            invoke_id=self.invoke_id,
            query=query,
            top_k=k
        )

        logger.info(f"[Search] 검색 결과: {len(search_results)}개")

        if not search_results:
            return {"context": None, "references": [], "results": []}

        # SearchResult → dict 변환
        results = [
            {
                "doc_id": r.doc_id,
                "score": r.score,
                "content": r.content,
                "metadata": r.metadata
            }
            for r in search_results
        ]

        # 필터링
        filtered_results = []
        references = []

        for doc in results:
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)

            # 파일명 필터링
            if filter_filename:
                if filter_filename not in source and source not in filter_filename:
                    continue

            filtered_results.append(doc)

            ref = {"source": source, "page": page}
            if ref not in references:
                references.append(ref)

        if not filtered_results:
            return {"context": None, "references": [], "results": results}

        context = self._format_context(filtered_results)
        return {"context": context, "references": references, "results": filtered_results}

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
        여러 쿼리 배치 검색

        Returns:
            [
                {"query": "...", "context": "...", "references": [...], "results": [...]},
                ...
            ]
        """
        k = top_k or settings.COLBERT_TOP_K

        logger.info(f"[Batch Search] {len(queries)}개 쿼리, invoke_id: {self.invoke_id}, filter: {filter_filename}")

        batch_results = await model_server_client.colbert_search_batch(
            invoke_id=self.invoke_id,
            queries=queries,
            top_k=k
        )

        logger.info(f"[Batch Search] {len(batch_results)}개 결과 수신")

        all_search_results = []

        for idx, search_results in enumerate(batch_results):
            query = queries[idx] if idx < len(queries) else ""

            if not search_results:
                all_search_results.append({
                    "query": query,
                    "context": None,
                    "references": [],
                    "results": []
                })
                continue

            # SearchResult → dict 변환
            results_dicts = []
            references = []

            for r in search_results:
                source = r.metadata.get("source", "unknown")
                page = r.metadata.get("page", 0)

                # 파일명 필터링
                if filter_filename:
                    if filter_filename not in source and source not in filter_filename:
                        continue

                results_dicts.append({
                    "doc_id": r.doc_id,
                    "score": r.score,
                    "content": r.content,
                    "metadata": r.metadata
                })

                ref = {"source": source, "page": page}
                if ref not in references:
                    references.append(ref)

            context = self._format_context(results_dicts) if results_dicts else None

            all_search_results.append({
                "query": query,
                "context": context,
                "references": references,
                "results": results_dicts
            })

        return all_search_results


def create_search_tool(invoke_id: str) -> ColBERTSearchTool:
    """invoke_id가 바인딩된 검색 도구 인스턴스 생성"""
    return ColBERTSearchTool(invoke_id)
