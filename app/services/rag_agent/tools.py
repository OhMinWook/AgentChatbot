"""
LangGraph 도구 정의

ColBERT 검색을 LangGraph 도구로 래핑
- 모델 서버: 인코딩만 담당 (Stateless)
- 게이트웨이: local_index_service로 검색
"""

from typing import List, Dict, Any
from app.services.rag.local_index_service import local_index_service
from app.core.config import settings


class ColBERTSearchTool:
    """ColBERT 검색 도구"""

    def __init__(self, invoke_id: str):
        self.invoke_id = invoke_id

    async def search(self, query: str, top_k: int = None, filter_filename: str = None) -> Dict[str, Any]:
        """
        ColBERT 검색 수행 (로컬 Voyager 인덱스 사용)

        Returns:
            {
                "context": "<documents>...</documents>" or None,
                "references": [{"source": "...", "page": N}, ...],
                "results": [raw search results]
            }
        """
        k = top_k or settings.COLBERT_TOP_K

        print(f"🔍 [Search] query: {query[:50]}...")
        print(f"🔍 [Search] invoke_id: {self.invoke_id}, top_k: {k}, filter: {filter_filename}")

        # 로컬 인덱스에서 검색
        search_results = await local_index_service.search(
            invoke_id=self.invoke_id,
            query=query,
            top_k=k
        )

        print(f"🔍 [Search] {len(search_results)}개 결과")

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

        # 결과 필터링 및 포맷팅
        valid_docs = []
        references = []
        seen_contents = set()

        for doc in results:
            score = doc.get("score", 0)
            print(f"🔍 [Score] {score:.2f}")

            # score 임계값 (필요시 조정)
            if score < 17.0:
                continue

            content = doc.get("content", "").strip()
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", 0)

            # [파일명 필터링]
            if filter_filename:
                if filter_filename not in source and source not in filter_filename:
                    continue

            # 중복 제거
            content_key = content[:300]
            if content_key in seen_contents:
                continue
            seen_contents.add(content_key)

            # 문서 포맷팅 (키워드 정보는 content에 이미 포함됨)
            formatted_doc = f'<document source="{source}" page="{page}">\n{content}\n</document>'
            valid_docs.append(formatted_doc)

            ref = {"source": source, "page": page}
            if ref not in references:
                references.append(ref)

        if not valid_docs:
            return {"context": None, "references": [], "results": results}

        xml_context = "<documents>\n" + "\n".join(valid_docs) + "\n</documents>"
        return {"context": xml_context, "references": references, "results": results}


    async def search_batch(self, queries: List[str], top_k: int = None, filter_filename: str = None) -> List[Dict[str, Any]]:
        """
        여러 쿼리 배치 검색 (로컬 Voyager 인덱스 사용)

        Returns:
            [
                {"query": "...", "context": "...", "references": [...]},
                ...
            ]
        """
        k = top_k or settings.COLBERT_TOP_K

        print(f"🔍 [Batch Search] {len(queries)}개 쿼리, invoke_id: {self.invoke_id}, filter: {filter_filename}")

        # 로컬 인덱스에서 배치 검색
        batch_results = await local_index_service.search_batch(
            invoke_id=self.invoke_id,
            queries=queries,
            top_k=k
        )

        print(f"🔍 [Batch Search] {len(batch_results)}개 결과 수신")

        all_search_results = []

        for idx, search_results in enumerate(batch_results):
            query = queries[idx] if idx < len(queries) else ""

            if not search_results:
                all_search_results.append({
                    "query": query,
                    "context": None,
                    "references": []
                })
                continue

            valid_docs = []
            references = []
            seen_contents = set()

            for r in search_results:
                score = r.score
                if score < 17.0:
                    continue

                content = r.content.strip()
                source = r.metadata.get("source", "unknown")
                page = r.metadata.get("page", 0)

                # [파일명 필터링]
                if filter_filename:
                    if filter_filename not in source and source not in filter_filename:
                        continue

                content_key = content[:300]
                if content_key in seen_contents:
                    continue
                seen_contents.add(content_key)

                formatted_doc = f'<document source="{source}" page="{page}">\n{content}\n</document>'
                valid_docs.append(formatted_doc)

                ref = {"source": source, "page": page}
                if ref not in references:
                    references.append(ref)

            if valid_docs:
                xml_context = "<documents>\n" + "\n".join(valid_docs) + "\n</documents>"
                all_search_results.append({
                    "query": query,
                    "context": xml_context,
                    "references": references
                })
            else:
                all_search_results.append({
                    "query": query,
                    "context": None,
                    "references": []
                })

        return all_search_results


def create_search_tool(invoke_id: str) -> ColBERTSearchTool:
    """invoke_id가 바인딩된 검색 도구 인스턴스 생성"""
    return ColBERTSearchTool(invoke_id)
