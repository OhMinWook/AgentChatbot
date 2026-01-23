"""
LangGraph 도구 정의

ColBERT 검색 및 Parent 청크 조회를 LangGraph 도구로 래핑
"""

from typing import List, Dict, Any
from app.services.clients.model_server_client import model_server_client
from app.services.rag.parent_chunk_store import parent_chunk_store
from app.core.config import settings


class ColBERTSearchTool:
    """ColBERT 검색 도구"""

    def __init__(self, invoke_id: str):
        self.invoke_id = invoke_id

    async def search_child_chunks(self, query: str, limit: int = None) -> List[Dict[str, Any]]:
        """Child 청크 검색 (ColBERT MaxSim)"""
        top_k = limit or settings.COLBERT_TOP_K
        return await model_server_client.colbert_search(
            query=query,
            invoke_id=self.invoke_id,
            top_k=top_k
        )

    async def retrieve_parent_chunks(self, parent_ids: List[str]) -> List[Dict[str, Any]]:
        """Parent 청크 조회"""
        return await parent_chunk_store.load_many(self.invoke_id, parent_ids)

    async def search_with_parent_expansion(self, query: str, limit: int = None) -> Dict[str, Any]:
        """검색 + Parent 확장"""
        child_results = await self.search_child_chunks(query, limit)

        if not child_results:
            return {"context": None, "references": [], "child_results": []}

        # Parent ID 수집
        parent_ids = []
        for result in child_results:
            parent_id = result.get("metadata", {}).get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)

        parent_chunks = {}
        if parent_ids:
            parents = await self.retrieve_parent_chunks(parent_ids)
            parent_chunks = {p["parent_id"]: p for p in parents if p}

        # 결과 포맷팅
        valid_docs = []
        references = []
        seen_contents = set()

        for doc in child_results:
            score = doc.get("score", 0)
            if score < 17.0:
                continue

            metadata = doc.get("metadata", {})
            source_file = metadata.get("source", "unknown")
            page_num = metadata.get("page", 0)
            parent_id = metadata.get("parent_id")

            # Parent가 있으면 사용, 없으면 Child 사용
            if parent_id and parent_id in parent_chunks:
                content = parent_chunks[parent_id].get("content", "")
            else:
                content = doc.get("content", "").strip()

            content_key = content[:500]
            if content_key in seen_contents:
                continue
            seen_contents.add(content_key)

            # [문서: ...] 접두사 제거
            clean_content = content
            if content.startswith("[문서:"):
                newline_idx = content.find("\n")
                if newline_idx != -1:
                    clean_content = content[newline_idx + 1:].strip()

            formatted_doc = f'<document source="{source_file}" page="{page_num}">\n{clean_content}\n</document>'
            valid_docs.append(formatted_doc)

            ref_info = {"source": source_file, "page": page_num}
            if ref_info not in references:
                references.append(ref_info)

        if not valid_docs:
            return {"context": None, "references": [], "child_results": child_results}

        xml_context = "<documents>\n" + "\n".join(valid_docs) + "\n</documents>"
        return {"context": xml_context, "references": references, "child_results": child_results}


def create_search_tool(invoke_id: str):
    """invoke_id가 바인딩된 검색 도구 인스턴스 생성"""
    return ColBERTSearchTool(invoke_id)
