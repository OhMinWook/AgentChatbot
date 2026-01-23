from typing import List, Dict, Set

from app.core.config import settings
from app.services.clients.model_server_client import model_server_client


class PrivateRagSearchService:
    """
    ColBERT 기반 RAG 검색 서비스.
    Jina ColBERT v2의 MaxSim을 사용하여 검색합니다.
    Reranker 없이 순수 ColBERT 성능만 사용합니다.
    """

    def __init__(self):
        pass

    async def search(self, query: str, invoke_id: str) -> dict:
        """
        [ColBERT Search]

        1. 모델 서버의 /search API 호출 (MaxSim)
        2. 결과 포맷팅 후 반환

        Returns: {"context": "...", "references": [...]}
        """
        try:
            print(f"🔍 [ColBERT Search] invoke_id='{invoke_id}', query='{query[:50]}...'")

            # ColBERT 검색 (MaxSim)
            results = await model_server_client.colbert_search(
                query=query,
                invoke_id=invoke_id,
                top_k=settings.COLBERT_TOP_K
            )

            print(f"   검색 결과: {len(results)}개")

            if not results:
                return {"context": None, "references": []}

            # 결과 출력
            print(f"\n{'='*60}")
            print(f"🔍 [ColBERT 결과] 상위 {len(results)}개 문서")
            print(f"{'='*60}")
            for idx, doc in enumerate(results):
                score = doc.get("score", 0)
                source = doc.get("metadata", {}).get("source", "unknown")
                page = doc.get("metadata", {}).get("page", 0)
                content_preview = doc.get("content", "")[:200].replace('\n', ' ')
                print(f"\n[{idx+1}] Score: {score:.4f}")
                print(f"    출처: {source} (p.{page})")
                print(f"    내용: {content_preview}...")
            print(f"{'='*60}\n")

            # 결과 포맷팅
            valid_docs = []
            references = []
            seen_contents: Set[str] = set()

            for doc in results:
                # 점수가 17점 미만이면 사용하지 않음
                score = doc.get("score", 0)
                if score < 17.0:
                    continue

                content = doc.get("content", "").strip()

                if content in seen_contents:
                    continue

                source_file = doc.get("metadata", {}).get("source", "unknown")
                page_num = doc.get("metadata", {}).get("page", 0)

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
