from typing import List
from app.core.config import settings


class RagSearchService:
    def __init__(self):
        # 나중에 여기서 Vector DB 클라이언트(Chroma, Qdrant 등)를 연결할 거야.
        # self.vector_db = Chroma(path=settings.VECTOR_DB_PATH)
        pass

    async def search(self, query: str) -> str:
        """
        사용자 질문(query)과 관련된 문서를 검색해서 텍스트로 반환
        """

        # ---------------------------------------------------------
        # [TODO] 실제 벡터 DB 검색 로직이 들어갈 자리
        # results = await self.vector_db.similarity_search(query, k=3)
        # context_text = "\n".join([doc.page_content for doc in results])
        # ---------------------------------------------------------

        # 지금은 기능 테스트를 위해 '가짜 검색 결과'를 리턴할게.
        # 실제로는 DB에서 꺼내온 텍스트가 들어가겠지?

        mock_context = "ㅁㅇㄴㄹ"

        return mock_context



# 인스턴스 생성
rag_service = RagSearchService()