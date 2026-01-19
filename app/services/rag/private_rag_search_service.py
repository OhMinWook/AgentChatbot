import asyncio
from typing import List, Optional
from langchain_redis import RedisVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from redisvl.query.filter import Tag as RedisTag
from app.core.config import settings


class PrivateRagSearchService:
    def __init__(self):
        # 1. 임베딩 모델 로드 (Ingestion 서비스와 동일한 모델 필수)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )

        self.reranker = CrossEncoder(
            "Dongjin-kr/ko-reranker",
            device="cpu"
        )

        # 2. Redis 연결 정보
        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        self.index_name = "otinus_knowledge_index"

        # ✅ 저장할 때와 동일한 스키마 정의를 추가합니다.
        # 차원 계산 (Ingestion과 동일하게)
        dims = len(self.embeddings.embed_query("dim_probe"))
        from langchain_redis import RedisConfig
        self.config = RedisConfig.with_metadata_schema(
            [
                {"name": "invoke_id", "type": "tag"},
                {"name": "source", "type": "text"},
                {"name": "page", "type": "numeric"},
            ],
            index_name=self.index_name,
            redis_url=self.redis_url,
            embedding_dimensions=dims,
            distance_metric="COSINE",
        )

    async def _rerank(self, query: str, docs: List) -> List:
        """CPU 부하가 큰 리랭킹 작업을 별도 스레드에서 수행합니다."""
        if not docs:
            return []

        # (질문, 본문) 쌍 생성 - 튜플 형식으로 전달
        pairs = [(query, doc.page_content) for doc in docs]

        # CrossEncoder.predict는 동기 함수이므로 thread로 실행
        scores = await asyncio.to_thread(self.reranker.predict, pairs)

        # 점수 매칭 및 정렬
        for i, doc in enumerate(docs):
            doc.metadata["rerank_score"] = scores[i]

        # 점수 높은 순 정렬 후 상위 5개 반환
        docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        return docs[:5]


    async def search(self, query: str, invoke_id: str) -> str | None:
        """
        [개인화 검색]
        Redis에서 'invoke_id' 태그가 붙은 문서 중에서만 검색을 수행합니다.
        """
        try:
            # Redis Vector Store 연결
            vectorstore = RedisVectorStore(
                embeddings=self.embeddings,
                config=self.config,  # 인덱스 설정 명시
            )

            # [핵심] Metadata Filter: 내 방의 문서만 보여줘!
            # Redis는 메타데이터 필드를 기반으로 강력한 필터링을 지원합니다.
            filter_rule = RedisTag("invoke_id") == invoke_id

            print(f"🔒 [Private Search] Searching for '{invoke_id}' with query: {query}")

            initial_docs = await asyncio.to_thread(
                vectorstore.max_marginal_relevance_search,
                query=query,
                k=5,  # 리랭커에게 전달할 후보 수
                fetch_k=30,  # 처음에 Redis에서 가져올 후보 수
                lambda_mult=0.6,  # 낮을수록 중복 방지 강화
                filter=filter_rule
            )

            if not initial_docs:
                return None

            # 🧠 2단계: Reranking (재정렬)
            # (질문, 문서내용) 쌍을 만들어 리랭커에 전달합니다.
            final_docs = await self._rerank(query, initial_docs)

            valid_docs = []
            seen_contents = set()

            # 리랭킹 점수 계산 (CPU 연산)
            for doc in final_docs:
                score = doc.metadata.get("rerank_score")
                content = doc.page_content.strip()
                if content not in seen_contents:
                    source_file = doc.metadata.get("source", "unknown")

                    formatted_doc = f"[출처: {source_file}]\n{content}"
                    valid_docs.append(formatted_doc)
                    seen_contents.add(content)

            if not valid_docs:
                return None

            return "\n\n".join(valid_docs)

        except Exception as e:
            print(f"🔥 [Search Error]: {str(e)}")
            return "검색 중 오류가 발생했습니다."


# 싱글톤 인스턴스
private_rag_service = PrivateRagSearchService()