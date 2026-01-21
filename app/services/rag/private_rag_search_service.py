import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_redis import RedisVectorStore, RedisConfig
from redisvl.query.filter import Tag as RedisTag

from app.core.config import settings
from app.services.clients.model_server_client import model_server_client


class ExternalServiceEmbeddings(Embeddings):
    """
    외부 모델 서버를 통해 임베딩을 수행하는 LangChain 호환 Embeddings 클래스.
    """
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await model_server_client.get_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await model_server_client.get_embeddings([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """ [수정] 동기 문서 임베딩 시 동기 클라이언트 사용 """
        return model_server_client.get_embeddings_sync(texts)

    def embed_query(self, text: str) -> List[float]:
        """ [수정] 동기 질의 임베딩 시 동기 클라이언트 사용 """
        embeddings = model_server_client.get_embeddings_sync([text])
        return embeddings[0]


class PrivateRagSearchService:
    def __init__(self):
        self.embeddings = ExternalServiceEmbeddings()

        self.redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"
        self.index_name = settings.RAG_INDEX_NAME
        self.config = RedisConfig.with_metadata_schema(
            [
                {"name": "invoke_id", "type": "tag"},
                {"name": "source", "type": "text"},
                {"name": "page", "type": "numeric"},
            ],
            index_name=self.index_name,
            redis_url=self.redis_url,
            embedding_dimensions=settings.EMBEDDING_DIMS,
            distance_metric="COSINE",
        )

    async def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """외부 모델 서버를 사용하여 Reranking 수행"""
        if not docs:
            return []

        # API에 전달할 문서 내용 목록 (누락되었던 부분)
        doc_contents = [doc.page_content for doc in docs]
        # 문서 내용을 키로, 원본 Document 객체를 값으로 하는 맵 생성
        doc_map = {doc.page_content: doc for doc in docs}

        # 외부 API 호출
        reranked_results = await model_server_client.rerank(query, doc_contents)
        
        # 결과 재구성: 점수 주입 및 정렬된 문서 목록 생성
        final_docs = []
        for result in reranked_results:
            content = result["text"] # 'document' 대신 'text' 키 사용
            score = result["score"]
            if content in doc_map:
                doc = doc_map[content]
                doc.metadata["rerank_score"] = score
                final_docs.append(doc)
        
        final_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        top_docs = final_docs[:5]

        # 디버그: rerank 결과 출력
        print(f"\n{'='*60}")
        print(f"🔍 [Rerank 결과] 상위 {len(top_docs)}개 문서")
        print(f"{'='*60}")
        for idx, doc in enumerate(top_docs):
            score = doc.metadata.get("rerank_score", 0)
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            content_preview = doc.page_content[:200].replace('\n', ' ')
            print(f"\n[{idx+1}] 점수: {score:.4f} | 출처: {source} (p.{page})")
            print(f"    내용: {content_preview}...")
        print(f"{'='*60}\n")

        return top_docs


    async def search(self, query: str, invoke_id: str) -> dict:
        """
        [개인화 검색]
        Redis에서 'invoke_id' 태그가 붙은 문서 중에서만 검색을 수행합니다.
        반환값: {"context": "...", "references": [{"source": "...", "page": ...}, ...]}
        """
        try:
            vectorstore = RedisVectorStore(
                embeddings=self.embeddings,
                config=self.config,
            )

            filter_rule = RedisTag("invoke_id") == invoke_id

            print(f"🔒 [Private Search] Searching for '{invoke_id}' with query: {query}")

            initial_docs = await asyncio.to_thread(
                vectorstore.max_marginal_relevance_search,
                query=query,
                k=10,
                fetch_k=50,
                lambda_mult=0.6,
                filter=filter_rule
            )

            if not initial_docs:
                return {"context": None, "references": []}
            
            final_docs = await self._rerank(query, initial_docs)

            valid_docs = []
            references = []
            seen_contents = set()

            for doc in final_docs:
                content = doc.page_content.strip()
                score = doc.metadata.get("rerank_score", 0)

                # 점수가 0.05 이상인 경우에만 사용
                if score < 0.05:
                    continue

                if content not in seen_contents:
                    source_file = doc.metadata.get("source", "unknown")
                    page_num = doc.metadata.get("page", 0)
                    
                    formatted_doc = f"[출처: {source_file}]\n{content}"
                    valid_docs.append(formatted_doc)
                    
                    # 참조 정보 추가 (중복 제거)
                    ref_info = {"source": source_file, "page": page_num}
                    if ref_info not in references:
                        references.append(ref_info)
                        
                    seen_contents.add(content)

            if not valid_docs:
                return {"context": None, "references": []}

            return {
                "context": "\n\n".join(valid_docs),
                "references": references
            }

        except Exception as e:
            print(f"🔥 [Search Error]: {str(e)}")
            return {"context": "검색 중 오류가 발생했습니다.", "references": []}


# 싱글톤 인스턴스
private_rag_service = PrivateRagSearchService()