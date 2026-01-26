"""
로컬 인덱스 서비스

- Voyager: ColBERT 임베딩 저장/검색
- Redis: 문서 메타데이터 저장
- 모델 서버: 인코딩만 담당 (Stateless)
"""

import os
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from pylate import indexes, retrieve

from app.core.config import settings
from app.services.clients.model_server_client import model_server_client

# Redis 클라이언트
import redis.asyncio as redis


# ========================================
# 설정
# ========================================
INDEX_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "colbert_indexes")
os.makedirs(INDEX_BASE_DIR, exist_ok=True)


@dataclass
class SearchResult:
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class LocalIndexService:
    """로컬 Voyager 인덱스 + Redis 메타데이터 관리"""

    def __init__(self):
        self._indexes: Dict[str, indexes.Voyager] = {}
        self._retrievers: Dict[str, retrieve.ColBERT] = {}
        self._redis: Optional[redis.Redis] = None

    async def _get_redis(self) -> redis.Redis:
        """Redis 연결 (lazy init)"""
        if self._redis is None:
            self._redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
        return self._redis

    def _get_index_path(self, invoke_id: str) -> str:
        """인덱스 경로 반환"""
        return os.path.join(INDEX_BASE_DIR, invoke_id)

    def _get_redis_key(self, invoke_id: str, doc_id: str) -> str:
        """Redis 키 생성"""
        return f"colbert:{invoke_id}:{doc_id}"

    async def _load_or_create_index(self, invoke_id: str, create_if_missing: bool = True) -> Optional[indexes.Voyager]:
        """인덱스 로드 또는 생성"""
        if invoke_id in self._indexes:
            return self._indexes[invoke_id]

        index_path = self._get_index_path(invoke_id)

        # 기존 인덱스 존재 확인
        if os.path.exists(index_path):
            try:
                self._indexes[invoke_id] = indexes.Voyager(
                    index_folder=index_path,
                    index_name="index",
                    override=False
                )
                self._retrievers[invoke_id] = retrieve.ColBERT(index=self._indexes[invoke_id])
                print(f"📂 [LocalIndex] Loaded existing index: {invoke_id}")
                return self._indexes[invoke_id]
            except Exception as e:
                print(f"⚠️ [LocalIndex] Failed to load index {invoke_id}: {e}")

        if create_if_missing:
            os.makedirs(index_path, exist_ok=True)
            self._indexes[invoke_id] = indexes.Voyager(
                index_folder=index_path,
                index_name="index",
                override=True
            )
            self._retrievers[invoke_id] = retrieve.ColBERT(index=self._indexes[invoke_id])
            print(f"📂 [LocalIndex] Created new index: {invoke_id}")
            return self._indexes[invoke_id]

        return None

    async def index_documents(
        self,
        invoke_id: str,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        문서 인덱싱

        1. 모델 서버에서 임베딩 생성
        2. 로컬 Voyager에 저장
        3. Redis에 메타데이터 저장
        """
        if not documents:
            return 0

        # 문서 내용 추출
        doc_ids = [doc["id"] for doc in documents]
        doc_contents = [doc["content"] for doc in documents]

        print(f"🔄 [LocalIndex] Indexing {len(documents)} documents for '{invoke_id}'...")

        # 1. 모델 서버에서 임베딩 생성
        embeddings = await model_server_client.encode_documents(doc_contents)

        if not embeddings:
            print(f"❌ [LocalIndex] Failed to get embeddings")
            return 0

        # ColBERT 임베딩: 각 문서마다 토큰 수가 다름 → 개별 numpy 배열로 변환 (float32)
        embeddings_list = [np.array(emb, dtype=np.float32) for emb in embeddings]
        print(f"📐 [LocalIndex] Embeddings: {len(embeddings_list)} docs, first shape: {embeddings_list[0].shape}")

        # 2. Voyager 인덱스에 저장
        index = await self._load_or_create_index(invoke_id)
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=embeddings_list
        )

        # 3. Redis에 메타데이터 저장
        r = await self._get_redis()
        pipe = r.pipeline()

        for doc in documents:
            key = self._get_redis_key(invoke_id, doc["id"])
            pipe.hset(key, mapping={
                "content": doc["content"],
                "metadata": json.dumps(doc.get("metadata", {}), ensure_ascii=False)
            })
            # TTL 설정 (7일)
            pipe.expire(key, settings.CHAT_HISTORY_TTL)

        await pipe.execute()

        print(f"✅ [LocalIndex] Indexed {len(documents)} documents")
        return len(documents)

    async def search(
        self,
        invoke_id: str,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        검색

        1. 모델 서버에서 쿼리 임베딩 생성
        2. 로컬 Voyager에서 검색
        3. Redis에서 메타데이터 조회
        """
        index = await self._load_or_create_index(invoke_id, create_if_missing=False)
        if index is None:
            print(f"⚠️ [LocalIndex] No index for '{invoke_id}'")
            return []

        print(f"🔍 [LocalIndex] Searching '{invoke_id}': {query[:50]}...")

        # 1. 모델 서버에서 쿼리 임베딩 생성
        query_embeddings = await model_server_client.encode_query([query])

        if not query_embeddings:
            print(f"❌ [LocalIndex] Failed to get query embedding")
            return []

        # ColBERT 쿼리 임베딩: 개별 numpy 배열로 변환 (float32)
        query_embeddings_list = [np.array(emb, dtype=np.float32) for emb in query_embeddings]

        # 2. Voyager에서 검색
        retriever = self._retrievers.get(invoke_id)
        if retriever is None:
            retriever = retrieve.ColBERT(index=index)
            self._retrievers[invoke_id] = retriever

        search_results = retriever.retrieve(
            queries_embeddings=query_embeddings_list,
            k=top_k
        )

        if not search_results or len(search_results) == 0:
            return []

        # 3. Redis에서 메타데이터 조회
        r = await self._get_redis()
        results = []

        for result in search_results[0]:
            doc_id = result.get("id", "")
            score = float(result.get("score", 0.0))

            key = self._get_redis_key(invoke_id, doc_id)
            doc_data = await r.hgetall(key)

            if doc_data:
                content = doc_data.get("content", "")
                metadata = json.loads(doc_data.get("metadata", "{}"))
            else:
                content = ""
                metadata = {}

            results.append(SearchResult(
                doc_id=doc_id,
                score=score,
                content=content,
                metadata=metadata
            ))

        print(f"✅ [LocalIndex] Found {len(results)} results")
        return results

    async def search_batch(
        self,
        invoke_id: str,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[SearchResult]]:
        """배치 검색"""
        index = await self._load_or_create_index(invoke_id, create_if_missing=False)
        if index is None:
            return [[] for _ in queries]

        print(f"🔍 [LocalIndex] Batch searching '{invoke_id}': {len(queries)} queries...")

        # 1. 모델 서버에서 쿼리 임베딩 생성 (배치)
        query_embeddings = await model_server_client.encode_query(queries)

        if not query_embeddings:
            return [[] for _ in queries]

        # ColBERT 쿼리 임베딩: 개별 numpy 배열로 변환 (float32)
        query_embeddings_list = [np.array(emb, dtype=np.float32) for emb in query_embeddings]

        # 2. Voyager에서 검색
        retriever = self._retrievers.get(invoke_id)
        if retriever is None:
            retriever = retrieve.ColBERT(index=index)
            self._retrievers[invoke_id] = retriever

        all_search_results = retriever.retrieve(
            queries_embeddings=query_embeddings_list,
            k=top_k
        )

        # 3. Redis에서 메타데이터 조회
        r = await self._get_redis()
        all_results = []

        for query_results in all_search_results:
            results = []
            for result in query_results:
                doc_id = result.get("id", "")
                score = float(result.get("score", 0.0))

                key = self._get_redis_key(invoke_id, doc_id)
                doc_data = await r.hgetall(key)

                if doc_data:
                    content = doc_data.get("content", "")
                    metadata = json.loads(doc_data.get("metadata", "{}"))
                else:
                    content = ""
                    metadata = {}

                results.append(SearchResult(
                    doc_id=doc_id,
                    score=score,
                    content=content,
                    metadata=metadata
                ))

            all_results.append(results)

        print(f"✅ [LocalIndex] Batch search complete: {len(all_results)} queries")
        return all_results

    async def delete_index(self, invoke_id: str) -> int:
        """인덱스 삭제"""
        import shutil

        # 메모리에서 제거
        self._indexes.pop(invoke_id, None)
        self._retrievers.pop(invoke_id, None)

        # 디스크에서 삭제
        index_path = self._get_index_path(invoke_id)
        if os.path.exists(index_path):
            shutil.rmtree(index_path)

        # Redis에서 삭제
        r = await self._get_redis()
        pattern = f"colbert:{invoke_id}:*"
        cursor = 0
        deleted = 0

        while True:
            cursor, keys = await r.scan(cursor, match=pattern, count=100)
            if keys:
                await r.delete(*keys)
                deleted += len(keys)
            if cursor == 0:
                break

        print(f"🗑️ [LocalIndex] Deleted index '{invoke_id}' ({deleted} docs)")
        return deleted

    async def load_existing_indexes(self):
        """서버 시작 시 기존 인덱스 로드"""
        if not os.path.exists(INDEX_BASE_DIR):
            return

        for invoke_id in os.listdir(INDEX_BASE_DIR):
            index_path = os.path.join(INDEX_BASE_DIR, invoke_id)
            if os.path.isdir(index_path):
                try:
                    await self._load_or_create_index(invoke_id, create_if_missing=False)
                except Exception as e:
                    print(f"⚠️ [LocalIndex] Failed to load {invoke_id}: {e}")

        print(f"✅ [LocalIndex] Loaded {len(self._indexes)} indexes")


# 싱글톤
local_index_service = LocalIndexService()
