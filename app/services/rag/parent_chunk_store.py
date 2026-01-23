"""
Parent 청크 Redis 저장소

Parent-Child 청킹 전략에서 Parent 청크(큰 컨텍스트)를 저장하고 조회합니다.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from redis import asyncio as aioredis
from app.core.config import settings

logger = logging.getLogger(__name__)


class ParentChunkStore:
    """
    Redis 기반 Parent 청크 저장소

    Key 형식: parent_chunk:{invoke_id}:{parent_id}
    Value: JSON 직렬화된 청크 데이터
    """

    def __init__(self):
        self.redis = aioredis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            decode_responses=True
        )
        self.ttl = settings.CHAT_HISTORY_TTL

    def _make_key(self, invoke_id: str, parent_id: str) -> str:
        return f"parent_chunk:{invoke_id}:{parent_id}"

    def _make_index_key(self, invoke_id: str) -> str:
        return f"parent_chunk_index:{invoke_id}"

    async def save_batch(self, invoke_id: str, chunks: List[Dict[str, Any]]) -> int:
        """여러 Parent 청크를 일괄 저장"""
        if not chunks:
            return 0

        saved_count = 0
        index_key = self._make_index_key(invoke_id)
        pipe = self.redis.pipeline()

        for chunk in chunks:
            parent_id = chunk.get("parent_id")
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            if not parent_id or not content:
                continue

            key = self._make_key(invoke_id, parent_id)
            data = {
                "content": content,
                "metadata": metadata,
                "parent_id": parent_id
            }

            pipe.setex(key, self.ttl, json.dumps(data, ensure_ascii=False))
            pipe.sadd(index_key, parent_id)
            saved_count += 1

        if saved_count > 0:
            pipe.expire(index_key, self.ttl)
            await pipe.execute()
            logger.info(f"Batch saved {saved_count} parent chunks for invoke_id: {invoke_id}")

        return saved_count

    async def load(self, invoke_id: str, parent_id: str) -> Optional[str]:
        """Parent 청크 조회 (content만 반환)"""
        key = self._make_key(invoke_id, parent_id)

        try:
            data = await self.redis.get(key)
            if data:
                parsed = json.loads(data)
                return parsed.get("content")
            return None
        except Exception as e:
            logger.error(f"Failed to load parent chunk: {e}")
            return None

    async def load_many(self, invoke_id: str, parent_ids: List[str]) -> List[Dict[str, Any]]:
        """여러 Parent 청크 일괄 조회"""
        if not parent_ids:
            return []

        keys = [self._make_key(invoke_id, pid) for pid in parent_ids]

        try:
            results = await self.redis.mget(keys)
            chunks = []
            for data in results:
                if data:
                    chunks.append(json.loads(data))
            return chunks
        except Exception as e:
            logger.error(f"Failed to load parent chunks: {e}")
            return []

    async def delete_all(self, invoke_id: str) -> int:
        """특정 invoke_id의 모든 Parent 청크 삭제"""
        index_key = self._make_index_key(invoke_id)

        try:
            parent_ids = await self.redis.smembers(index_key)
            if not parent_ids:
                return 0

            keys = [self._make_key(invoke_id, pid) for pid in parent_ids]
            keys.append(index_key)

            deleted = await self.redis.delete(*keys)
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete parent chunks: {e}")
            return 0


parent_chunk_store = ParentChunkStore()
