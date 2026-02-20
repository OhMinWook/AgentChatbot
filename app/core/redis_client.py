"""공유 Redis 클라이언트

async_redis  : memory_service 등 비동기 컨텍스트에서 사용
sync_redis   : download_service 등 동기 컨텍스트에서 사용
"""

import redis
from redis import asyncio as aioredis

from app.core.config import settings

_REDIS_URL = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

# 비동기 클라이언트 (connection pool 내장)
async_redis: aioredis.Redis = aioredis.from_url(_REDIS_URL, decode_responses=True)

# 동기 클라이언트 (connection pool 내장)
sync_redis: redis.Redis = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    decode_responses=True,
)
