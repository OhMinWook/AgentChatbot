"""
질문-답변 Redis 캐시 서비스
- 동일 질문에 대해 LLM 호출 없이 즉시 응답
- 캐시 키: sha256(invoke_id + "|" + normalized_query)
"""

import hashlib
import json
import logging
from typing import Optional

from app.core.config import settings
from app.core.redis_client import async_redis

logger = logging.getLogger(__name__)

_PREFIX = "qcache:"


def _make_key(invoke_id: str, query: str) -> str:
    raw = f"{invoke_id}|{query.strip().lower()}"
    return _PREFIX + hashlib.sha256(raw.encode()).hexdigest()[:32]


class AnswerCacheService:
    async def get(self, invoke_id: str, query: str) -> Optional[dict]:
        """캐시된 답변 조회. 없으면 None 반환"""
        if not settings.ANSWER_CACHE_ENABLED:
            return None
        key = _make_key(invoke_id, query)
        try:
            data = await async_redis.get(key)
            if data:
                logger.info(f"[Cache] HIT: {query[:50]}")
                return json.loads(data)
        except Exception as e:
            logger.warning(f"[Cache] get 실패 (무시): {e}")
        return None

    async def add_translation(self, invoke_id: str, query: str, lang: str, translation: str) -> None:
        """기존 캐시 항목에 번역 결과 추가"""
        if not settings.ANSWER_CACHE_ENABLED:
            return
        key = _make_key(invoke_id, query)
        try:
            data = await async_redis.get(key)
            if data:
                cached = json.loads(data)
                translations = cached.get("translations", {})
                translations[lang] = translation
                cached["translations"] = translations
                ttl = await async_redis.ttl(key)
                if ttl > 0:
                    await async_redis.setex(key, ttl, json.dumps(cached, ensure_ascii=False))
                logger.info(f"[Cache] 번역 추가: {query[:50]} ({lang})")
        except Exception as e:
            logger.warning(f"[Cache] add_translation 실패 (무시): {e}")

    async def set(self, invoke_id: str, query: str, answer: str, references: list) -> None:
        """답변 캐시 저장"""
        if not settings.ANSWER_CACHE_ENABLED:
            return
        key = _make_key(invoke_id, query)
        try:
            value = json.dumps({"answer": answer, "references": references}, ensure_ascii=False)
            await async_redis.setex(key, settings.ANSWER_CACHE_TTL, value)
            logger.info(f"[Cache] SET: {query[:50]} (TTL={settings.ANSWER_CACHE_TTL}s)")
        except Exception as e:
            logger.warning(f"[Cache] set 실패 (무시): {e}")


answer_cache_service = AnswerCacheService()
