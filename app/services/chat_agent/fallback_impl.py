"""
Fallback 구현체 — 이 프로젝트 전용

FallbackChain 구성:
    1. CachedAnswerFallback  — Redis 캐시에서 이전 답변 조회
    2. SafeMessageFallback   — 안전한 기본 메시지 반환

적용 위치:
    - LLM 호출 실패 시 (node_utils.call_llm)
    - aggregate_node에서 agent_answers가 없을 때
"""

import logging
from typing import Any, Dict

from app.services.agent_base.fallback import (
    BaseFallbackHandler,
    FallbackChain,
    FallbackResult,
)
from app.services.utils.answer_cache_service import answer_cache_service

logger = logging.getLogger(__name__)


# ── Fallback 핸들러 ───────────────────────────────────────────────────────────

class CachedAnswerFallback(BaseFallbackHandler):
    """
    Redis 캐시에서 동일 질문의 이전 답변을 조회한다.
    LLM 장애 시 캐시된 답변으로 대체한다.
    """

    async def handle(self, question: str, context: Dict[str, Any] = None) -> FallbackResult:
        ctx = context or {}
        invoke_id = ctx.get("invoke_id", "")

        if not invoke_id:
            return FallbackResult(success=False)

        cached = await answer_cache_service.get(invoke_id, question)
        if cached:
            answer = cached.get("answer", "")
            logger.warning(f"[Fallback] 캐시 응답 사용: {question[:50]}")
            return FallbackResult(
                success=True,
                content=answer,
                source="cache",
                metadata={"cached": True},
            )

        return FallbackResult(success=False)


class SafeMessageFallback(BaseFallbackHandler):
    """
    항상 성공하는 안전 메시지 핸들러.
    FallbackChain의 마지막에 위치하여 최종 대체 메시지를 반환한다.
    """

    def __init__(self, message: str = "일시적으로 답변이 어렵습니다. 잠시 후 다시 시도해 주세요."):
        self.message = message

    async def handle(self, question: str, context: Dict[str, Any] = None) -> FallbackResult:
        error_type = (context or {}).get("error_type", "unknown")
        logger.warning(f"[Fallback] 안전 메시지 반환 (error_type={error_type}): {question[:50]}")
        return FallbackResult(
            success=True,
            content=self.message,
            source="safe_message",
        )

