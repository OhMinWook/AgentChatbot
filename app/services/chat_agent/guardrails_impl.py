"""
Guardrails 구현체 — 이 프로젝트 전용

InputGuard:
    BlankInputGuard      — 빈 메시지 차단
    ProfanityInputGuard  — 금칙어 포함 입력 차단

OutputGuard:
    ProfanityOutputGuard — 금칙어 포함 출력 정제 (치환)
"""

import logging
from typing import Any, Dict

from app.services.agent_base.guardrails import (
    BaseInputGuard,
    BaseOutputGuard,
    GuardResult,
    GuardrailsPipeline,
)
from app.services.utils.profanity_filter import filter_profanity, _PATTERN

logger = logging.getLogger(__name__)


# ── 입력 Guard ────────────────────────────────────────────────────────────────

class BlankInputGuard(BaseInputGuard):
    """빈 메시지 또는 공백만 있는 입력을 차단한다."""

    async def check(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        if not text or not text.strip():
            logger.warning("[Guard] 빈 메시지 차단")
            return GuardResult(
                allowed=False,
                text=text,
                reason="메시지를 입력해 주세요.",
            )
        return GuardResult(allowed=True, text=text.strip())


class ProfanityInputGuard(BaseInputGuard):
    """금칙어가 포함된 입력을 차단한다."""

    async def check(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        if _PATTERN and _PATTERN.search(text):
            logger.warning(f"[Guard] 금칙어 포함 입력 차단: {text[:50]}")
            return GuardResult(
                allowed=False,
                text=text,
                reason="부적절한 표현이 포함되어 있습니다.",
            )
        return GuardResult(allowed=True, text=text)


# ── 출력 Guard ────────────────────────────────────────────────────────────────

class ProfanityOutputGuard(BaseOutputGuard):
    """금칙어가 포함된 출력을 치환한다 (차단하지 않고 정제)."""

    async def check(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        cleaned = filter_profanity(text)
        if cleaned != text:
            logger.warning("[Guard] 출력 금칙어 정제 완료")
        return GuardResult(allowed=True, text=cleaned)


# ── 파이프라인 싱글톤 ─────────────────────────────────────────────────────────

chat_guardrails = GuardrailsPipeline(
    input_guards=[
        BlankInputGuard(),
        ProfanityInputGuard(),
    ],
    output_guards=[
        ProfanityOutputGuard(),
    ],
)
