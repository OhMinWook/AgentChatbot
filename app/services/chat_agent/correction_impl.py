"""
Self-Correction 구현체 — 이 프로젝트 전용

BaseVerifier  → HallucinationVerifier  (LLM 기반 할루시네이션 검증)
BaseCorrector → StrictPromptCorrector  (AGENT_STRICT 프롬프트로 재시도)
"""

import json
import logging
from typing import Any, Dict

from app.services.agent_base.correction import (
    BaseCorrector,
    BaseVerifier,
    CorrectionContext,
    VerifyResult,
)
from app.core.config import settings
from app.services.api_clients.llm_utils import strip_markdown_codeblock
from app.services.chat_agent.node_utils import (
    VERIFY_ANSWER_JSON_SCHEMA,
    call_llm,
    classify_question,
)
from app.services.chat_agent.prompts import get_verify_prompt

logger = logging.getLogger(__name__)


class HallucinationVerifier(BaseVerifier):
    """
    LLM을 사용하여 답변이 문서에 근거하는지 검증한다.
    질문 유형별로 다른 검증 프롬프트를 적용한다.
    """

    async def verify(self, question: str, answer: str, context: str) -> VerifyResult:
        question_type = classify_question(question)
        type_id = question_type.id if question_type else None
        verify_prompt = get_verify_prompt(type_id)

        logger.debug(f"[Verify] 유형: {type_id or '미분류'} | 질문: {question[:50]}")

        response = await call_llm(
            [
                {"role": "system", "content": verify_prompt.system},
                {"role": "user", "content": verify_prompt.user.format(
                    context=context,
                    question=question,
                    answer=answer,
                )},
            ],
            max_tokens=settings.VERIFY_MAX_TOKENS,
            json_schema=VERIFY_ANSWER_JSON_SCHEMA,
        )

        try:
            result = json.loads(strip_markdown_codeblock(response))
            passed = result.get("passed", True)
            issues = result.get("issues", [])
        except json.JSONDecodeError:
            logger.warning(f"[Verify] JSON 파싱 실패 - 통과 처리: {response[:100]}")
            passed, issues = True, []

        if not passed:
            logger.warning(f"[Verify] 검증 실패 | 질문: {question[:50]} | 문제: {issues}")
        else:
            logger.info(f"[Verify] 검증 통과 | 질문: {question[:50]}")

        return VerifyResult(passed=passed, issues=issues)


class StrictPromptCorrector(BaseCorrector):
    """
    검증 실패 시 AGENT_STRICT 프롬프트로 재시도하도록 state를 업데이트한다.
    agent_answers를 초기화하여 process_question_node가 재실행되도록 한다.
    """

    async def correct(self, ctx: CorrectionContext) -> Dict[str, Any]:
        logger.warning(
            f"[Correct] AGENT_STRICT 재시도 ({ctx.retry_count + 1}회) "
            f"| 문제: {ctx.issues}"
        )
        return {
            "verification_passed": False,
            "retry_count": ctx.retry_count + 1,
            "agent_answers": [],
        }
