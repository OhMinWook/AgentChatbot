"""
Self-Correction 베이스 인터페이스

사용법:
    1. BaseVerifier  를 상속하여 verify() 구현  → 뭐가 잘못됐는지
    2. BaseCorrector 를 상속하여 correct() 구현 → 어떻게 고칠지
    3. BaseStopRule  를 상속하여 should_stop() 구현 → 언제 포기할지
    4. CorrectionPolicy 에 세 개를 조합하여 사용

기본 구현체:
    - MaxRetryStopRule: 최대 재시도 횟수 기반 종료 규칙
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class VerifyResult:
    """검증 결과"""
    passed: bool
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrectionContext:
    """교정 실행에 필요한 컨텍스트"""
    question: str
    answer: str
    context: str
    retry_count: int
    issues: List[str]
    state: Dict[str, Any]


# ── 베이스 인터페이스 ──────────────────────────────────────────────────────────

class BaseVerifier(ABC):
    """답변 검증 인터페이스 — 구현체는 verify()를 반드시 구현해야 한다."""

    @abstractmethod
    async def verify(self, question: str, answer: str, context: str) -> VerifyResult:
        """
        답변이 올바른지 검증한다.

        Args:
            question: 원본 질문
            answer:   생성된 답변
            context:  검색된 문서 컨텍스트

        Returns:
            VerifyResult(passed, issues)
        """
        pass


class BaseCorrector(ABC):
    """교정 전략 인터페이스 — 구현체는 correct()를 반드시 구현해야 한다."""

    @abstractmethod
    async def correct(self, ctx: CorrectionContext) -> Dict[str, Any]:
        """
        검증 실패 시 LangGraph state 업데이트를 반환한다.

        Args:
            ctx: 교정에 필요한 컨텍스트

        Returns:
            LangGraph state 업데이트 dict
            예: {"verification_passed": False, "retry_count": 1, "agent_answers": []}
        """
        pass


class BaseStopRule(ABC):
    """교정 종료 조건 인터페이스 — 구현체는 should_stop()을 반드시 구현해야 한다."""

    @abstractmethod
    def should_stop(self, retry_count: int, results: List[VerifyResult]) -> bool:
        """
        교정을 중단할지 판단한다.

        Args:
            retry_count: 현재 재시도 횟수
            results:     이번 검증 결과 목록

        Returns:
            True면 교정 중단 (현재 답변 그대로 사용)
        """
        pass


# ── 기본 구현체 ────────────────────────────────────────────────────────────────

class MaxRetryStopRule(BaseStopRule):
    """최대 재시도 횟수 기반 종료 규칙"""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries

    def should_stop(self, retry_count: int, results: List[VerifyResult]) -> bool:
        return retry_count >= self.max_retries


# ── 오케스트레이터 ─────────────────────────────────────────────────────────────

class CorrectionPolicy:
    """
    Verifier + Corrector + StopRule 을 조합하는 오케스트레이터.

    LangGraph 노드에서 아래와 같이 사용한다:

        policy = CorrectionPolicy(
            verifier=HallucinationVerifier(),
            corrector=StrictPromptCorrector(),
            stop_rule=MaxRetryStopRule(max_retries=1),
        )
        result = await policy.run(agent_answers, retry_count, state)
        return result
    """

    def __init__(
        self,
        verifier: BaseVerifier,
        corrector: BaseCorrector,
        stop_rule: BaseStopRule,
    ):
        self.verifier = verifier
        self.corrector = corrector
        self.stop_rule = stop_rule

    async def run(
        self,
        answers: List[Dict[str, Any]],
        retry_count: int,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        검증 → 실패 시 교정 → 종료 조건 판단 순서로 실행한다.

        Args:
            answers:     agent_answers 리스트
            retry_count: 현재 재시도 횟수
            state:       LangGraph 전체 state

        Returns:
            LangGraph state 업데이트 dict
        """
        answers_to_verify = [a for a in answers if a.get("context")]

        if not answers_to_verify:
            return {"verification_passed": True}

        results = []
        for answer in answers_to_verify:
            result = await self.verifier.verify(
                question=answer.get("question", ""),
                answer=answer.get("answer", ""),
                context=answer.get("context", ""),
            )
            results.append(result)

        failed = [r for r in results if not r.passed]

        if not failed:
            return {"verification_passed": True}

        if self.stop_rule.should_stop(retry_count, results):
            return {"verification_passed": True}

        # 첫 번째 실패 답변 기준으로 교정 컨텍스트 구성
        first_failed_answer = answers_to_verify[0]
        ctx = CorrectionContext(
            question=first_failed_answer.get("question", ""),
            answer=first_failed_answer.get("answer", ""),
            context=first_failed_answer.get("context", ""),
            retry_count=retry_count,
            issues=[issue for r in failed for issue in r.issues],
            state=state,
        )

        return await self.corrector.correct(ctx)
