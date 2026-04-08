"""
Guardrails 베이스 인터페이스

입력/출력 필터를 파이프라인으로 관리한다.

사용법:
    1. BaseInputGuard  를 상속하여 check() 구현  → 입력 검증/차단
    2. BaseOutputGuard 를 상속하여 check() 구현  → 출력 검증/정제
    3. GuardrailsPipeline 에 Guard 목록을 등록하여 사용

기본 구현체:
    - GuardrailsPipeline: 등록된 Guard를 순서대로 실행
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    """Guard 실행 결과"""
    allowed: bool                     # True: 통과 / False: 차단
    text: str                         # 통과 시 원본 또는 정제된 텍스트
    reason: Optional[str] = None      # 차단 시 사유
    metadata: Dict[str, Any] = None   # 추가 정보

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ── 베이스 인터페이스 ──────────────────────────────────────────────────────────

class BaseInputGuard(ABC):
    """입력 검증 인터페이스 — 구현체는 check()를 반드시 구현해야 한다."""

    @abstractmethod
    async def check(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        """
        입력 텍스트를 검증한다.

        Args:
            text:    사용자 입력 텍스트
            context: 추가 컨텍스트 (session_id, user_id 등)

        Returns:
            GuardResult(allowed, text, reason)
            - allowed=False: 요청 차단, reason에 사유 포함
            - allowed=True:  통과, text에 정제된 텍스트 포함
        """
        pass


class BaseOutputGuard(ABC):
    """출력 검증 인터페이스 — 구현체는 check()를 반드시 구현해야 한다."""

    @abstractmethod
    async def check(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        """
        출력 텍스트를 검증한다.

        Args:
            text:    LLM 생성 답변 텍스트
            context: 추가 컨텍스트

        Returns:
            GuardResult(allowed, text, reason)
            - allowed=False: 출력 차단 (대체 메시지 사용)
            - allowed=True:  통과, text에 정제된 텍스트 포함
        """
        pass


# ── 파이프라인 ─────────────────────────────────────────────────────────────────

class GuardrailsPipeline:
    """
    InputGuard / OutputGuard 목록을 순서대로 실행하는 파이프라인.

    하나라도 차단되면 즉시 중단하고 GuardResult(allowed=False)를 반환한다.

    사용 예시:
        pipeline = GuardrailsPipeline(
            input_guards=[BlankInputGuard(), ProfanityInputGuard()],
            output_guards=[ProfanityOutputGuard()],
        )

        # 입력 검증
        result = await pipeline.check_input(user_message)
        if not result.allowed:
            return result.reason

        # 출력 검증
        result = await pipeline.check_output(llm_answer)
        final_text = result.text
    """

    def __init__(
        self,
        input_guards: List[BaseInputGuard] = None,
        output_guards: List[BaseOutputGuard] = None,
    ):
        self.input_guards: List[BaseInputGuard] = input_guards or []
        self.output_guards: List[BaseOutputGuard] = output_guards or []

    async def check_input(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        """등록된 InputGuard를 순서대로 실행한다."""
        current_text = text
        for guard in self.input_guards:
            result = await guard.check(current_text, context)
            if not result.allowed:
                return result
            current_text = result.text  # 정제된 텍스트를 다음 Guard에 전달
        return GuardResult(allowed=True, text=current_text)

    async def check_output(self, text: str, context: Dict[str, Any] = None) -> GuardResult:
        """등록된 OutputGuard를 순서대로 실행한다."""
        current_text = text
        for guard in self.output_guards:
            result = await guard.check(current_text, context)
            if not result.allowed:
                return result
            current_text = result.text
        return GuardResult(allowed=True, text=current_text)
