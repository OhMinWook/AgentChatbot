"""
Fallback 베이스 인터페이스

정상 경로 실패 시 대체 경로를 순서대로 시도한다.

사용법:
    1. BaseFallbackHandler 를 상속하여 handle() 구현 → 대체 처리 로직
    2. FallbackChain 에 핸들러 목록을 등록하여 사용
       → 첫 번째로 성공한 핸들러 결과를 반환

기본 구현체:
    - FallbackChain: 핸들러를 순서대로 시도
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class FallbackResult:
    """Fallback 실행 결과"""
    success: bool                      # True: 대체 처리 성공
    content: Optional[str] = None      # 대체 응답 텍스트
    source: Optional[str] = None       # 어떤 핸들러가 처리했는지
    metadata: Dict[str, Any] = None    # 추가 정보

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ── 베이스 인터페이스 ──────────────────────────────────────────────────────────

class BaseFallbackHandler(ABC):
    """Fallback 처리 인터페이스 — 구현체는 handle()을 반드시 구현해야 한다."""

    @abstractmethod
    async def handle(self, question: str, context: Dict[str, Any] = None) -> FallbackResult:
        """
        대체 처리를 시도한다.

        Args:
            question: 원본 질문
            context:  추가 컨텍스트 (error_type, invoke_id 등)

        Returns:
            FallbackResult(success, content, source)
            - success=True:  대체 처리 성공, content에 응답 포함
            - success=False: 이 핸들러로 처리 불가, 다음 핸들러로 넘김
        """
        pass


# ── 파이프라인 ─────────────────────────────────────────────────────────────────

class FallbackChain:
    """
    핸들러를 순서대로 시도하고 첫 번째 성공 결과를 반환한다.
    모든 핸들러가 실패하면 default_message를 반환한다.

    사용 예시:
        chain = FallbackChain(
            handlers=[
                CachedAnswerFallback(cache_service),
                SafeMessageFallback("일시적으로 답변이 어렵습니다."),
            ],
            default_message="서비스 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

        result = await chain.run(question, context={"error_type": "llm_timeout"})
        return result.content
    """

    def __init__(
        self,
        handlers: List[BaseFallbackHandler],
        default_message: str = "일시적으로 답변이 어렵습니다. 잠시 후 다시 시도해 주세요.",
    ):
        self.handlers = handlers
        self.default_message = default_message

    async def run(self, question: str, context: Dict[str, Any] = None) -> FallbackResult:
        """
        핸들러를 순서대로 시도한다.

        Args:
            question: 원본 질문
            context:  추가 컨텍스트

        Returns:
            FallbackResult — 성공한 핸들러 결과 또는 default_message
        """
        for handler in self.handlers:
            result = await handler.handle(question, context)
            if result.success:
                return result

        return FallbackResult(
            success=True,
            content=self.default_message,
            source="default",
        )
