"""
Observability 베이스 인터페이스

Agent 실행 과정의 트레이싱과 스코어 기록을 표준화한다.
노드마다 반복되는 try/except + langfuse.score() 패턴을 공통화한다.

사용법:
    1. BaseTracer 를 상속하여 score() / update_metadata() 구현
    2. LangfuseTracer 는 기본 구현체로 바로 사용 가능

기본 구현체:
    - LangfuseTracer: Langfuse 연동 (노드에서 직접 사용)
    - NoopTracer:     Langfuse 없는 환경용 (테스트 등)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────

@dataclass
class TraceEvent:
    """스코어 기록 이벤트"""
    name: str                          # 스코어 이름 (예: "reranker_score")
    value: float                       # 스코어 값
    comment: Optional[str] = None      # 부가 설명
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── 베이스 인터페이스 ──────────────────────────────────────────────────────────

class BaseTracer(ABC):
    """트레이싱 인터페이스 — 구현체는 score()와 update_metadata()를 반드시 구현해야 한다."""

    @abstractmethod
    async def score(self, event: TraceEvent) -> None:
        """
        스코어를 기록한다.

        Args:
            event: 기록할 스코어 이벤트
        """
        pass

    @abstractmethod
    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        현재 트레이스의 메타데이터를 업데이트한다.

        Args:
            metadata: 추가할 메타데이터 dict
        """
        pass


# ── 구현체 ────────────────────────────────────────────────────────────────────

class LangfuseTracer(BaseTracer):
    """
    Langfuse 연동 트레이서.

    기존 노드에 반복되던 아래 패턴을 대체한다:
        try:
            trace_id = langfuse_context.get_current_trace_id()
            if trace_id:
                langfuse.score(trace_id=trace_id, name=..., value=..., comment=...)
        except Exception as e:
            logger.debug(f"Langfuse score 기록 실패 (무시): {e}")

    대체 코드:
        await tracer.score(TraceEvent(name=..., value=..., comment=...))
    """

    async def score(self, event: TraceEvent) -> None:
        try:
            from langfuse.decorators import langfuse_context
            from app.core.langfuse_client import langfuse

            trace_id = langfuse_context.get_current_trace_id()
            if trace_id:
                langfuse.score(
                    trace_id=trace_id,
                    name=event.name,
                    value=event.value,
                    comment=event.comment,
                )
        except Exception as e:
            logger.debug(f"[Tracer] score 기록 실패 (무시): {event.name} - {e}")

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        try:
            from langfuse.decorators import langfuse_context
            langfuse_context.update_current_observation(metadata=metadata)
        except Exception as e:
            logger.debug(f"[Tracer] metadata 업데이트 실패 (무시): {e}")


class NoopTracer(BaseTracer):
    """
    아무것도 하지 않는 트레이서.
    Langfuse 없는 환경(테스트, 로컬 개발)에서 사용한다.
    """

    async def score(self, event: TraceEvent) -> None:
        logger.debug(f"[NoopTracer] score: {event.name}={event.value}")

    def update_metadata(self, metadata: Dict[str, Any]) -> None:
        logger.debug(f"[NoopTracer] metadata: {metadata}")


# ── 싱글톤 ────────────────────────────────────────────────────────────────────

# 전체 프로젝트에서 공통으로 사용하는 트레이서 인스턴스
tracer: BaseTracer = LangfuseTracer()
