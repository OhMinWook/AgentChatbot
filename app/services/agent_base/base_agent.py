"""
Agent 베이스 인터페이스

모든 Agent의 공통 인터페이스를 정의한다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 식별자"""
        pass

    @abstractmethod
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Agent를 실행한다.

        Args:
            query: 사용자 질문 또는 입력
            **kwargs: Agent별 추가 파라미터

        Returns:
            실행 결과 dict (Agent별 구조는 다를 수 있음)
        """
        pass
