"""
Tool Use 베이스 인터페이스

Agent가 사용하는 모든 Tool의 공통 인터페이스를 정의한다.
Tool 종류에 상관없이 Agent는 동일한 방식으로 Tool을 호출할 수 있다.

사용법:
    1. BaseTool 을 상속하여 execute() 구현   → 단일 실행
    2. execute_batch() 는 기본 구현 제공     → 필요 시 오버라이드
    3. name 프로퍼티로 Tool 식별

기본 구현체:
    - execute_batch: execute()를 병렬로 호출하는 기본 구현 제공
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """
    Tool 베이스 인터페이스.

    모든 Tool은 execute()를 구현해야 한다.
    Agent는 Tool 종류와 무관하게 아래 방식으로 호출한다:

        result = await tool.execute(input_text)
        results = await tool.execute_batch(input_list)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool 식별자. 로깅 및 트레이싱에 사용된다."""
        pass

    @abstractmethod
    async def execute(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """
        Tool을 실행한다.

        Args:
            input_text: 입력 텍스트 (질문, 쿼리 등)
            **kwargs:   Tool별 추가 파라미터

        Returns:
            실행 결과 dict (Tool별 구조는 다를 수 있음)
        """
        pass

    async def execute_batch(self, inputs: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        여러 입력을 병렬로 실행한다.
        오버라이드하지 않으면 execute()를 병렬 호출하는 기본 구현을 사용한다.

        Args:
            inputs: 입력 텍스트 목록
            **kwargs: Tool별 추가 파라미터

        Returns:
            실행 결과 dict 목록
        """
        return list(await asyncio.gather(*[
            self.execute(input_text, **kwargs) for input_text in inputs
        ]))
