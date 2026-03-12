"""
LangGraph 엣지 및 라우팅 로직
"""

from typing import Literal
from app.services.chat_agent.graph_state import MainState


def route_after_verify(state: MainState) -> Literal["retry", "aggregate"]:
    """검증 후 라우팅 - 실패 시 process_question 재시도, 통과 시 aggregate"""
    verification_passed = state.get("verification_passed", True)
    return "aggregate" if verification_passed else "retry"
