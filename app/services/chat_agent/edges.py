"""
LangGraph 엣지 및 라우팅 로직
"""

from typing import Literal
from app.services.chat_agent.graph_state import MainState


def route_after_analyze(state: MainState) -> Literal["human_input", "fan_out_agents"]:
    """분석 후 라우팅"""
    is_clear = state.get("question_is_clear", False)
    return "fan_out_agents" if is_clear else "human_input"


def route_after_human_input(state: MainState) -> Literal["analyze_rewrite", "end"]:
    """Human input 후 라우팅"""
    awaiting = state.get("awaiting_human_input", True)
    return "end" if awaiting else "analyze_rewrite"


def route_after_verify(state: MainState) -> Literal["retry", "aggregate"]:
    """검증 후 라우팅 - 실패 시 process_question 재시도, 통과 시 aggregate"""
    verification_passed = state.get("verification_passed", True)
    return "aggregate" if verification_passed else "retry"
