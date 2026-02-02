"""
LangGraph 엣지 및 라우팅 로직
"""

from typing import Literal, List
from langgraph.types import Send
from app.services.chat_agent.graph_state import MainState


def route_after_analyze(state: MainState) -> Literal["human_input", "fan_out_agents"]:
    """분석 후 라우팅"""
    is_clear = state.get("question_is_clear", False)
    return "fan_out_agents" if is_clear else "human_input"


def route_after_human_input(state: MainState) -> Literal["analyze_rewrite", "end"]:
    """Human input 후 라우팅"""
    awaiting = state.get("awaiting_human_input", True)
    return "end" if awaiting else "analyze_rewrite"


def fan_out_to_agents(state: MainState) -> List[Send]:
    """Send API를 사용하여 에이전트를 병렬로 실행"""
    invoke_id = state.get("invoke_id", "")
    questions = state.get("rewritten_questions", [])

    if not questions:
        original = state.get("original_query", "")
        questions = [original] if original else []

    sends = []
    for idx, question in enumerate(questions):
        sends.append(
            Send(
                "process_question",
                {
                    "invoke_id": invoke_id,
                    "question": question,
                    "question_index": idx,
                    "final_answer": "",
                    "search_results": []
                }
            )
        )
    return sends
