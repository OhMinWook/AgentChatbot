"""
LangGraph State 클래스 정의
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


def accumulate_or_reset(left: List[dict], right: List[dict]) -> List[dict]:
    """에이전트 답변을 누적하는 리듀서 함수"""
    if not right:
        return []
    return left + right


class MainState(MessagesState):
    """메인 그래프 상태"""
    invoke_id: str = ""
    question_is_clear: bool = False
    conversation_summary: str = ""
    original_query: str = ""
    rewritten_questions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    clarification_message: Optional[str] = None
    awaiting_human_input: bool = False


class AgentSubState(TypedDict):
    """개별 에이전트 서브그래프 상태"""
    invoke_id: str
    question: str
    question_index: int
    final_answer: str
    search_results: List[Dict[str, Any]]
