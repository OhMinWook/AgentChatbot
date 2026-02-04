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
    original_query: str = ""
    rewritten_questions: List[str] = []
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    clarification_message: Optional[str] = None
    awaiting_human_input: bool = False
    clarification_count: int = 0  # 명확화 요청 횟수
    filter_filename: Optional[str] = None  # 특정 파일 검색 필터
    streaming_payload: Optional[dict] = None  # SSE adapter에서 스트리밍 생성에 사용

    # RAG 답변 충분성 검증 관련
    answer_is_sufficient: bool = True  # RAG 답변이 충분한지 여부
    rag_clarification_count: int = 0  # RAG 불충분으로 인한 재질문 횟수
    rag_validation_result: Optional[dict] = None  # 검증 결과 (found_topics, missing_info 등)


class AgentSubState(TypedDict):
    """개별 에이전트 서브그래프 상태"""
    invoke_id: str
    question: str
    question_index: int
    final_answer: str
    search_results: List[Dict[str, Any]]
