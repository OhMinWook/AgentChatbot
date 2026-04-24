"""
LangGraph 그래프 빌더
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.services.chat_agent.graph_state import MainState
from app.services.chat_agent.nodes import (
    process_question_node,
    verify_answer_node,
)
from app.services.chat_agent.edges import (
    route_after_verify,
)

logger = logging.getLogger(__name__)


def create_rag_graph(checkpointer=None):
    """Agentic RAG 그래프 생성"""

    builder = StateGraph(MainState)

    # 노드 추가
    builder.add_node("process_question", process_question_node)
    builder.add_node("verify_answer", verify_answer_node)

    # 엣지 연결
    builder.add_edge(START, "process_question")

    # process_question → verify_answer → [retry: process_question | pass: END]
    builder.add_edge("process_question", "verify_answer")
    builder.add_conditional_edges(
        "verify_answer",
        route_after_verify,
        {
            "retry": "process_question",
            "end": END
        }
    )

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(checkpointer=checkpointer)

    logger.info("RAG Graph compiled successfully")
    return graph
