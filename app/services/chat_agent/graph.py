"""
LangGraph 그래프 빌더
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.services.chat_agent.graph_state import MainState
from app.services.chat_agent.nodes import (
    analyze_rewrite_node,
    human_input_node,
    process_question_node,
    aggregate_node
)
from app.services.chat_agent.edges import (
    route_after_analyze,
    route_after_human_input,
)

logger = logging.getLogger(__name__)


def create_rag_graph(checkpointer=None):
    """Agentic RAG 그래프 생성"""

    builder = StateGraph(MainState)

    # 노드 추가
    builder.add_node("analyze_rewrite", analyze_rewrite_node)
    builder.add_node("human_input", human_input_node)
    builder.add_node("process_question", process_question_node)
    builder.add_node("aggregate", aggregate_node)

    # 엣지 연결
    builder.add_edge(START, "analyze_rewrite")

    builder.add_conditional_edges(
        "analyze_rewrite",
        route_after_analyze,
        {
            "human_input": "human_input",
            "fan_out_agents": "process_question"
        }
    )

    builder.add_conditional_edges(
        "human_input",
        route_after_human_input,
        {
            "analyze_rewrite": "analyze_rewrite",
            "end": END
        }
    )

    # process_question → aggregate → END
    builder.add_edge("process_question", "aggregate")
    builder.add_edge("aggregate", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_input"]
    )

    logger.info("RAG Graph compiled successfully")
    return graph
