"""
LangGraph 노드 함수 정의
"""

import json
import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

from app.services.rag_agent.graph_state import MainState, AgentSubState
from app.services.rag_agent.prompts import (
    SUMMARIZE_PROMPT,
    ANALYZE_REWRITE_PROMPT,
    AGENT_PROMPT,
    AGGREGATE_PROMPT,
    DEFAULT_CLARIFICATION_MESSAGE
)
from app.services.rag_agent.tools import create_search_tool
from app.services.utils.memory_service import memory_service
from app.services.clients.llm_client import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)


async def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
    """LLM 호출 헬퍼 함수"""
    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    response = await llm_client.chat_completions(payload)
    return response["choices"][0]["message"]["content"]


async def _call_llm_json(messages: List[Dict[str, str]], max_tokens: int = 1024) -> Dict:
    """LLM 호출 후 JSON 파싱"""
    content = await _call_llm(messages, max_tokens)

    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        content = content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + 3
        end = content.find("```", start)
        content = content[start:end].strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning(f"JSON 파싱 실패: {content[:200]}")
        return {"raw_content": content}


async def summarize_node(state: MainState) -> Dict[str, Any]:
    """대화 요약 노드"""
    invoke_id = state.get("invoke_id", "")
    if not invoke_id:
        return {"conversation_summary": ""}

    history = await memory_service.get_history(invoke_id)
    if not history:
        return {"conversation_summary": ""}

    history_text = ""
    for msg in history[-6:]:
        role = "사용자" if msg["role"] == "user" else "AI"
        history_text += f"{role}: {msg['content']}\n"

    if len(history_text) < 100:
        return {"conversation_summary": history_text.strip()}

    prompt = SUMMARIZE_PROMPT.format(conversation_history=history_text)
    summary = await _call_llm([{"role": "user", "content": prompt}], max_tokens=500)
    logger.info(f"[Summarize] 대화 요약 완료")
    return {"conversation_summary": summary}


async def analyze_rewrite_node(state: MainState) -> Dict[str, Any]:
    """쿼리 분석 및 재작성 노드"""
    original_query = state.get("original_query", "")
    conversation_summary = state.get("conversation_summary", "")

    if not original_query:
        return {
            "question_is_clear": False,
            "clarification_message": "질문이 비어있습니다.",
            "awaiting_human_input": True
        }

    prompt = ANALYZE_REWRITE_PROMPT.format(
        conversation_summary=conversation_summary or "(이전 대화 없음)",
        user_query=original_query
    )
    result = await _call_llm_json([{"role": "user", "content": prompt}])

    is_clear = result.get("is_clear", True)
    clarification_message = result.get("clarification_message")
    rewritten_questions = result.get("rewritten_questions", [original_query])

    logger.info(f"[Analyze] is_clear={is_clear}, questions={rewritten_questions}")

    if not is_clear:
        return {
            "question_is_clear": False,
            "clarification_message": clarification_message or DEFAULT_CLARIFICATION_MESSAGE,
            "rewritten_questions": [],
            "awaiting_human_input": True
        }

    if not rewritten_questions:
        rewritten_questions = [original_query]

    return {
        "question_is_clear": True,
        "rewritten_questions": rewritten_questions,
        "clarification_message": None,
        "awaiting_human_input": False
    }


async def human_input_node(state: MainState) -> Dict[str, Any]:
    """Human-in-the-loop 노드"""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            return {
                "original_query": last_message.content,
                "awaiting_human_input": False,
                "clarification_message": None
            }
    return {"awaiting_human_input": True}


async def process_question_node(state: AgentSubState) -> Dict[str, Any]:
    """개별 질문 처리 에이전트 노드"""
    invoke_id = state.get("invoke_id", "")
    question = state.get("question", "")
    question_index = state.get("question_index", 0)

    if not question:
        return {"final_answer": "질문이 없습니다.", "search_results": []}

    search_tool = create_search_tool(invoke_id)
    search_result = await search_tool.search_with_parent_expansion(question)

    context = search_result.get("context")
    references = search_result.get("references", [])

    if not context:
        return {
            "final_answer": f"'{question}'에 대한 관련 문서를 찾지 못했습니다.",
            "search_results": []
        }

    prompt = AGENT_PROMPT.format(question=question, context=context)
    answer = await _call_llm([{"role": "user", "content": prompt}], max_tokens=settings.DEFAULT_MAX_TOKENS)

    logger.info(f"[Agent {question_index}] 답변 생성 완료")
    return {"final_answer": answer, "search_results": references}


async def aggregate_node(state: MainState) -> Dict[str, Any]:
    """답변 통합 노드"""
    original_query = state.get("original_query", "")
    agent_answers = state.get("agent_answers", [])

    if not agent_answers:
        return {"messages": [AIMessage(content="관련 정보를 찾지 못했습니다.")]}

    if len(agent_answers) == 1:
        answer = agent_answers[0]
        return {"messages": [AIMessage(content=answer.get("answer", ""))]}

    answers_text = ""
    for i, ans in enumerate(agent_answers):
        answers_text += f"\n### 답변 {i+1}\n{ans.get('answer', '')}\n"

    prompt = AGGREGATE_PROMPT.format(original_query=original_query, agent_answers=answers_text)
    final_answer = await _call_llm([{"role": "user", "content": prompt}], max_tokens=settings.DEFAULT_MAX_TOKENS)

    logger.info(f"[Aggregate] 통합 답변 생성 완료")
    return {"messages": [AIMessage(content=final_answer)]}
