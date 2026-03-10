"""
LangGraph 노드 함수 정의
"""

import asyncio
import json
import logging
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, AIMessage

from app.services.chat_agent.graph_state import MainState
from app.services.chat_agent.prompts import (
    ANALYZE_REWRITE,
    AGENT,
    AGENT_STRICT,
    AGGREGATE,
    VERIFY_ANSWER,
    DEFAULT_CLARIFICATION_MESSAGE,
)
from app.services.chat_agent.node_utils import (
    ANALYZE_REWRITE_JSON_SCHEMA,
    VERIFY_ANSWER_JSON_SCHEMA,
    MAX_VERIFY_RETRIES,
    strip_markdown_codeblock,
    call_llm,
    generate_single_answer,
)
from app.services.chat_agent.tools import create_search_tool
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── analyze_rewrite ───────────────────────────────────────────────────────────

async def analyze_rewrite_node(state: MainState) -> Dict[str, Any]:
    """쿼리 분석 및 재작성 노드"""
    messages = state.get("messages", [])
    filter_filename = state.get("filter_filename", None)

    original_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break

    query_for_analysis = original_query
    if filter_filename:
        file_label = filter_filename.rsplit(".", 1)[0]
        query_for_analysis = f"[문서: {file_label}] {original_query}"

    logger.info(f"[Analyze] 쿼리 분석 시작: {query_for_analysis[:100]}...")

    response = await call_llm(
        [
            {"role": "system", "content": ANALYZE_REWRITE.system},
            {"role": "user", "content": ANALYZE_REWRITE.user.format(user_query=query_for_analysis)}
        ],
        max_tokens=512,
        json_schema=ANALYZE_REWRITE_JSON_SCHEMA
    )

    logger.debug(f"[Analyze] LLM 응답: {response[:500] if response else '(빈 응답)'}")

    if not response:
        logger.warning("LLM returned empty response, using original query")
        return {
            "question_is_clear": True,
            "rewritten_questions": [original_query],
            "clarification_message": None
        }

    clarification_count = state.get("clarification_count", 0)

    try:
        result = json.loads(strip_markdown_codeblock(response))
        is_clear = result.get("is_clear", True)
        rewritten_questions = result.get("rewritten_questions", [original_query])
        clarification_message = result.get("clarification_message", DEFAULT_CLARIFICATION_MESSAGE)

        if filter_filename and len(rewritten_questions) > 2:
            rewritten_questions = rewritten_questions[:2]
            logger.info("[Analyze] Private chat: 질문 2개로 제한")

        if filter_filename:
            file_label = filter_filename.rsplit(".", 1)[0]
            cleaned = []
            for q in rewritten_questions:
                if file_label in q:
                    q = q.replace(file_label, "").strip().lstrip("의은는이가에서 ").strip()
                    logger.debug(f"[Analyze] 파일명 제거 후: {q}")
                cleaned.append(q)
            rewritten_questions = cleaned

        logger.info(f"[Analyze] 원본 쿼리: {original_query}")
        logger.info(f"[Analyze] 재작성 ({len(rewritten_questions)}개): {rewritten_questions}")

        if not is_clear and clarification_count >= 2:
            logger.info(f"[Analyze] 명확화 횟수 초과 ({clarification_count}회), 강제 진행")
            is_clear = True

        if not rewritten_questions:
            rewritten_questions = [original_query]

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}, response was: {response[:200]}")
        is_clear = True
        rewritten_questions = [original_query]
        clarification_message = ""

    update = {
        "original_query": original_query,
        "question_is_clear": is_clear,
        "rewritten_questions": rewritten_questions,
        "clarification_message": clarification_message if not is_clear else None,
        "awaiting_human_input": not is_clear,
    }
    if not is_clear:
        update["clarification_count"] = clarification_count + 1

    return update


# ── human_input ───────────────────────────────────────────────────────────────

async def human_input_node(state: MainState) -> Dict[str, Any]:
    """Human-in-the-loop 노드 - 사용자 입력 대기"""
    logger.info("[HumanInput] 사용자 입력 대기 중...")
    # interrupt_before로 일시 정지, 재개 시 update_state로 awaiting_human_input=False 주입
    return {}


# ── process_question ──────────────────────────────────────────────────────────

async def process_question_node(state: MainState) -> Dict[str, Any]:
    """질문 처리 노드 - 문서 검색 및 답변 생성"""
    invoke_id = state.get("invoke_id", "")
    rewritten_questions = state.get("rewritten_questions", [])
    original_query = state.get("original_query", "")
    filter_filename = state.get("filter_filename", None)
    retry_count = state.get("retry_count", 0)

    questions = rewritten_questions if rewritten_questions else [original_query]
    if not questions or not questions[0]:
        return {"agent_answers": []}

    search_invoke_id = invoke_id if filter_filename else settings.GLOBAL_INVOKE_ID
    agent_prompt = AGENT_STRICT if retry_count > 0 else AGENT

    logger.info(f"[Process] {len(questions)}개 질문 처리 시작 (인덱스: {search_invoke_id}, 재시도: {retry_count > 0})")

    search_tool = create_search_tool(search_invoke_id)
    search_results = await search_tool.search_batch(questions, filter_filename=filter_filename)

    all_results = await asyncio.gather(*[
        generate_single_answer(agent_prompt, idx, questions[idx], search_results[idx], settings.DEFAULT_MAX_TOKENS)
        for idx in range(len(questions))
    ])
    all_results = sorted(all_results, key=lambda x: x["idx"])

    all_answers = [{
        "question": r["question"],
        "answer": r["answer"],
        "context": r["context"],
        "messages": r.get("messages"),
        "sources": r["sources"],
        "rag_docs": r.get("rag_docs", [])
    } for r in all_results]

    logger.info(f"[Process] {len(all_answers)}개 답변 완료")
    return {"agent_answers": all_answers}


# ── verify_answer ─────────────────────────────────────────────────────────────

async def verify_answer_node(state: MainState) -> Dict[str, Any]:
    """할루시네이션 검증 노드 - 답변이 문서 내용에 근거하는지 판단"""
    agent_answers = state.get("agent_answers", [])
    retry_count = state.get("retry_count", 0)

    if not agent_answers:
        return {"verification_passed": True}

    answers_to_verify = [a for a in agent_answers if a.get("context")]

    if not answers_to_verify:
        logger.info(f"[Verify] 검증 스킵 ({len(agent_answers)}개 저위험 답변 통과)")
        return {"verification_passed": True}

    async def verify_single(answer: dict) -> dict:
        question = answer.get("question", "")
        response = await call_llm(
            [
                {"role": "system", "content": VERIFY_ANSWER.system},
                {"role": "user", "content": VERIFY_ANSWER.user.format(
                    context=answer.get("context", ""),
                    answer=answer.get("answer", "")
                )}
            ],
            max_tokens=256,
            json_schema=VERIFY_ANSWER_JSON_SCHEMA
        )
        try:
            result = json.loads(strip_markdown_codeblock(response))
            passed = result.get("passed", True)
            issues = result.get("issues", [])
        except json.JSONDecodeError:
            logger.warning(f"[Verify] JSON 파싱 실패 - 통과 처리: {response[:100]}")
            passed, issues = True, []

        if not passed:
            logger.warning(f"[Verify] 검증 실패 | 질문: {question[:50]} | 문제: {issues}")
        else:
            logger.info(f"[Verify] 검증 통과 | 질문: {question[:50]}")

        return {"passed": passed, "issues": issues, "question": question}

    results = await asyncio.gather(*[verify_single(a) for a in answers_to_verify])
    failed = [r for r in results if not r["passed"]]

    if not failed:
        logger.info("[Verify] 전체 답변 검증 통과")
        return {"verification_passed": True}

    if retry_count < MAX_VERIFY_RETRIES:
        logger.warning(f"[Verify] {len(failed)}개 실패 → 재시도 ({retry_count + 1}/{MAX_VERIFY_RETRIES})")
        return {
            "verification_passed": False,
            "retry_count": retry_count + 1,
            "agent_answers": []
        }

    logger.warning(f"[Verify] 재시도 횟수 초과 ({retry_count}회) - 그대로 사용")
    return {"verification_passed": True}


# ── aggregate ─────────────────────────────────────────────────────────────────

async def aggregate_node(state: MainState) -> Dict[str, Any]:
    """답변 통합 노드 - streaming_payload 구성"""
    original_query = state.get("original_query", "")
    agent_answers = state.get("agent_answers", [])

    if not agent_answers:
        fallback = "관련 정보를 찾지 못했습니다."
        return {
            "messages": [AIMessage(content=fallback)],
            "streaming_payload": {"precomputed": True, "content": fallback}
        }

    if len(agent_answers) == 1:
        answer = agent_answers[0]
        stream_messages = answer.get("messages")
        if stream_messages:
            logger.info("[Aggregate] 단일 답변 스트리밍 준비")
            return {
                "messages": [AIMessage(content="")],
                "streaming_payload": {
                    "precomputed": False,
                    "messages": stream_messages,
                    "max_tokens": settings.DEFAULT_MAX_TOKENS
                }
            }
        answer_text = answer.get("answer", "")
        return {
            "messages": [AIMessage(content=answer_text)],
            "streaming_payload": {"precomputed": True, "content": answer_text}
        }

    answers_text = "".join(
        f"\n### 답변 {i+1}\n{ans.get('answer', '')}\n"
        for i, ans in enumerate(agent_answers)
    )
    messages = [
        {"role": "system", "content": AGGREGATE.system},
        {"role": "user", "content": AGGREGATE.user.format(
            original_query=original_query, agent_answers=answers_text
        )}
    ]

    logger.info("[Aggregate] 복수 답변 통합 스트리밍 준비")
    return {
        "messages": [AIMessage(content="")],
        "streaming_payload": {
            "precomputed": False,
            "messages": messages,
            "max_tokens": settings.DEFAULT_MAX_TOKENS
        }
    }
