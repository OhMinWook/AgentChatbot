"""
LangGraph 노드 함수 정의
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from langfuse.decorators import observe, langfuse_context
from app.core.langfuse_client import langfuse

from app.services.chat_agent.graph_state import MainState
from app.services.chat_agent.prompts import (
    AGENT,
    AGENT_STRICT,
    VERIFY_ANSWER,
)
from app.services.chat_agent.node_utils import (
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

# ── process_question ──────────────────────────────────────────────────────────

@observe()
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

    translate_to = state.get("translate_to", None)
    search_tool = create_search_tool(search_invoke_id)

    t0 = time.perf_counter()
    search_results = await search_tool.search_batch(questions, filter_filename=filter_filename)
    t1 = time.perf_counter()
    logger.info(f"[Timing] search_batch ({len(questions)}개): {t1 - t0:.2f}s")

    t2 = time.perf_counter()
    all_results = await asyncio.gather(*[
        generate_single_answer(agent_prompt, idx, questions[idx], search_results[idx], settings.DEFAULT_MAX_TOKENS, translate_to=translate_to)
        for idx in range(len(questions))
    ])
    t3 = time.perf_counter()
    logger.info(f"[Timing] generate_answers ({len(questions)}개): {t3 - t2:.2f}s")

    all_results = sorted(all_results, key=lambda x: x["idx"])

    all_answers = [{
        "question": r["question"],
        "answer": r["answer"],
        "context": r["context"],
        "messages": r.get("messages"),
        "sources": r["sources"],
        "rag_docs": r.get("rag_docs", [])
    } for r in all_results]

    # Langfuse reranker_score 기록
    try:
        trace_id = langfuse_context.get_current_trace_id()
        if trace_id:
            scores = [
                doc["score"]
                for ans in all_answers
                for doc in ans.get("rag_docs", [])
                if isinstance(doc.get("score"), (int, float))
            ]
            if scores:
                avg_score = sum(scores) / len(scores)
                langfuse.score(
                    trace_id=trace_id,
                    name="reranker_score",
                    value=round(avg_score, 4),
                    comment=f"청크 {len(scores)}개 평균"
                )
            langfuse.score(
                trace_id=trace_id,
                name="rag_doc_count",
                value=len(scores),
            )
            langfuse.score(
                trace_id=trace_id,
                name="cache_hit",
                value=0,
            )
    except Exception as e:
        logger.debug(f"[Process] Langfuse reranker_score 기록 실패 (무시): {e}")

    logger.info(f"[Process] {len(all_answers)}개 답변 완료")
    return {"agent_answers": all_answers}


# ── verify_answer ─────────────────────────────────────────────────────────────

@observe()
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
                    question=answer.get("question", ""),
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

    # Langfuse 스코어 기록
    try:
        trace_id = langfuse_context.get_current_trace_id()
        if trace_id:
            score_value = 0.0 if failed else 1.0
            comment = "; ".join(i for r in failed for i in r.get("issues", [])) if failed else "검증 통과"
            langfuse.score(
                trace_id=trace_id,
                name="hallucination_check",
                value=score_value,
                comment=comment,
            )
    except Exception as e:
        logger.debug(f"[Verify] Langfuse score 기록 실패 (무시): {e}")

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

@observe()
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

