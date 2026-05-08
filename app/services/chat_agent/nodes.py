"""
LangGraph 노드 함수 정의
"""

import asyncio
import logging
import time
import json
from typing import Dict, Any

from app.core.langfuse_client import observe, langfuse, langfuse_context  # langfuse 비활성화 스텁

from app.services.chat_agent.graph_state import MainState
from app.services.chat_agent.prompts import (
    AGENT,
    AGENT_STRICT,
    VERIFY_ANSWER,
)
from app.services.chat_agent.node_utils import (
    VERIFY_ANSWER_JSON_SCHEMA,
    QuestionTask,
    HallucinationResult,
    call_llm,
    generate_single_answer,
)
from app.services.api_clients.llm_utils import strip_markdown_codeblock
from app.services.chat_agent.tools import create_search_tool
from app.core.config import settings

logger = logging.getLogger(__name__)


# ── 모듈 레벨 헬퍼 ────────────────────────────────────────────────────────────

def _build_streaming_payload(answer: dict, translate_to) -> dict:
    """answer dict에서 SSE 스트리밍 payload 구성"""
    answer_text = answer.get("answer", "")
    stream_messages = answer.get("messages")
    if answer_text:
        logger.info("[Process] 고위험 답변 precomputed 설정")
        return {"precomputed": True, "content": answer_text, "translate_to": translate_to}
    logger.info("[Process] 저위험 답변 스트리밍 준비")
    return {
        "precomputed": False,
        "messages": stream_messages,
        "max_tokens": settings.DEFAULT_MAX_TOKENS,
        "translate_to": translate_to,
    }


def _record_reranker_scores(all_answers: list) -> None:
    """Langfuse에 reranker 점수와 RAG 문서 수 기록"""
    try:
        trace_id = langfuse_context.get_current_trace_id()
        if not trace_id:
            return
        scores = [
            doc["score"]
            for ans in all_answers
            for doc in ans.get("rag_docs", [])
            if isinstance(doc.get("score"), (int, float))
        ]
        if scores:
            langfuse.create_score(
                trace_id=trace_id,
                name="reranker_score",
                value=round(sum(scores) / len(scores), 4),
                comment=f"청크 {len(scores)}개 평균",
            )
        langfuse.create_score(trace_id=trace_id, name="rag_doc_count", value=len(scores))
        langfuse.create_score(trace_id=trace_id, name="cache_hit", value=0)
    except Exception as e:
        logger.debug(f"[Process] Langfuse reranker_score 기록 실패 (무시): {e}")


def _record_hallucination_score(failed: list) -> None:
    """Langfuse에 hallucination_check 점수 기록"""
    try:
        trace_id = langfuse_context.get_current_trace_id()
        if not trace_id:
            return
        score_value = 0.0 if failed else 1.0
        comment = "; ".join(i for r in failed for i in r.issues) if failed else "검증 통과"
        langfuse.create_score(
            trace_id=trace_id,
            name="hallucination_check",
            value=score_value,
            comment=comment,
        )
    except Exception as e:
        logger.debug(f"[Verify] Langfuse score 기록 실패 (무시): {e}")


async def _verify_single_answer(answer: dict) -> HallucinationResult:
    """단일 답변 할루시네이션 검증 수행"""
    question = answer.get("question", "")
    response = await call_llm(
        [
            {"role": "system", "content": VERIFY_ANSWER.system},
            {"role": "user", "content": VERIFY_ANSWER.user.format(
                context=answer.get("context", ""),
                question=question,
                answer=answer.get("answer", ""),
            )},
        ],
        max_tokens=settings.VERIFY_MAX_TOKENS,
        json_schema=VERIFY_ANSWER_JSON_SCHEMA,
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

    return HallucinationResult(passed=passed, issues=issues, question=question)


# ── process_question ──────────────────────────────────────────────────────────

@observe()
async def process_question_node(state: MainState) -> Dict[str, Any]:
    """질문 처리 노드 - 문서 검색 및 답변 생성"""
    invoke_id = state.get("invoke_id", "")
    original_query = state.get("original_query", "")
    filter_filename = state.get("filter_filename", None)
    retry_count = state.get("retry_count", 0)

    questions = [original_query]
    if not original_query:
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
        generate_single_answer(agent_prompt, QuestionTask(idx, questions[idx], search_results[idx], settings.DEFAULT_MAX_TOKENS, translate_to))
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

    _record_reranker_scores(all_answers)
    logger.info(f"[Process] {len(all_answers)}개 답변 완료")

    streaming_payload = _build_streaming_payload(all_answers[0], translate_to)
    return {"agent_answers": all_answers, "streaming_payload": streaming_payload}


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

    results = await asyncio.gather(*[_verify_single_answer(a) for a in answers_to_verify])
    failed = [r for r in results if not r.passed]
    _record_hallucination_score(failed)

    if not failed:
        logger.info("[Verify] 전체 답변 검증 통과")
        return {"verification_passed": True}

    if retry_count < settings.MAX_VERIFY_RETRIES:
        logger.warning(f"[Verify] {len(failed)}개 실패 → 재시도 ({retry_count + 1}/{settings.MAX_VERIFY_RETRIES})")
        return {
            "verification_passed": False,
            "retry_count": retry_count + 1,
            "agent_answers": [],
            "streaming_payload": None
        }

    logger.warning(f"[Verify] 재시도 횟수 초과 ({retry_count}회) - 그대로 사용")
    return {"verification_passed": True}


