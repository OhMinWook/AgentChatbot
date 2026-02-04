"""
LangGraph 노드 함수 정의
"""

import json
import logging
from typing import Dict, Any, List, AsyncGenerator

from langchain_core.messages import HumanMessage, AIMessage

from app.services.chat_agent.graph_state import MainState, AgentSubState
from app.services.chat_agent.prompts import (
    ANALYZE_REWRITE_PROMPT,
    AGENT_PROMPT,
    AGGREGATE_PROMPT,
    DEFAULT_CLARIFICATION_MESSAGE,
)
from app.services.chat_agent.tools import create_search_tool
from app.services.api_clients.llm_client import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)


async def _call_llm(messages: List[Dict[str, str]], max_tokens: int = 2048, json_schema: Dict = None) -> str:
    """LLM 호출 헬퍼 함수"""
    try:
        payload = {
            "model": settings.VLLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0
        }

        # vLLM structured_outputs (JSON 양식 고정)
        if json_schema:
            payload["extra_body"] = {
                "structured_outputs": {"json": json_schema}
            }

        response = await llm_client.chat_completions(payload)
        content = llm_client.extract_content(response)
        return content.strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


async def analyze_rewrite_node(state: MainState) -> Dict[str, Any]:
    """쿼리 분석 및 재작성 노드"""
    messages = state.get("messages", [])
    filter_filename = state.get("filter_filename", None)

    # 원본 쿼리 추출 (마지막 HumanMessage)
    original_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            original_query = msg.content
            break

    # private chat: 파일명을 쿼리에 포함하여 문맥 제공
    query_for_analysis = original_query
    if filter_filename:
        file_label = filter_filename.rsplit(".", 1)[0]
        query_for_analysis = f"[문서: {file_label}] {original_query}"

    logger.info(f"[Analyze] 쿼리 분석 시작: {query_for_analysis[:100]}...")

    prompt = ANALYZE_REWRITE_PROMPT.format(user_query=query_for_analysis)

    # JSON 스키마로 구조화된 응답 요청
    json_schema = {
        "type": "object",
        "properties": {
            "is_clear": {"type": "boolean"},
            "clarification_message": {"type": "string"},
            "rewritten_questions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "reasoning": {"type": "string"}
        },
        "required": ["is_clear", "rewritten_questions"]
    }

    response = await _call_llm(
        [{"role": "user", "content": prompt}],
        max_tokens=1024,
        json_schema=json_schema
    )

    logger.debug(f"[Analyze] LLM 응답: {response[:500] if response else '(빈 응답)'}")

    # 빈 응답 처리
    if not response:
        logger.warning("LLM returned empty response, using original query")
        return {
            "question_is_clear": True,
            "rewritten_questions": [original_query],
            "clarification_message": None
        }

    # 마크다운 코드블록 제거 (```json ... ```)
    clean_response = response.strip()
    if clean_response.startswith("```"):
        # 첫 줄 제거 (```json)
        first_newline = clean_response.find("\n")
        if first_newline != -1:
            clean_response = clean_response[first_newline + 1:]
        # 마지막 ``` 제거
        if clean_response.endswith("```"):
            clean_response = clean_response[:-3].strip()

    clarification_count = state.get("clarification_count", 0)

    try:
        result = json.loads(clean_response)
        is_clear = result.get("is_clear", True)
        rewritten_questions = result.get("rewritten_questions", [original_query])
        clarification_message = result.get("clarification_message", DEFAULT_CLARIFICATION_MESSAGE)

        # 쿼리 분석 결과 로깅
        logger.info(f"[Analyze] 원본 쿼리: {original_query}")
        logger.info(f"[Analyze] 재작성 ({len(rewritten_questions)}개): {rewritten_questions}")

        # 명확화를 이미 1회 요청했으면 불명확하더라도 강제 진행
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

    # 명확화 요청 시 카운트 증가
    if not is_clear:
        update["clarification_count"] = clarification_count + 1

    return update


async def human_input_node(state: MainState) -> Dict[str, Any]:
    """Human-in-the-loop 노드 - 사용자 입력 대기"""
    logger.info("[HumanInput] 사용자 입력 대기 중...")

    # 이 노드는 interrupt_before에 의해 그래프가 일시 정지됨
    # SSE를 통해 클라이언트에 clarification_needed 이벤트를 보내고
    # 클라이언트가 응답하면 그래프가 재개됨

    # 중요: 재개 시 외부에서 update_state로 awaiting_human_input=False를 주입하므로
    # 여기서 True를 반환하면 안 됨 (상태 덮어쓰기 방지)
    return {}


async def process_question_node(state: MainState) -> Dict[str, Any]:
    """질문 처리 노드 - 문서 검색 기반 답변 생성"""
    import asyncio

    invoke_id = state.get("invoke_id", "")
    rewritten_questions = state.get("rewritten_questions", [])
    original_query = state.get("original_query", "")
    filter_filename = state.get("filter_filename", None)  # 파일명 필터

    questions = rewritten_questions if rewritten_questions else [original_query]

    if not questions or not questions[0]:
        return {"agent_answers": []}

    # Open chat은 글로벌 인덱스 사용, Private chat은 사용자별 invoke_id 사용
    search_invoke_id = invoke_id if filter_filename else settings.GLOBAL_INVOKE_ID

    logger.info(f"[Process] {len(questions)}개 질문 배치 처리 시작 (인덱스: {search_invoke_id}, 필터: {filter_filename})")
    for i, q in enumerate(questions):
        logger.debug(f"  질문 {i+1}: {q}")

    # 1. 배치 검색 (모델 서버 원격 인덱스)
    search_tool = create_search_tool(search_invoke_id)
    search_results = await search_tool.search_batch(questions, filter_filename=filter_filename)

    # 2. 프롬프트 구성 + 답변 생성
    def build_prompt(question: str, doc_res: Dict) -> str | None:
        """검색 결과로부터 LLM 프롬프트를 구성한다. 결과가 없으면 None."""
        doc_context = doc_res.get("context", "")
        if not doc_context:
            return None
        return AGENT_PROMPT.format(context=doc_context, question=question)

    single_question = len(questions) == 1

    if single_question:
        # 단일 질문: LLM 호출을 하지 않고 프롬프트만 저장 → SSE adapter에서 스트리밍
        doc_res = search_results[0]
        prompt = build_prompt(questions[0], doc_res)

        if prompt is None:
            no_result_msg = f"'{questions[0]}'에 대한 관련 문서를 찾지 못했습니다."
            all_answers = [{"question": questions[0], "answer": no_result_msg, "sources": [], "rag_docs": [], "prompt": None}]
        else:
            all_answers = [{
                "question": questions[0],
                "answer": "",  # 스트리밍에서 생성될 예정
                "sources": doc_res.get("references", []),
                "rag_docs": doc_res.get("results", []),  # 디버깅용 검색 결과
                "prompt": prompt
            }]

        logger.info("[Process] 단일 질문 - 스트리밍 준비 완료")
        return {"agent_answers": all_answers}

    # 복수 질문: 각각 LLM 호출 (통합 시 필요)
    async def generate_answer(idx: int, question: str, doc_res: Dict):
        references = doc_res.get("references", [])
        rag_docs = doc_res.get("results", [])
        prompt = build_prompt(question, doc_res)

        if prompt is None:
            return {
                "idx": idx,
                "question": question,
                "answer": f"'{question}'에 대한 관련 문서를 찾지 못했습니다.",
                "sources": [],
                "rag_docs": []
            }

        answer = await _call_llm([{"role": "user", "content": prompt}], max_tokens=settings.DEFAULT_MAX_TOKENS)
        return {"idx": idx, "question": question, "answer": answer, "sources": references, "rag_docs": rag_docs}

    tasks = [
        generate_answer(idx, questions[idx], search_results[idx])
        for idx in range(len(questions))
    ]

    logger.info(f"[Process] {len(tasks)}개 질문 답변 생성 중...")
    all_results = await asyncio.gather(*tasks)

    # 원래 순서대로 정렬 (idx 기준)
    all_results.sort(key=lambda x: x["idx"])

    all_answers = [{
        "question": r["question"],
        "answer": r["answer"],
        "sources": r["sources"],
        "rag_docs": r.get("rag_docs", [])
    } for r in all_results]

    logger.info(f"[Process] {len(all_answers)}개 답변 완료")
    return {"agent_answers": all_answers}


async def aggregate_node(state: MainState) -> Dict[str, Any]:
    """답변 통합 노드 - LLM 직접 호출 대신 streaming_payload를 저장"""
    original_query = state.get("original_query", "")
    agent_answers = state.get("agent_answers", [])

    if not agent_answers:
        return {
            "messages": [AIMessage(content="관련 정보를 찾지 못했습니다.")],
            "streaming_payload": {"precomputed": True, "content": "관련 정보를 찾지 못했습니다."}
        }

    if len(agent_answers) == 1:
        answer = agent_answers[0]
        prompt = answer.get("prompt")
        if prompt:
            # 단일 질문 + 프롬프트 있음 → vLLM 스트리밍
            logger.info("[Aggregate] 단일 답변 스트리밍 준비")
            return {
                "messages": [AIMessage(content="")],
                "streaming_payload": {
                    "precomputed": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": settings.DEFAULT_MAX_TOKENS
                }
            }
        # 프롬프트 없음 (검색 결과 없음 등) → 이미 완성된 텍스트
        answer_text = answer.get("answer", "")
        return {
            "messages": [AIMessage(content=answer_text)],
            "streaming_payload": {"precomputed": True, "content": answer_text}
        }

    # 복수 답변: LLM 통합이 필요 → 호출 정보만 저장
    answers_text = ""
    for i, ans in enumerate(agent_answers):
        answers_text += f"\n### 답변 {i+1}\n{ans.get('answer', '')}\n"

    prompt = AGGREGATE_PROMPT.format(original_query=original_query, agent_answers=answers_text)
    messages = [{"role": "user", "content": prompt}]

    logger.info(f"[Aggregate] 복수 답변 통합 스트리밍 준비")
    return {
        "messages": [AIMessage(content="")],  # placeholder (스트리밍 후 갱신 불필요)
        "streaming_payload": {
            "precomputed": False,
            "messages": messages,
            "max_tokens": settings.DEFAULT_MAX_TOKENS
        }
    }


async def stream_llm_tokens(messages: List[Dict[str, str]], max_tokens: int = 2048) -> AsyncGenerator[str, None]:
    """vLLM SSE 스트리밍 응답을 토큰 단위로 yield하는 async generator"""
    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0
    }

    try:
        stream = await llm_client.chat_completions_stream(payload)
        buffer = ""
        async for raw_chunk in stream:
            buffer += raw_chunk.decode("utf-8", errors="replace")
            # SSE 라인 단위로 파싱
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"stream_llm_tokens error: {e}")
        raise
