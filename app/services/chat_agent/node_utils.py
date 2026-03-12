"""
LangGraph 노드 공용 유틸리티
- LLM 호출 헬퍼
- 스트리밍 토큰 생성기
- 할루시네이션 위험도 판단
"""

import json
import logging
import re
from typing import Dict, List, AsyncGenerator

from app.services.api_clients.llm_client import llm_client
from app.core.config import settings
from app.services.utils.llm_payload import build_chat_payload

logger = logging.getLogger(__name__)

# ── JSON 스키마 ──────────────────────────────────────────────────────────────

VERIFY_ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["passed", "issues"]
}

# ── 할루시네이션 위험도 ────────────────────────────────────────────────────────

MAX_VERIFY_RETRIES = 1

_HIGH_RISK_QUESTION_PATTERN = re.compile(
    r"""
    몇|얼마|언제|기한|날짜|기간|기준|조건|요건|자격|대상  # 구체적 정보 요구
    | 비율|퍼센트|요율|[%％]                              # 수치 요구
    | 금액|수당|급여|임금|보수|한도|상한|하한              # 금액 요구
    | \d+                                               # 질문 자체에 숫자 포함
    | 이상|이하|초과|미만|이내                            # 범위 조건 질문
    """,
    re.VERBOSE
)


def is_high_risk_question(question: str) -> bool:
    """질문이 수치·날짜·조건 등 고위험 답변을 유도하는지 판단"""
    return bool(_HIGH_RISK_QUESTION_PATTERN.search(question))


# ── LLM 호출 헬퍼 ─────────────────────────────────────────────────────────────

def strip_think_blocks(text: str) -> str:
    """<think>...</think> 블록 제거"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def strip_markdown_codeblock(text: str) -> str:
    """```json ... ``` 마크다운 코드블록 제거"""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith("```"):
            text = text[:-3].strip()
    return text


async def call_llm(messages: List[Dict[str, str]], max_tokens: int = 2048, json_schema: Dict = None) -> str:
    """LLM 호출 헬퍼 — think 블록 자동 제거"""
    try:
        payload = build_chat_payload(messages, max_tokens=max_tokens, json_schema=json_schema)
        response = await llm_client.chat_completions(payload)
        content = llm_client.extract_content(response)
        return strip_think_blocks(content)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


# ── process_question 헬퍼 ────────────────────────────────────────────────────

def build_agent_messages(agent_prompt, question: str, doc_res: Dict) -> List[Dict[str, str]] | None:
    """검색 결과로부터 LLM 메시지를 구성한다. context가 없으면 None."""
    doc_context = doc_res.get("context", "")
    if not doc_context:
        return None
    return [
        {"role": "system", "content": agent_prompt.system},
        {"role": "user", "content": agent_prompt.user.format(context=doc_context, question=question)}
    ]


async def generate_single_answer(agent_prompt, idx: int, question: str, doc_res: Dict, max_tokens: int) -> Dict:
    """단일 질문에 대해 검색 결과 기반 답변을 생성한다.
    - 고위험 질문: LLM pre-generate (검증용)
    - 저위험 질문: 스트리밍 준비만 (answer="")
    """
    references = doc_res.get("references", [])
    rag_docs = doc_res.get("results", [])
    context = doc_res.get("context", "")
    messages = build_agent_messages(agent_prompt, question, doc_res)

    if messages is None:
        return {
            "idx": idx, "question": question,
            "answer": f"'{question}'에 대한 관련 문서를 찾지 못했습니다.",
            "context": "", "messages": None, "sources": [], "rag_docs": []
        }

    high_risk = is_high_risk_question(question)
    if high_risk:
        logger.info(f"[Process] 고위험 질문 - pre-generate: {question[:50]}")
        answer = await call_llm(messages, max_tokens=max_tokens)
    else:
        logger.info(f"[Process] 저위험 질문 - 스트리밍 준비: {question[:50]}")
        answer = ""

    return {
        "idx": idx, "question": question,
        "answer": answer,
        "context": context if high_risk else "",
        "messages": messages,
        "sources": references,
        "rag_docs": rag_docs
    }


# ── vLLM 스트리밍 ─────────────────────────────────────────────────────────────

async def stream_llm_tokens(messages: List[Dict[str, str]], max_tokens: int = 2048) -> AsyncGenerator[str, None]:
    """vLLM SSE 스트리밍 응답을 토큰 단위로 yield하는 async generator"""
    payload = build_chat_payload(messages, max_tokens=max_tokens)
    logger.info(f"[LLM Stream] 요청 시작 (max_tokens={max_tokens})")
    token_count = 0

    try:
        stream = await llm_client.chat_completions_stream(payload)
        buffer = ""
        think_buffer = ""
        in_think = False

        async for raw_chunk in stream:
            buffer += raw_chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    logger.info(f"[LLM Stream] 완료 (tokens={token_count})")
                    return
                try:
                    data = json.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    token = delta.get("content")
                    finish_reason = choice.get("finish_reason")

                    if token:
                        think_buffer += token
                        while True:
                            if in_think:
                                end_idx = think_buffer.find("</think>")
                                if end_idx != -1:
                                    in_think = False
                                    think_buffer = think_buffer[end_idx + len("</think>"):]
                                else:
                                    think_buffer = ""
                                    break
                            else:
                                start_idx = think_buffer.find("<think>")
                                if start_idx != -1:
                                    visible = think_buffer[:start_idx]
                                    if visible:
                                        token_count += 1
                                        yield visible
                                    in_think = True
                                    think_buffer = think_buffer[start_idx + len("<think>"):]
                                else:
                                    if think_buffer:
                                        token_count += 1
                                        yield think_buffer
                                    think_buffer = ""
                                    break

                    if finish_reason:
                        logger.info(f"[LLM Stream] finish_reason={finish_reason}, tokens={token_count}")
                        if finish_reason == "length":
                            logger.warning(f"[LLM Stream] 토큰 제한으로 응답 잘림! max_tokens={max_tokens}")

                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"[LLM Stream] error: {e}, tokens={token_count}")
        raise
