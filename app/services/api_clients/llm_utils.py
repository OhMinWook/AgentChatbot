"""
LLM 호출 공통 헬퍼

- call_llm: Non-streaming LLM 호출
- stream_llm_tokens: 토큰 단위 스트리밍
- strip_think_blocks: <think> 블록 제거
- strip_markdown_codeblock: 마크다운 코드블록 제거
"""

import json
import logging
import re
from typing import AsyncGenerator, Dict, List, Optional

from app.core.langfuse_client import observe  # langfuse 비활성화 스텁
from app.services.api_clients.llm_client import llm_client
from app.services.utils.llm_payload import build_chat_payload

logger = logging.getLogger(__name__)


def strip_think_blocks(text: str) -> str:
    """<think>...</think> 블록 제거 (닫힌 태그 없는 경우도 처리)"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


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


def extract_json_object(text: str) -> Optional[str]:
    """텍스트에서 첫 번째 JSON 객체({...}) 추출"""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start:end + 1]
    return None


@observe()
async def call_llm(messages: List[Dict[str, str]], max_tokens: int = 2048, json_schema: Optional[Dict] = None) -> str:
    """LLM 호출 헬퍼 — think 블록 자동 제거.

    json_schema가 지정된 경우 <think> 블록을 제거하고 JSON 객체만 추출한다.
    """
    try:
        payload = build_chat_payload(messages, max_tokens=max_tokens, json_schema=json_schema)
        response = await llm_client.chat_completions(payload)
        content = llm_client.extract_content(response)
        if json_schema:
            think_end = content.rfind("</think>")
            if think_end != -1:
                content = content[think_end + len("</think>"):].strip()
            else:
                content = re.sub(r"<think>[^{]*", "", content, flags=re.DOTALL).strip()
            return extract_json_object(content) or content
        return strip_think_blocks(content)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return ""


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
                                    think_buffer = think_buffer[end_idx + len("</think>"):].lstrip("\n")
                                else:
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
                    logger.debug(f"[LLM Stream] JSON 파싱 실패 (스킵): {data_str[:100]}")
                    continue
    except Exception as e:
        logger.error(f"[LLM Stream] error: {e}, tokens={token_count}")
        raise
