"""
LLM 요청 payload 생성 유틸리티

vLLM /v1/chat/completions 요청에 필요한 dict 조립을 중앙화한다.
"""
from typing import Optional

from app.core.config import settings


def build_chat_payload(
    messages: list[dict],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    json_schema: Optional[dict] = None,
) -> dict:
    """
    vLLM chat completions 요청용 payload 생성
    설정 변경이나 vLLM 옵션 추가 시 이 곳만 변경하면 됨.

    Args:
        messages: [{"role": "system"/"user"/"assistant", "content": "..."}, ...]
        max_tokens: 최대 토큰 수 (None이면 settings.DEFAULT_MAX_TOKENS 사용)
        temperature: 샘플링 온도 (None이면 settings.DEFAULT_TEMPERATURE 사용)
        json_schema: structured output JSON 스키마 (None이면 미적용)
    """
    payload = {
        "model": settings.VLLM_MODEL,
        "messages": messages,
        "max_tokens": max_tokens if max_tokens is not None else settings.DEFAULT_MAX_TOKENS,
        "temperature": temperature if temperature is not None else settings.DEFAULT_TEMPERATURE,
    }

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    if json_schema:
        extra_body["structured_outputs"] = {"json": json_schema}

    payload["extra_body"] = extra_body

    return payload
