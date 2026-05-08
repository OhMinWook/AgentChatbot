"""
DB Agent SSE 어댑터 (토큰 스트리밍 지원)
"""

import json
import logging
import re
import time
from typing import AsyncGenerator, Dict, Any, Optional

from app.core.langfuse_client import observe  # langfuse 비활성화 스텁

from app.services.db_agent.agent import db_main_agent
from app.core.config import settings
from app.services.api_clients.llm_utils import stream_llm_tokens
from app.services.utils.sse_utils import SSEType

logger = logging.getLogger(__name__)


class DBSSEAdapter:
    """DB Agent와 SSE 스트리밍을 연결하는 어댑터"""

    @observe(capture_input=False, capture_output=False)
    async def invoke_with_sse(
        self,
        invoke_id: str,
        user_query: str,
        db_results: list,
    ) -> AsyncGenerator[bytes, None]:
        """DB Agent 실행 및 SSE 스트리밍"""
        t_start = time.perf_counter()

        try:
            yield self._format_sse({"type": SSEType.PROGRESS, "step": "답변을 준비하고 있습니다"})

            payload = await db_main_agent.run(user_query, db_results=db_results)

            async for chunk in self._stream_payload(payload, t_start):
                yield chunk

            yield self._format_sse({"type": SSEType.DONE})

        except Exception as e:
            logger.exception(f"DB Agent SSE error: {e}")
            yield self._format_sse({"type": SSEType.ERROR, "message": str(e)})

    async def _stream_payload(
        self, payload: dict, t_start: float
    ) -> AsyncGenerator[bytes, None]:
        """payload 모드에 따라 SSE 스트리밍"""
        first_token_logged = False

        def log_ttft():
            nonlocal first_token_logged
            if not first_token_logged and t_start:
                logger.info(f"[Timing] TTFT: {time.perf_counter() - t_start:.2f}s")
                first_token_logged = True

        if payload.get("precomputed"):
            content = self._normalize_markdown(payload.get("content", ""))
            for i in range(0, len(content), settings.SSE_CHUNK_SIZE):
                log_ttft()
                yield self._format_sse({"type": SSEType.ANSWER, "content": content[i:i + settings.SSE_CHUNK_SIZE]})
        else:
            messages = payload.get("messages", [])
            max_tokens = payload.get("max_tokens", 2048)
            async for token in stream_llm_tokens(messages, max_tokens):
                if token:
                    token = re.sub(r"\*+", "", token)
                    log_ttft()
                    yield self._format_sse({"type": SSEType.ANSWER, "content": token})

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        text = re.sub(r"\*+", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _format_sse(self, data: Dict[str, Any]) -> bytes:
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n".encode("utf-8")


db_sse_adapter = DBSSEAdapter()
