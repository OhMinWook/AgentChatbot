"""SSE (Server-Sent Events) 관련 공통 유틸리티"""
import json
import logging
from enum import Enum
from typing import AsyncGenerator, Dict
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


class SSEType(str, Enum):
    """SSE 이벤트 타입"""
    PROGRESS = "progress"
    DONE = "done"
    ERROR = "error"
    ANSWER = "answer"
    REFERENCES = "references"
    CLARIFICATION = "clarification_needed"
    RESULT = "result"
    RAG_DOCUMENTS = "rag_documents"  # 디버깅용 RAG 검색 결과
    MARKDOWN_PREVIEW = "markdown_preview"  # 디버깅용 마크다운 변환 결과
    TRANSLATION = "translation"  # 번역 결과


def create_sse_data(data: dict) -> str:
    """이벤트 없이 data만 포함하는 SSE 메시지 생성

    Args:
        data: 전송할 데이터 딕셔너리

    Returns:
        SSE 형식 문자열 (data: ...\n\n)
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


SSE_HEADERS: Dict[str, str] = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no"  # nginx 버퍼링 비활성화
}


async def stream_and_collect_answer(
    generator,
    invoke_id: str,
    user_message: str,
    log_label: str = "History",
) -> AsyncGenerator[bytes, None]:
    """SSE 청크를 yield하면서 answer 내용을 누적해 히스토리에 저장한다.

    Args:
        generator: SSE bytes를 yield하는 async generator
        invoke_id: 세션 ID
        user_message: 사용자 질문 (히스토리 저장용)
        log_label: 로그 접두어 (예: "History (Open)", "DB History")
    """
    from app.services.utils.memory_service import memory_service  # 순환 import 방지

    full_answer = ""

    async for chunk in generator:
        yield chunk
        try:
            chunk_str = chunk.decode("utf-8").strip()
            if chunk_str.startswith("data:"):
                json_str = chunk_str[5:].strip()
                if json_str:
                    data = json.loads(json_str)
                    if data.get("type") == SSEType.ANSWER:
                        full_answer += data.get("content", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    if full_answer:
        await memory_service.add_history(invoke_id, user_message, full_answer)
        logger.info(f"[{log_label} Saved] invokeId: {invoke_id}")


def create_sse_response(generator, headers: Dict[str, str] = None) -> StreamingResponse:
    """SSE StreamingResponse 생성

    Args:
        generator: async generator yielding SSE messages
        headers: 추가 헤더 (기본 SSE 헤더에 병합됨)

    Returns:
        StreamingResponse with SSE media type and headers
    """
    final_headers = {**SSE_HEADERS}
    if headers:
        final_headers.update(headers)

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=final_headers
    )
