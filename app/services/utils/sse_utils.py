"""SSE (Server-Sent Events) 관련 공통 유틸리티"""
import json
from typing import Dict, Any
from fastapi.responses import StreamingResponse


def create_sse_message(event: str, data: dict) -> str:
    """SSE 형식의 메시지 생성

    Args:
        event: SSE 이벤트 이름 (예: 'progress', 'result', 'error')
        data: 전송할 데이터 딕셔너리

    Returns:
        SSE 형식 문자열 (event: ...\ndata: ...\n\n)
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
