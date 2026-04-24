"""
통합 라우팅 엔드포인트

문서 업로드 여부에 따라 chat_agent 또는 db_agent로 자동 라우팅
"""

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Form

from app.services.utils.memory_service import memory_service
from app.services.utils.sse_utils import create_sse_response, SSEType
from app.services.chat_agent.guardrails_impl import chat_guardrails
from app.services.router_agent import router_agent

logger = logging.getLogger(__name__)

router = APIRouter()


async def _stream_response(generator, invoke_id: str, user_message: str):
    """SSE 스트리밍 및 히스토리 저장 헬퍼"""
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
        logger.info(f"[Unified History Saved] invokeId: {invoke_id}")


@router.post("/message/{invokeId}", summary="통합 챗봇 (문서 검색 + 장애 이력 DB)")
async def send_unified_message(
    invokeId: str,
    message: str = Form(..., description="사용자 질문"),
    target_filename: Optional[str] = Form(None, description="문서 파일명 (업로드된 경우)"),
    translate_to: Optional[str] = Form(None, description="번역 언어 코드 (en/zh/ja)"),
):
    """
    통합 챗봇 엔드포인트

    - **target_filename 있음**: 해당 문서에서 검색하여 답변
    - **target_filename 없음**: 장애 이력 DB 조회 → 결과 있으면 DB 기반 답변, 없으면 "관련 이력 없음"
    """
    try:
        guard = await chat_guardrails.check_input(message)
        if not guard.allowed:
            raise HTTPException(status_code=400, detail=guard.reason)

        generator = router_agent.route(
            invoke_id=invokeId,
            user_query=guard.text,
            target_filename=target_filename,
            translate_to=translate_to,
        )
        return create_sse_response(_stream_response(generator, invokeId, guard.text))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Unified Message Error] {e}")
        raise HTTPException(status_code=500, detail="요청 처리 중 오류가 발생했습니다.")


@router.get("/history/{invokeId}", summary="대화 기록 조회")
async def get_unified_history(invokeId: str):
    """통합 챗봇 대화 기록 반환"""
    try:
        history = await memory_service.get_history(invokeId)
        return {"invokeId": invokeId, "history": history or []}
    except Exception as e:
        logger.error(f"[Unified History Error] {e}")
        raise HTTPException(status_code=500, detail="대화 기록 조회 중 오류가 발생했습니다.")
