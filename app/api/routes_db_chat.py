"""
DB 챗봇 라우터

장애 원인 분석 DB 챗봇 엔드포인트
"""

import json
import logging
from fastapi import APIRouter, HTTPException, Form

from app.services.utils.memory_service import memory_service
from app.services.utils.sse_utils import create_sse_response, stream_and_collect_answer
from app.services.chat_agent.guardrails_impl import chat_guardrails
from app.services.db_agent.sse_adapter import db_sse_adapter

logger = logging.getLogger(__name__)

router = APIRouter()



@router.post("/message/db/{invokeId}", summary="장애 원인 분석 DB 챗봇 (SSE)")
async def send_db_message(
    invokeId: str,
    message: str = Form(..., description="유저 질문"),
):
    """
    DB 데이터를 기반으로 장애 원인 및 해결책을 답변합니다.
    """
    try:
        guard = await chat_guardrails.check_input(message)
        if not guard.allowed:
            raise HTTPException(status_code=400, detail=guard.reason)

        # NOTE: DBTool 미구현 — 실제 DB 조회 없이 빈 결과로 동작
        db_results = []

        generator = db_sse_adapter.invoke_with_sse(
            invoke_id=invokeId,
            user_query=guard.text,
            db_results=db_results,
        )
        return create_sse_response(stream_and_collect_answer(generator, invokeId, guard.text, "DB History"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DB Message Error] {e}")
        raise HTTPException(status_code=500, detail="요청 처리 중 오류가 발생했습니다.")


@router.get("/history/db/{invokeId}", summary="DB 챗봇 대화 기록 조회")
async def get_db_chat_history(invokeId: str):
    """
    DB 챗봇 대화 기록을 반환합니다.
    """
    try:
        history = await memory_service.get_history(invokeId)
        return {
            "invokeId": invokeId,
            "history": history or []
        }
    except Exception as e:
        logger.error(f"[DB History Error] {e}")
        raise HTTPException(status_code=500, detail="대화 기록 조회 중 오류가 발생했습니다.")
