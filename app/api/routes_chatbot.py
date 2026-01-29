import os
import json
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)
from app.services.utils.memory_service import memory_service
from app.services.utils.path_validator import safe_join
from app.services.utils.file_utils import save_upload_file
from app.services.utils.sse_utils import create_sse_data, create_sse_response
from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.rag_agent.sse_adapter import sse_graph_adapter

router = APIRouter()


async def _stream_chat_response(
    generator, 
    invoke_id: str, 
    trigger_message: str, 
    history_label: str = ""
):
    """공통 SSE 스트리밍 및 히스토리 저장 헬퍼"""
    full_answer = ""
    
    async for chunk in generator:
        yield chunk

        # 답변 청크 누적 (히스토리 저장용)
        try:
            chunk_str = chunk.decode('utf-8').strip()
            if chunk_str.startswith("data:"):
                json_str = chunk_str[5:].strip()
                if json_str:
                    data = json.loads(json_str)
                    if data.get("type") == "answer":
                        full_answer += data.get("content", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    # 히스토리 저장 (답변이 있는 경우만)
    if full_answer:
        await memory_service.add_history(invoke_id, trigger_message, full_answer)
        label = f" ({history_label})" if history_label else ""
        logger.info(f"[History Saved] invokeId: {invoke_id}{label}")


@router.post("/upload/{invokeId}", summary="문서 업로드 및 인덱싱 (SSE)")
async def upload_document(
        invokeId: str,
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름 (확장자 포함, 예: report.pdf)"),
        attachFile_bin: UploadFile = File(..., description="실제 파일")
):
    """
    RAG 검색을 위한 문서 업로드 엔드포인트 (SSE 스트리밍)
    """
    # 명확화 대기 중인 세션이 있으면 폐기
    sse_graph_adapter.cancel_pending(invokeId)

    try:
        # 파일 저장 (Path Traversal 방지 포함)
        # attachFile_name이 있으면 UploadFile의 filename을 덮어씀
        if attachFile_name:
            attachFile_bin.filename = attachFile_name

        saved_file_path, _ = await save_upload_file(
            attachFile_bin, invokeId,
            default_filename="uploaded_file"
        )
        final_filename = attachFile_name if attachFile_name else attachFile_bin.filename

        logger.info(f"[Upload] Start ingesting file: {final_filename} for room: {invokeId}")

        async def stream_progress():
            queue = asyncio.Queue()

            # 진행률 콜백 (큐에 넣음)
            async def on_progress(percent: int, message: str):
                logger.debug(f"[SSE] Queueing progress: {percent}% - {message}")
                await queue.put(create_sse_data({"type": "progress", "percent": percent, "message": message}))

            # 인덱싱 작업을 별도 태스크로 실행
            async def run_ingestion():
                try:
                    await rag_ingestion_service.ingest_file(
                        final_filename,
                        invokeId,
                        saved_file_path,
                        on_progress=on_progress
                    )
                    # 완료 이벤트
                    await queue.put(create_sse_data({"type": "done", "message": "모든 인덱싱 작업이 완료되었습니다."}))
                except Exception as e:
                    logger.error(f"[Upload Stream Error] {e}")
                    await queue.put(create_sse_data({"type": "error", "detail": str(e)}))
                finally:
                    # 종료 신호
                    await queue.put(None)

            # 태스크 시작
            task = asyncio.create_task(run_ingestion())

            # 큐 소비 및 스트리밍
            while True:
                data = await queue.get()
                if data is None:
                    break
                yield data

            # 태스크 완료 대기 및 예외 전파
            await task

        return create_sse_response(stream_progress())

    except Exception as e:
        logger.error(f"[Upload Failed] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/private/{invokeId}", summary="특정 문서 지정 대화 (Private Search)")
async def send_private_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        target_filename: str = Form(..., description="검색할 대상 파일명 (확장자 포함)")
):
    """
    특정 파일 내에서만 정보를 검색하여 답변합니다 (Pinpoint Search).
    
    - **target_filename**: 반드시 정확한 파일명을 입력해야 합니다. (예: `manual.pdf`)
    - 해당 파일이 없거나 내용이 없으면 답변하지 못할 수 있습니다.
    """
    try:
        # SSE 생성기 생성
        generator = sse_graph_adapter.invoke_with_sse(invokeId, message, filter_filename=target_filename)
        
        # 공통 헬퍼로 스트리밍 반환
        return create_sse_response(_stream_chat_response(generator, invokeId, message, "Private"))

    except Exception as e:
        logger.error(f"[Private Message Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/open/{invokeId}", summary="전체 문서 대화 (Global Search)")
async def send_open_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역")
):
    """
    업로드된 모든 문서를 대상으로 정보를 검색하여 답변합니다 (Open/Global Search).
    """
    try:
        # SSE 생성기 생성
        generator = sse_graph_adapter.invoke_with_sse(invokeId, message, filter_filename=None)
        
        # 공통 헬퍼로 스트리밍 반환
        return create_sse_response(_stream_chat_response(generator, invokeId, message, "Open"))

    except Exception as e:
        logger.error(f"[Open Message Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/{invokeId}/continue", summary="Human-in-the-loop 계속")
async def continue_conversation(
        invokeId: str,
        thread_id: Optional[str] = Form(None, description="이전 대화 스레드 ID"),
        response: Optional[str] = Form(None, description="사용자 명확화 응답")
):
    """
    Human-in-the-loop 후 그래프 재개

    clarification_needed 이벤트에서 받은 thread_id와
    사용자의 명확화 응답을 사용하여 대화를 계속합니다.
    """
    logger.info(f"[Continue Request] invokeId: {invokeId}, thread_id: {thread_id}, response: {response}")

    if not thread_id or not response:
        logger.warning(f"[Continue Validation Failed] Missing thread_id or response")
        raise HTTPException(status_code=422, detail="thread_id와 response는 필수입니다.")

    try:
        # SSE 생성기 생성
        generator = sse_graph_adapter.continue_with_sse(invokeId, thread_id, response)
        
        # 공통 헬퍼로 스트리밍 반환
        return create_sse_response(_stream_chat_response(generator, invokeId, response, "Continue"))

    except Exception as e:
        logger.error(f"[Continue Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files/{invokeId}", summary="업로드된 파일 목록 조회")
async def get_uploaded_files(invokeId: str):
    """
    특정 invokeId(대화방)에 업로드된 파일 이름 목록을 반환합니다.
    (정렬: 오래된 파일 -> 최신 파일 순)
    """
    try:
        upload_dir = safe_join(settings.UPLOAD_DIR, invokeId)

        if not os.path.exists(upload_dir):
            return {"files": []}
            
        # 파일명과 전체 경로를 함께 가져옴
        files_with_path = []
        for f in os.listdir(upload_dir):
            full_path = os.path.join(upload_dir, f)
            if os.path.isfile(full_path):
                files_with_path.append((f, full_path))
        
        # 수정 시간(getmtime) 기준으로 내림차순 정렬 (최신 -> 오래된 것)
        files_with_path.sort(key=lambda x: os.path.getmtime(x[1]), reverse=True)
        
        # 파일명만 추출하여 반환
        sorted_files = [f[0] for f in files_with_path]
        
        return {"files": sorted_files}
        
    except Exception as e:
        logger.error(f"[File List Error] {e}")
        raise HTTPException(status_code=500, detail=str(e))
