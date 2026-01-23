import os
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.services.utils.memory_service import memory_service
from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.rag_agent.sse_adapter import sse_graph_adapter

router = APIRouter()


@router.post("/message/{invokeId}", summary="LangGraph 기반 대화 (Agentic RAG)")
async def send_chat_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름"),
        attachFile_extension: Optional[str] = Form(None, description="확장자"),
        attachFile_bin: Optional[UploadFile] = File(None, description="실제 파일")
):
    """
    LangGraph 기반 Agentic RAG 대화 엔드포인트

    - 복잡한 질문 분석 및 분할
    - Human-in-the-loop (불명확한 질문 시 clarification 요청)
    - ColBERT 검색 + Parent-Child 청킹

    SSE 이벤트 타입:
    - progress: 진행 상태
    - clarification_needed: 명확화 필요 (thread_id 포함)
    - references: 참조 문서 목록
    - answer: 최종 답변
    - done: 완료
    - error: 오류
    """
    try:
        # 파일 업로드 처리
        if attachFile_bin:
            upload_dir = os.path.join(settings.UPLOAD_DIR, invokeId)
            os.makedirs(upload_dir, exist_ok=True)
            filename = attachFile_bin.filename
            saved_file_path = os.path.join(upload_dir, filename)
            content = await attachFile_bin.read()
            with open(saved_file_path, "wb") as fp:
                fp.write(content)
            try:
                print(f"📂 [Upload] Start ingesting file: {filename} for room: {invokeId}")
                await rag_ingestion_service.ingest_file(attachFile_name, invokeId, saved_file_path, attachFile_extension)
                print(f"✅ [Upload] Successfully indexed!")
            except Exception as e:
                print(f"⚠️ [Ingestion Failed] {e}")

        # SSE 스트리밍 응답
        async def stream_response():
            full_answer = ""

            async for chunk in sse_graph_adapter.invoke_with_sse(invokeId, message):
                yield chunk

                # 답변 누적 (히스토리 저장용)
                try:
                    chunk_str = chunk.decode('utf-8').strip()
                    if chunk_str.startswith("data:"):
                        json_str = chunk_str[5:].strip()
                        if json_str:
                            data = json.loads(json_str)
                            if data.get("type") == "answer":
                                full_answer = data.get("content", "")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            # 히스토리 저장 (clarification_needed가 아닌 경우)
            if full_answer:
                await memory_service.add_history(invokeId, message, full_answer)
                print(f"📝 [History Saved] invokeId: {invokeId}")

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/{invokeId}/continue", summary="Human-in-the-loop 계속")
async def continue_conversation(
        invokeId: str,
        thread_id: str = Form(..., description="이전 대화 스레드 ID"),
        response: str = Form(..., description="사용자 명확화 응답")
):
    """
    Human-in-the-loop 후 그래프 재개

    clarification_needed 이벤트에서 받은 thread_id와
    사용자의 명확화 응답을 사용하여 대화를 계속합니다.
    """
    try:
        async def stream_continuation():
            full_answer = ""

            async for chunk in sse_graph_adapter.continue_with_sse(invokeId, thread_id, response):
                yield chunk

                try:
                    chunk_str = chunk.decode('utf-8').strip()
                    if chunk_str.startswith("data:"):
                        json_str = chunk_str[5:].strip()
                        if json_str:
                            data = json.loads(json_str)
                            if data.get("type") == "answer":
                                full_answer = data.get("content", "")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            if full_answer:
                await memory_service.add_history(invokeId, response, full_answer)
                print(f"📝 [Continue History Saved] invokeId: {invokeId}")

        return StreamingResponse(stream_continuation(), media_type="text/event-stream")

    except Exception as e:
        print(f"Continue Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
