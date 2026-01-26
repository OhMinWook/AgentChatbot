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


@router.post("/upload/{invokeId}", summary="문서 업로드 및 인덱싱 (SSE)")
async def upload_document(
        invokeId: str,
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름"),
        attachFile_extension: Optional[str] = Form(None, description="확장자"),
        attachFile_bin: UploadFile = File(..., description="실제 파일")
):
    """
    RAG 검색을 위한 문서 업로드 엔드포인트 (SSE 스트리밍)
    
    - 파일을 저장하고 ColBERT 및 LightRAG 인덱싱을 수행합니다.
    - SSE를 통해 실시간 진행률(%)을 제공합니다.
    
    **SSE 이벤트 타입:**
    - `progress`: {"percent": int, "message": str}
    - `done`: {"message": str}
    - `error`: {"detail": str}
    """
    try:
        # 파일 저장 준비
        upload_dir = os.path.join(settings.UPLOAD_DIR, invokeId)
        os.makedirs(upload_dir, exist_ok=True)
        
        filename = attachFile_bin.filename
        if attachFile_name:
            if not os.path.splitext(attachFile_name)[1]:
                attachFile_name += os.path.splitext(filename)[1]
        
        saved_file_path = os.path.join(upload_dir, filename)
        
        # 파일 내용을 미리 읽음 (스트리밍 함수 내부에서 읽으면 File closed 에러 가능성)
        content = await attachFile_bin.read()
        with open(saved_file_path, "wb") as fp:
            fp.write(content)
            
        print(f"📂 [Upload] Start ingesting file: {filename} for room: {invokeId}")

        async def stream_progress():
            try:
                # 초기 진행률 전송
                yield f"event: progress\ndata: {json.dumps({'percent': 0, 'message': '파일 업로드 및 저장 완료'}, ensure_ascii=False)}\n\n"
                
                # 진행률 콜백 함수
                async def on_progress(percent: int, message: str):
                    data = json.dumps({"percent": percent, "message": message}, ensure_ascii=False)
                    yield f"event: progress\ndata: {data}\n\n"

                # 인덱싱 수행 (콜백 전달)
                await rag_ingestion_service.ingest_file(
                    attachFile_name or filename, 
                    invokeId, 
                    saved_file_path, 
                    attachFile_extension,
                    on_progress=on_progress
                )
                
                # 완료 이벤트 전송
                yield f"event: done\ndata: {json.dumps({'message': '모든 인덱싱 작업이 완료되었습니다.'}, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                print(f"⚠️ [Upload Stream Error] {e}")
                yield f"event: error\ndata: {json.dumps({'detail': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            stream_progress(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        print(f"⚠️ [Upload Failed] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/{invokeId}", summary="LangGraph 기반 대화 (Agentic RAG)")
async def send_chat_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        attachFile_name: Optional[str] = Form(None, description="검색할 특정 파일명 (Optional)")
):
    """
    LangGraph 기반 Agentic RAG 대화 엔드포인트

    - 복잡한 질문 분석 및 분할
    - Human-in-the-loop (불명확한 질문 시 clarification 요청)
    - ColBERT 검색 + Parent-Child 청킹
    - attachFile_name 지정 시 해당 파일 내에서만 검색 (Pinpoint Search)

    SSE 이벤트 타입:
    - progress: 진행 상태
    - clarification_needed: 명확화 필요 (thread_id 포함)
    - references: 참조 문서 목록
    - answer: 최종 답변
    - done: 완료
    - error: 오류
    """
    try:
        # SSE 스트리밍 응답
        async def stream_response():
            full_answer = ""

            # 파일명 필터링 정보를 함께 전달
            async for chunk in sse_graph_adapter.invoke_with_sse(invokeId, message, filter_filename=attachFile_name):
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
