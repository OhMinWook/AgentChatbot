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
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름 (확장자 포함, 예: report.pdf)"),
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
        
        # 파일명 결정 (클라이언트가 지정한 이름 우선, 없으면 원본 파일명)
        final_filename = attachFile_name if attachFile_name else attachFile_bin.filename
        
        saved_file_path = os.path.join(upload_dir, final_filename)
        
        # 파일 내용을 미리 읽음 (스트리밍 함수 내부에서 읽으면 File closed 에러 가능성)
        content = await attachFile_bin.read()
        with open(saved_file_path, "wb") as fp:
            fp.write(content)
            
        print(f"📂 [Upload] Start ingesting file: {final_filename} for room: {invokeId}")

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
                    final_filename, 
                    invokeId, 
                    saved_file_path, 
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
        # SSE 스트리밍 응답
        async def stream_response():
            full_answer = ""

            # 파일명 필터링 적용
            async for chunk in sse_graph_adapter.invoke_with_sse(invokeId, message, filter_filename=target_filename):
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
                await memory_service.add_history(invokeId, message, full_answer)
                print(f"📝 [History Saved] invokeId: {invokeId} (Private)")

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
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
        # SSE 스트리밍 응답
        async def stream_response():
            full_answer = ""

            # 파일명 필터링 없이 전체 검색 (filter_filename=None)
            async for chunk in sse_graph_adapter.invoke_with_sse(invokeId, message, filter_filename=None):
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
                await memory_service.add_history(invokeId, message, full_answer)
                print(f"📝 [History Saved] invokeId: {invokeId} (Open)")

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
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
    print(f"📥 [Continue Request] invokeId: {invokeId}, thread_id: {thread_id}, response: {response}")

    if not thread_id or not response:
        # 422 에러 원인을 파악하기 위해 로그 출력
        print(f"⚠️ [Continue Validation Failed] Missing thread_id or response")
        raise HTTPException(status_code=422, detail="thread_id와 response는 필수입니다.")

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


@router.get("/files/{invokeId}", summary="업로드된 파일 목록 조회")
async def get_uploaded_files(invokeId: str):
    """
    특정 invokeId(대화방)에 업로드된 파일 이름 목록을 반환합니다.
    """
    try:
        upload_dir = os.path.join(settings.UPLOAD_DIR, invokeId)
        
        if not os.path.exists(upload_dir):
            return {"files": []}
            
        files = [
            f for f in os.listdir(upload_dir) 
            if os.path.isfile(os.path.join(upload_dir, f))
        ]
        
        return {"files": sorted(files)}
        
    except Exception as e:
        print(f"File List Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
