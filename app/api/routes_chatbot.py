import os
import json
from typing import Optional, AsyncGenerator, List, Dict
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.schemas.chatbot import ChatRequest
from app.services.prompt_builders.chatbot_prompt_builder import prompt_builder
from app.services.clients.llm_client import llm_client
from app.services.utils.memory_service import memory_service
from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.rag.private_rag_search_service import private_rag_service
from app.services.rag.rag_search_service import rag_service
from app.services.rag_agent.sse_adapter import sse_graph_adapter

router = APIRouter()

@router.post("/message/{invokeId}", summary="대화하기 (Multipart/Form-data)")
async def send_chat_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름"),
        attachFile_extension: Optional[str] = Form(None, description="확장자"),
        deepResearch: bool = Form(False, description="심층 리서치 사용 여부"),
        attachFile_bin: Optional[UploadFile] = File(None, description="실제 파일")
):
    try:
        request_data = ChatRequest(
            message=message,
            attachFile_name=attachFile_name,
            attachFile_extension=attachFile_extension,
            deepResearch=deepResearch
        )

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
                print(f"✅ [Upload] Successfully indexed to Redis!")
            except Exception as e:
                print(f"⚠️ [Ingestion Failed] {e}")

        # [공통] 스트리밍 중 전체 답변을 누적하고, 스트림 종료 후 DB에 저장하는 생성자
        async def stream_and_save(stream: AsyncGenerator[bytes, None], references: Optional[List[Dict]] = None):
            full_ai_answer = ""
            
            # 1. 참조 문서 정보가 있다면 가장 먼저 전송 (OpenAI 호환 포맷은 아니지만 data: 라인으로 전달)
            if references:
                ref_payload = {
                    "type": "references",
                    "docs": references
                }
                yield f"data: {json.dumps(ref_payload, ensure_ascii=False)}\n\n".encode('utf-8')

            # 2. LLM 응답 스트리밍
            async for chunk in stream:
                yield chunk
                try:
                    chunk_str = chunk.decode('utf-8').strip()
                    if chunk_str.startswith("data:"):
                        json_str = chunk_str[5:].strip()
                        if json_str and json_str != "[DONE]":
                            data = json.loads(json_str)
                            if 'choices' in data and data['choices']:
                                delta = data['choices'][0].get('delta', {})
                                content_part = delta.get('content')
                                if content_part:
                                    full_ai_answer += content_part
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass # 파싱 오류는 무시
            
            # 스트림이 모두 끝나면 전체 답변을 DB에 저장
            await memory_service.add_history(invokeId, message, full_ai_answer)
            print(f"📝 [History Saved] invokeId: {invokeId}")

        # 2. [Deep Research] 심층 검색 (옵션이 켜졌을 때)
        if deepResearch:
            print(f"🌎 [Deep Research] invokeId: {invokeId}")
            # rag_service.search는 이제 스트림을 반환
            rag_stream = await rag_service.search(message)
            return StreamingResponse(stream_and_save(rag_stream), media_type="text/event-stream")

        # 3. [Private RAG] 개인화 검색 (일반적인 경우)
        doc_result = await private_rag_service.search(message, invokeId)
        
        # doc_result는 이제 {"context": "...", "references": [...]} 형식임
        context_text = doc_result.get("context")
        references = doc_result.get("references", [])
        
        final_rag_context = (f"[검색된 개인 문서 내용]:\n{context_text}" if context_text else None)
        history = await memory_service.get_history(invokeId)

        llm_payload = prompt_builder.build_openai_payload(
            request_data=request_data,
            history_override=history,
            rag_context=final_rag_context
        )

        # llm_client에서 직접 스트림을 받아 처리
        llm_stream = await llm_client.chat_completions_stream(llm_payload)
        return StreamingResponse(stream_and_save(llm_stream, references), media_type="text/event-stream")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/{invokeId}/agent", summary="LangGraph 에이전트 대화 (Agentic RAG)")
async def send_agent_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름"),
        attachFile_extension: Optional[str] = Form(None, description="확장자"),
        attachFile_bin: Optional[UploadFile] = File(None, description="실제 파일")
):
    """
    LangGraph 기반 Agentic RAG 엔드포인트

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
    if not settings.AGENT_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Agentic RAG가 비활성화되어 있습니다. AGENT_ENABLED=True로 설정하세요."
        )

    try:
        # 파일 업로드 처리 (기존 로직과 동일)
        if attachFile_bin:
            upload_dir = os.path.join(settings.UPLOAD_DIR, invokeId)
            os.makedirs(upload_dir, exist_ok=True)
            filename = attachFile_bin.filename
            saved_file_path = os.path.join(upload_dir, filename)
            content = await attachFile_bin.read()
            with open(saved_file_path, "wb") as fp:
                fp.write(content)
            try:
                print(f"📂 [Agent Upload] Start ingesting file: {filename} for room: {invokeId}")
                await rag_ingestion_service.ingest_file(attachFile_name, invokeId, saved_file_path, attachFile_extension)
                print(f"✅ [Agent Upload] Successfully indexed!")
            except Exception as e:
                print(f"⚠️ [Agent Ingestion Failed] {e}")

        # SSE 스트리밍 응답 (에이전트 답변 후 히스토리 저장)
        async def stream_agent_response():
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
                print(f"📝 [Agent History Saved] invokeId: {invokeId}")

        return StreamingResponse(stream_agent_response(), media_type="text/event-stream")

    except Exception as e:
        print(f"Agent Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message/{invokeId}/continue", summary="Human-in-the-loop 계속")
async def continue_agent_conversation(
        invokeId: str,
        thread_id: str = Form(..., description="이전 대화 스레드 ID"),
        response: str = Form(..., description="사용자 명확화 응답")
):
    """
    Human-in-the-loop 후 그래프 재개

    clarification_needed 이벤트에서 받은 thread_id와
    사용자의 명확화 응답을 사용하여 대화를 계속합니다.

    SSE 이벤트 타입은 /agent 엔드포인트와 동일합니다.
    """
    if not settings.AGENT_ENABLED:
        raise HTTPException(
            status_code=400,
            detail="Agentic RAG가 비활성화되어 있습니다."
        )

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