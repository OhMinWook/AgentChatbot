import os
from fastapi import APIRouter, HTTPException, Form, File, UploadFile
from typing import Optional
from app.core.config import settings
from app.schemas.chatbot import ChatRequest
from app.services.chatbot_prompt_builder import prompt_builder
from app.services.llm_client import llm_client
from app.services.memory_service import memory_service
from app.services.rag_ingestion_service import rag_ingestion_service
from app.services.private_rag_search_service import private_rag_service
from app.services.rag_search_service import rag_service

router = APIRouter()

@router.post("/message/{invokeId}", summary="대화하기 (Multipart/Form-data)")
async def send_chat_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        attachFile_name: Optional[str] = Form(None, description="첨부 파일 이름"),
        attachFile_extension: Optional[str] = Form(None, description="확장자"),
        deepResearch: bool = Form(False, description="심층 리서치 사용 여부"),
        # 바이너리 파일 받기 (Optional)
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
            # 1. 저장할 기본 디렉토리 + 방 번호(invokeId)로 경로 생성
            # 예: uploaded_files/550e84.../
            upload_dir = os.path.join(settings.UPLOAD_DIR, invokeId)

            # 2. 폴더가 없으면 생성 (exist_ok=True: 이미 있어도 에러 안 냄)
            os.makedirs(upload_dir, exist_ok=True)

            # 3. 파일명 결정 (보안을 위해선 UUID로 바꾸기도 하지만, 지금은 원본 유지)
            # attachFile_bin.filename에 사용자가 올린 파일명이 들어있음
            filename = attachFile_bin.filename
            saved_file_path = os.path.join(upload_dir, filename)
            print(saved_file_path)

            # 4. 진짜로 저장 (디스크에 쓰기)
            # 대용량 파일일 수도 있으니 read/write 방식으로 안전하게
            content = await attachFile_bin.read()
            with open(saved_file_path, "wb") as fp:
                fp.write(content)

            try:
                print(f"📂 [Upload] Start ingesting file: {filename} for room: {invokeId}")

                # Unstructured 파싱 -> 임베딩 -> Redis 저장까지 한 번에 수행
                await rag_ingestion_service.ingest_file(attachFile_name, invokeId, saved_file_path)

                print(f"✅ [Upload] Successfully indexed to Redis!")
            except Exception as e:
                print(f"⚠️ [Ingestion Failed] {e}")


        # 2. [Deep Research] 심층 검색 (옵션이 켜졌을 때만) 임시로 만듬
        if deepResearch:
            global_context = await rag_service.search(message)
            return global_context
            #rag_context_parts.append(f"[심층 검색(Web/Global DB) 결과]:\n{global_context}")


        doc_result = await private_rag_service.search(message, invokeId)
        final_rag_context = (
            f"[검색된 개인 문서 내용]:\n{doc_result}"
            if doc_result else None
        )


        history = await memory_service.get_history(invokeId)


        # 4. 프롬프트 빌더에 전달

        llm_payload = prompt_builder.build_openai_payload(
            request_data=request_data,
            history_override=history,
            rag_context=final_rag_context  # 여기에 합쳐진 지식이 들어감
        )

        response_json = await llm_client.chat_completions(llm_payload)


        ai_answer = response_json['choices'][0]['message']['content']
        await memory_service.add_history(invokeId, message, ai_answer)

        return response_json

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))