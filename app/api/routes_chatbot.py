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
from app.services.utils.sse_utils import create_sse_data, create_sse_response, SSEType
from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.chat_agent.sse_adapter import sse_graph_adapter
from app.services.api_clients.llm_client import llm_client
from app.services.api_clients.model_server_client import model_server_client, SearchResult
from app.services.prompt_builders.document_summary_prompt_builder import document_summary_prompt_builder

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
                await queue.put(create_sse_data({"type": SSEType.PROGRESS, "percent": percent, "message": message}))

            # 마크다운 결과 콜백 (디버깅용, 100KB 제한)
            async def on_markdown(markdown_content: str):
                logger.debug(f"[SSE] Queueing markdown preview: {len(markdown_content)} chars")
                # SSE 청크 크기 제한 (100KB) - 너무 크면 truncate
                max_size = 100_000
                truncated = len(markdown_content) > max_size
                content_to_send = markdown_content[:max_size] if truncated else markdown_content
                if truncated:
                    content_to_send += f"\n\n... (이하 {len(markdown_content) - max_size:,}자 생략)"
                await queue.put(create_sse_data({
                    "type": SSEType.MARKDOWN_PREVIEW,
                    "content": content_to_send,
                    "length": len(markdown_content),
                    "truncated": truncated
                }))

            # 인덱싱 작업을 별도 태스크로 실행
            async def run_ingestion():
                try:
                    await rag_ingestion_service.ingest_file(
                        final_filename,
                        invokeId,
                        saved_file_path,
                        on_progress=on_progress,
                        on_markdown=on_markdown
                    )
                    # 완료 이벤트
                    await queue.put(create_sse_data({"type": SSEType.DONE, "message": "모든 인덱싱 작업이 완료되었습니다."}))
                except Exception as e:
                    logger.error(f"[Upload Stream Error] {e}")
                    await queue.put(create_sse_data({"type": SSEType.ERROR, "detail": str(e)}))
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
    (정렬: 최신 파일 -> 오래된 파일 순)
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


@router.post("/message/document-summary/{invokeId}", summary="문서 체계적 요약 (SSE)")
async def summarize_document(
        invokeId: str,
        target_filename: str = Form(..., description="요약할 문서 파일명 (확장자 포함)")
):
    """
    업로드된 문서를 체계적으로 요약합니다.

    - **20페이지 미만**: 단순 검색 후 요약 (LLM 1회)
    - **20페이지 이상**: 질문 분해 → 병렬 분석 → 통합 (LLM 5회)

    SSE를 통해 실시간 진행률을 전송합니다.
    """
    async def event_stream():
        try:
            # 1. 문서 메타데이터 조회
            yield create_sse_data({"type": SSEType.PROGRESS, "percent": 5, "message": "문서 정보 확인 중..."})

            doc_meta = await model_server_client.colbert_get_document_metadata(invokeId, target_filename)

            if not doc_meta:
                yield create_sse_data({"type": SSEType.ERROR, "detail": f"문서를 찾을 수 없습니다: {target_filename}"})
                return

            total_pages = doc_meta["total_pages"]

            # 2. 페이지 수에 따른 방식 분기
            if document_summary_prompt_builder.needs_decomposition(total_pages):
                # ===== 20페이지 이상: 질문 분해 방식 =====
                questions = document_summary_prompt_builder.get_questions()
                question_count = len(questions)

                yield create_sse_data({
                    "type": SSEType.PROGRESS,
                    "percent": 10,
                    "message": f"대용량 문서 ({total_pages}페이지), 질문 분해 방식으로 분석"
                })

                # ColBERT 배치 검색
                yield create_sse_data({
                    "type": SSEType.PROGRESS,
                    "percent": 20,
                    "message": "관련 문서 검색 중..."
                })

                query_texts = [q["question"] for q in questions]
                batch_results = await model_server_client.colbert_search_batch(
                    invokeId, query_texts, top_k=settings.COLBERT_TOP_K
                )

                # LLM 병렬 호출 준비
                yield create_sse_data({
                    "type": SSEType.PROGRESS,
                    "percent": 40,
                    "message": f"질문 {question_count}개 병렬 분석 중..."
                })

                all_references = []
                llm_tasks = []

                for i, q in enumerate(questions):
                    search_results = batch_results[i] if i < len(batch_results) else []

                    filtered_chunks = [
                        r for r in search_results
                        if r.metadata.get("source") == target_filename
                    ]

                    if filtered_chunks:
                        for chunk in filtered_chunks:
                            ref = {
                                "source": chunk.metadata.get("source", ""),
                                "page": chunk.metadata.get("page", 0)
                            }
                            if ref not in all_references:
                                all_references.append(ref)

                        context = "\n\n---\n\n".join([c.content for c in filtered_chunks])
                        payload = document_summary_prompt_builder.build_qa_payload(q["question"], context)
                        llm_tasks.append((q["key"], llm_client.chat_completions(payload)))
                    else:
                        llm_tasks.append((q["key"], None))

                # LLM 병렬 실행
                qa_results = {}
                async_tasks = [task for key, task in llm_tasks if task is not None]
                task_keys = [key for key, task in llm_tasks if task is not None]

                if async_tasks:
                    responses = await asyncio.gather(*async_tasks, return_exceptions=True)
                    for key, response in zip(task_keys, responses):
                        if isinstance(response, Exception):
                            logger.error(f"LLM 호출 실패 ({key}): {response}")
                            qa_results[key] = "분석 실패"
                        else:
                            qa_results[key] = llm_client.extract_content(response)

                for key, task in llm_tasks:
                    if task is None:
                        qa_results[key] = "해당 정보 없음"

                # 최종 통합
                yield create_sse_data({
                    "type": SSEType.PROGRESS,
                    "percent": 85,
                    "message": "분석 결과 통합 중..."
                })

                merge_payload = document_summary_prompt_builder.build_merge_payload(qa_results, target_filename)
                merge_response = await llm_client.chat_completions(merge_payload)
                summary = llm_client.extract_content(merge_response)

            else:
                # ===== 20페이지 미만: 단순 검색 방식 =====
                yield create_sse_data({
                    "type": SSEType.PROGRESS,
                    "percent": 20,
                    "message": f"문서 검색 중 ({total_pages}페이지)..."
                })

                # 단일 검색 쿼리
                search_query = f"{target_filename} 요약"
                search_results = await model_server_client.colbert_search(
                    invokeId, search_query, top_k=settings.COLBERT_TOP_K
                )

                filtered_chunks = [
                    r for r in search_results
                    if r.metadata.get("source") == target_filename
                ]

                all_references = []
                if filtered_chunks:
                    for chunk in filtered_chunks:
                        ref = {
                            "source": chunk.metadata.get("source", ""),
                            "page": chunk.metadata.get("page", 0)
                        }
                        if ref not in all_references:
                            all_references.append(ref)

                    context = "\n\n---\n\n".join([c.content for c in filtered_chunks])
                else:
                    yield create_sse_data({"type": SSEType.ERROR, "detail": "문서 내용을 찾을 수 없습니다."})
                    return

                yield create_sse_data({
                    "type": SSEType.PROGRESS,
                    "percent": 50,
                    "message": "요약 생성 중..."
                })

                payload = document_summary_prompt_builder.build_simple_summary_payload(context, target_filename)
                response = await llm_client.chat_completions(payload)
                summary = llm_client.extract_content(response)

            # 레퍼런스 전송
            if all_references:
                all_references.sort(key=lambda x: x.get("page", 0))
                yield create_sse_data({"type": SSEType.REFERENCES, "docs": all_references})

            # 답변 스트리밍
            chunk_size = 6
            for i in range(0, len(summary), chunk_size):
                yield create_sse_data({"type": SSEType.ANSWER, "content": summary[i:i + chunk_size]})

            yield create_sse_data({"type": SSEType.DONE})

        except Exception as e:
            logger.error(f"[Document Summary Error] {e}")
            yield create_sse_data({"type": SSEType.ERROR, "detail": str(e)})

    return create_sse_response(event_stream())
