import json
import asyncio
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Form, File, UploadFile

from app.core.config import settings

logger = logging.getLogger(__name__)
from app.services.utils.memory_service import memory_service
from app.services.utils.file_utils import save_upload_file
from app.services.utils.sse_utils import create_sse_data, create_sse_response, stream_and_collect_answer, SSEType
from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.chat_agent.sse_adapter import sse_graph_adapter
from app.services.chat_agent.guardrails_impl import chat_guardrails
from app.services.chat_agent.node_utils import stream_llm_tokens
from app.services.chat_agent.tools import create_search_tool
from app.services.chat_agent.document_summary_service import document_summary_service
from app.services.rag.qdrant_service import qdrant_service

router = APIRouter()



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

            # 마크다운 결과 콜백 (디버깅용)
            async def on_markdown(markdown_content: str):
                logger.debug(f"[SSE] Queueing markdown preview: {len(markdown_content)} chars")
                await queue.put(create_sse_data({
                    "type": SSEType.MARKDOWN_PREVIEW,
                    "content": markdown_content,
                    "length": len(markdown_content),
                    "truncated": False
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
                    await queue.put(create_sse_data({"type": SSEType.DONE, "message": "문서 등록이 완료되었습니다."}))
                except Exception as e:
                    logger.error(f"[Upload Stream Error] {e}")
                    await queue.put(create_sse_data({"type": SSEType.ERROR, "detail": "파일 처리 중 오류가 발생했습니다."}))
                finally:
                    # 종료 신호
                    await queue.put(None)

            # 태스크 시작
            task = asyncio.create_task(run_ingestion())

            # 큐 소비 및 스트리밍
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=settings.SSE_QUEUE_TIMEOUT)
                except asyncio.TimeoutError:
                    yield create_sse_data({"type": SSEType.ERROR, "detail": "처리 시간이 초과되었습니다."})
                    task.cancel()
                    return
                if data is None:
                    break
                yield data

            # 태스크 완료 대기 및 예외 전파
            await task

        return create_sse_response(stream_progress())

    except Exception as e:
        logger.error(f"[Upload Failed] {e}")
        raise HTTPException(status_code=500, detail="파일 업로드 중 오류가 발생했습니다.")


@router.post("/message/private/{invokeId}", summary="특정 문서 지정 대화 (Private Search)")
async def send_private_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        target_filename: str = Form(..., description="검색할 대상 파일명 (확장자 포함)"),
        translate_to: Optional[str] = Form(None, description="번역 언어 코드 (en/zh/ja)")
):
    """
    특정 파일 내에서만 정보를 검색하여 답변합니다 (Pinpoint Search).

    - **target_filename**: 반드시 정확한 파일명을 입력해야 합니다. (예: `manual.pdf`)
    - 해당 파일이 없거나 내용이 없으면 답변하지 못할 수 있습니다.
    - **translate_to**: 번역 언어 코드 (en=영어, zh=중국어, ja=일본어). 생략 시 번역 안 함.
    """
    try:
        guard = await chat_guardrails.check_input(message)
        if not guard.allowed:
            raise HTTPException(status_code=400, detail=guard.reason)

        generator = sse_graph_adapter.invoke_with_sse(invokeId, guard.text, filter_filename=target_filename, translate_to=translate_to)
        return create_sse_response(stream_and_collect_answer(generator, invokeId, guard.text, "History (Private)"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Private Message Error] {e}")
        raise HTTPException(status_code=500, detail="요청 처리 중 오류가 발생했습니다.")


@router.post("/message/open/{invokeId}", summary="전체 문서 대화 (Global Search)")
async def send_open_message(
        invokeId: str,
        message: str = Form(..., description="유저 대화 내역"),
        translate_to: Optional[str] = Form(None, description="번역 언어 코드 (en/zh/ja)")
):
    """
    업로드된 모든 문서를 대상으로 정보를 검색하여 답변합니다 (Open/Global Search).

    - **translate_to**: 번역 언어 코드 (en=영어, zh=중국어, ja=일본어). 생략 시 번역 안 함.
    """
    try:
        guard = await chat_guardrails.check_input(message)
        if not guard.allowed:
            raise HTTPException(status_code=400, detail=guard.reason)

        generator = sse_graph_adapter.invoke_with_sse(invokeId, guard.text, filter_filename=None, translate_to=translate_to)
        return create_sse_response(stream_and_collect_answer(generator, invokeId, guard.text, "History (Open)"))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Open Message Error] {e}")
        raise HTTPException(status_code=500, detail="요청 처리 중 오류가 발생했습니다.")



@router.get("/files/{invokeId}", summary="업로드된 파일 목록 조회")
async def get_uploaded_files(invokeId: str):
    """
    특정 invokeId(대화방)에 인덱싱된 파일 목록을 반환합니다.
    Qdrant에서 조회하므로 실제 검색 가능한 파일만 표시됩니다.
    """
    try:
        doc_list = await qdrant_service.get_document_list(invokeId)

        # 파일명만 추출
        files = [doc["file_name"] for doc in doc_list]

        return {"files": files}

    except Exception as e:
        logger.error(f"[File List Error] {e}")
        raise HTTPException(status_code=500, detail="파일 목록 조회 중 오류가 발생했습니다.")


@router.get("/history/{invokeId}", summary="대화 기록 조회")
async def get_chat_history(invokeId: str):
    """
    특정 invokeId(대화방)의 대화 기록을 반환합니다.

    Returns:
        {
            "invokeId": "...",
            "history": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }
    """
    try:
        history = await memory_service.get_history(invokeId)
        return {
            "invokeId": invokeId,
            "history": history or []
        }
    except Exception as e:
        logger.error(f"[History Error] {e}")
        raise HTTPException(status_code=500, detail="대화 기록 조회 중 오류가 발생했습니다.")


@router.post("/message/document-summary/{invokeId}", summary="문서 체계적 요약 (SSE)")
async def summarize_document(
        invokeId: str,
        target_filename: str = Form(..., description="요약할 문서 파일명 (확장자 포함)")
):
    """
    업로드된 문서를 체계적으로 요약합니다.

    - **100청크 미만**: 단순 검색 후 요약 (LLM 1회)
    - **100청크 이상**: 질문 분해 → 병렬 분석 → 통합 (LLM 5회)

    SSE를 통해 실시간 진행률을 전송합니다.
    """
    LARGE_DOC_CHUNK_THRESHOLD = 100

    async def event_stream():
        queue = asyncio.Queue()
        result_holder = {"payload": None, "refs": [], "error": None}

        async def on_progress(percent: int, message: str):
            await queue.put(create_sse_data({"type": SSEType.PROGRESS, "percent": percent, "message": message}))

        async def run():
            try:
                await on_progress(5, "문서 정보 확인 중...")

                doc_list = await qdrant_service.get_document_list(invokeId)
                doc_info = next((d for d in doc_list if d["file_name"] == target_filename), None)

                if not doc_info:
                    result_holder["error"] = f"문서를 찾을 수 없습니다: {target_filename}"
                    return

                search_tool = create_search_tool(invokeId)

                if doc_info["total_chunks"] >= LARGE_DOC_CHUNK_THRESHOLD:
                    payload, refs = await document_summary_service.summarize_large(
                        search_tool, target_filename, on_progress
                    )
                else:
                    payload, refs = await document_summary_service.summarize_small(
                        search_tool, target_filename, on_progress
                    )
                    if payload is None:
                        result_holder["error"] = "문서 내용을 찾을 수 없습니다."
                        return

                result_holder["payload"] = payload
                result_holder["refs"] = refs

            except Exception as e:
                logger.error(f"[Document Summary Error] {e}")
                result_holder["error"] = "문서 요약 중 오류가 발생했습니다."
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())

        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=settings.SSE_QUEUE_TIMEOUT)
            except asyncio.TimeoutError:
                yield create_sse_data({"type": SSEType.ERROR, "detail": "처리 시간이 초과되었습니다."})
                task.cancel()
                return
            if item is None:
                break
            yield item

        await task

        if result_holder["error"]:
            yield create_sse_data({"type": SSEType.ERROR, "detail": result_holder["error"]})
            return

        payload = result_holder["payload"]
        refs = result_holder["refs"]

        if refs:
            refs.sort(key=lambda x: x.get("page", 0))
            yield create_sse_data({"type": SSEType.REFERENCES, "docs": refs})

        async for token in stream_llm_tokens(
            payload["messages"],
            payload.get("max_tokens", settings.DEFAULT_MAX_TOKENS)
        ):
            yield create_sse_data({"type": SSEType.ANSWER, "content": token})

        yield create_sse_data({"type": SSEType.DONE})

    return create_sse_response(event_stream())
