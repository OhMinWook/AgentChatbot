"""통화 요약 API - SSE 기반 실시간 진행률 제공"""
import os
import json
import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from app.core.config import settings
from app.services.clients.stt_client import stt_client
from app.services.clients.llm_client import llm_client
from app.services.prompt_builders.callsummary_prompt_builder import callsummary_prompt_builder
from app.schemas.summary_job import SummaryStage, ProgressEvent, SummaryResult

router = APIRouter()


def create_sse_message(event: str, data: dict) -> str:
    """SSE 형식의 메시지 생성"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/call-summary/{invoke_id}", summary="통화 요약 (SSE 실시간 진행률)")
async def summarize_call(
    invoke_id: str,
    audio: UploadFile = File(..., description="오디오 파일 (wav, mp3 등)")
):
    """
    오디오 파일을 받아 STT → LLM 요약을 수행합니다.
    SSE를 통해 실시간으로 진행률을 전송합니다.

    - **invoke_id**: 고유 요청 식별자
    - **audio**: 업로드할 오디오 파일
    """

    async def event_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()
        saved_path = None

        try:
            # ========== 1. 파일 수신 및 저장 (0%) ==========
            yield create_sse_message("progress", ProgressEvent(
                percent=0,
                stage=SummaryStage.RECEIVING,
                message="오디오 파일 수신 중..."
            ).model_dump())

            # 저장 디렉토리 생성
            upload_dir = os.path.join(settings.UPLOAD_DIR, "call_summary", invoke_id)
            os.makedirs(upload_dir, exist_ok=True)

            # 파일 저장
            filename = audio.filename or "audio.wav"
            saved_path = os.path.join(upload_dir, filename)

            content = await audio.read()
            with open(saved_path, "wb") as f:
                f.write(content)

            yield create_sse_message("progress", ProgressEvent(
                percent=10,
                stage=SummaryStage.RECEIVING,
                message=f"파일 저장 완료: {filename}"
            ).model_dump())

            # ========== 2. STT 처리 (10% → 50%) ==========
            yield create_sse_message("progress", ProgressEvent(
                percent=15,
                stage=SummaryStage.STT_START,
                message="음성 인식 서버에 요청 중..."
            ).model_dump())

            yield create_sse_message("progress", ProgressEvent(
                percent=25,
                stage=SummaryStage.STT_PROCESSING,
                message="음성을 텍스트로 변환 중..."
            ).model_dump())

            transcript = await stt_client.transcribe(saved_path)

            if not transcript:
                raise ValueError("STT 결과가 비어있습니다. 오디오 파일을 확인해주세요.")

            yield create_sse_message("progress", ProgressEvent(
                percent=50,
                stage=SummaryStage.STT_COMPLETE,
                message=f"음성 인식 완료 (텍스트 길이: {len(transcript)}자)"
            ).model_dump())

            # ========== 3. LLM 요약 (50% → 95%) ==========
            yield create_sse_message("progress", ProgressEvent(
                percent=55,
                stage=SummaryStage.SUMMARY_START,
                message="LLM 요약 요청 중..."
            ).model_dump())

            yield create_sse_message("progress", ProgressEvent(
                percent=70,
                stage=SummaryStage.SUMMARY_PROCESSING,
                message="통화 내용 분석 및 요약 생성 중..."
            ).model_dump())

            llm_payload = callsummary_prompt_builder.build_summary_payload(transcript)
            llm_response = await llm_client.chat_completions(llm_payload)

            summary = llm_response['choices'][0]['message']['content']

            yield create_sse_message("progress", ProgressEvent(
                percent=95,
                stage=SummaryStage.SUMMARY_PROCESSING,
                message="요약 생성 완료, 결과 정리 중..."
            ).model_dump())

            # ========== 4. 완료 (100%) ==========
            duration = time.time() - start_time

            yield create_sse_message("progress", ProgressEvent(
                percent=100,
                stage=SummaryStage.COMPLETE,
                message=f"처리 완료 (소요시간: {duration:.1f}초)"
            ).model_dump())

            # 최종 결과 전송
            result = SummaryResult(
                transcript=transcript,
                summary=summary,
                duration_seconds=round(duration, 2)
            )
            yield create_sse_message("result", result.model_dump())

        except Exception as e:
            # 에러 발생 시
            yield create_sse_message("progress", ProgressEvent(
                percent=-1,
                stage=SummaryStage.ERROR,
                message=str(e)
            ).model_dump())
            yield create_sse_message("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # nginx 버퍼링 비활성화
        }
    )


@router.post("/call-summary-sync/{invoke_id}", summary="통화 요약 (동기 방식)")
async def summarize_call_sync(
    invoke_id: str,
    audio: UploadFile = File(..., description="오디오 파일 (wav, mp3 등)")
):
    """
    SSE 없이 단순 JSON 응답을 반환하는 동기 방식 엔드포인트.
    진행률이 필요 없는 경우 사용.
    """
    start_time = time.time()

    try:
        # 1. 파일 저장
        upload_dir = os.path.join(settings.UPLOAD_DIR, "call_summary", invoke_id)
        os.makedirs(upload_dir, exist_ok=True)

        filename = audio.filename or "audio.wav"
        saved_path = os.path.join(upload_dir, filename)

        content = await audio.read()
        with open(saved_path, "wb") as f:
            f.write(content)

        # 2. STT
        transcript = await stt_client.transcribe(saved_path)
        if not transcript:
            raise HTTPException(status_code=400, detail="STT 결과가 비어있습니다.")

        # 3. LLM 요약
        llm_payload = callsummary_prompt_builder.build_summary_payload(transcript)
        llm_response = await llm_client.chat_completions(llm_payload)
        summary = llm_response['choices'][0]['message']['content']

        # 4. 결과 반환
        duration = time.time() - start_time
        return SummaryResult(
            transcript=transcript,
            summary=summary,
            duration_seconds=round(duration, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/call-summary-debug/{invoke_id}", summary="통화 요약 디버그 (텍스트 직접 입력)")
async def summarize_call_debug(
    invoke_id: str,
    transcript_file: UploadFile = File(..., description="통화록 텍스트 파일 (.txt)")
):
    """
    [디버그용] 텍스트 파일을 직접 받아 STT 없이 요약만 수행합니다.
    오디오 없이 통화록 텍스트만으로 요약 결과를 테스트할 때 사용.

    - **invoke_id**: 고유 요청 식별자
    - **transcript_file**: 통화록 텍스트 파일 (.txt)
    """
    start_time = time.time()

    try:
        # 1. 텍스트 파일 읽기
        content = await transcript_file.read()
        transcript = content.decode("utf-8")

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="텍스트 파일이 비어있습니다.")

        # 2. LLM 요약 (STT 건너뜀)
        llm_payload = callsummary_prompt_builder.build_summary_payload(transcript)
        llm_response = await llm_client.chat_completions(llm_payload)
        summary = llm_response['choices'][0]['message']['content']

        # 3. 결과 반환
        duration = time.time() - start_time
        return SummaryResult(
            transcript=transcript,
            summary=summary,
            duration_seconds=round(duration, 2)
        )

    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="텍스트 파일 인코딩 오류. UTF-8 형식이어야 합니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))