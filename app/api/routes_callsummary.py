"""통화 요약 API - SSE 기반 실시간 진행률 제공"""
import asyncio
import logging
import time
from enum import Enum
from typing import AsyncGenerator

from fastapi import APIRouter, File, UploadFile, HTTPException

from app.core.config import settings
from app.services.api_clients.stt_client import stt_client
from app.services.api_clients.llm_client import llm_client
from app.services.prompt_builders.callsummary_prompt_builder import callsummary_prompt_builder
from app.services.utils.sse_utils import create_sse_data, create_sse_response, SSEType
from app.services.utils.file_utils import save_upload_file

logger = logging.getLogger(__name__)


async def preprocess_audio_ffmpeg(audio_bytes: bytes, ext: str) -> bytes:
    """ffmpeg로 오디오 전처리: 통화 대역 필터(300~3400Hz) + 16kHz 업샘플링 + 모노 변환"""
    cmd = [
        settings.FFMPEG_PATH, "-f", ext, "-i", "pipe:0",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav", "pipe:1",
        "-loglevel", "quiet"
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate(input=audio_bytes)

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 전처리 실패 (code={proc.returncode}): {stderr.decode(errors='replace')}")

    return stdout


class SummaryStage(str, Enum):
    """통화 요약 진행 단계"""
    RECEIVING = "파일 수신"
    STT_START = "음성 인식 시작"
    STT_PROCESSING = "음성 인식 처리 중"
    STT_COMPLETE = "음성 인식 완료"
    SUMMARY_START = "요약 생성 시작"
    SUMMARY_PROCESSING = "요약 생성 중"
    COMPLETE = "완료"
    ERROR = "오류 발생"


router = APIRouter()


def _progress(percent: int, stage: SummaryStage, message: str) -> str:
    """진행률 SSE 메시지 생성 헬퍼"""
    return create_sse_data({
        "type": SSEType.PROGRESS,
        "percent": percent,
        "stage": stage.value,
        "message": message
    })


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

    # SSE 스트리밍 인프라만 담당
    async def event_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()
        # event_stream은 asyncio.Queue 패턴으로 전환 → SSE 스트리밍만 담당
        queue = asyncio.Queue()
        result_holder = {"transcript": None, "summary": None, "error": None}

        async def on_progress(percent, stage, message):
            await queue.put(_progress(percent, stage, message))

        async def run():
            try:
                # ========== 1. 파일 수신 및 저장 (0%) ==========
                await on_progress(0, SummaryStage.RECEIVING, "오디오 파일 수신 중...")

                saved_path, _ = await save_upload_file(
                    audio, "call_summary", invoke_id,
                    default_filename="audio.wav"
                )

                await on_progress(10, SummaryStage.RECEIVING, f"파일 저장 완료: {audio.filename or 'audio.wav'}")

                # ========== 2. STT 처리 (10% → 50%) ==========
                await on_progress(15, SummaryStage.STT_START, "음성을 분석하고 있습니다")

                filename = audio.filename or "audio.wav"
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
                audio_bytes = await asyncio.to_thread(lambda: open(saved_path, "rb").read())

                try:
                    audio_bytes = await preprocess_audio_ffmpeg(audio_bytes, ext)
                except Exception as e:
                    logger.warning(f"ffmpeg 전처리 실패, 원본 사용: {e}")

                await on_progress(25, SummaryStage.STT_PROCESSING, "음성을 텍스트로 변환 중...")

                transcript = await stt_client.transcribe_bytes(audio_bytes, filename)
                if not transcript:
                    raise ValueError("STT 결과가 비어있습니다. 오디오 파일을 확인해주세요.")

                result_holder["transcript"] = transcript
                await on_progress(50, SummaryStage.STT_COMPLETE, "음성 인식 완료")

                # ========== 3. LLM 요약 (50% → 95%) ==========
                await on_progress(55, SummaryStage.SUMMARY_START, "요약을 생성하고 있습니다")

                summary = await _summarize_transcript(transcript, on_progress=on_progress)
                result_holder["summary"] = summary

                await on_progress(95, SummaryStage.SUMMARY_PROCESSING, "요약 생성 완료, 결과 정리 중...")

                # ========== 4. 완료 (100%) ==========
                await on_progress(100, SummaryStage.COMPLETE, "처리 완료")

                duration = time.time() - start_time
                await queue.put(create_sse_data({
                    "type": SSEType.RESULT,
                    "transcript": transcript,
                    "summary": summary,
                    "duration_seconds": round(duration, 2)
                }))

            except Exception as e:
                result_holder["error"] = str(e)
                await queue.put(_progress(-1, SummaryStage.ERROR, str(e)))
                await queue.put(create_sse_data({"type": SSEType.ERROR, "detail": str(e)}))
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

        await task

    return create_sse_response(event_stream())


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
        saved_path, _ = await save_upload_file(
            audio, "call_summary", invoke_id,
            default_filename="audio.wav"
        )

        filename = audio.filename or "audio.wav"
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
        audio_bytes = await asyncio.to_thread(lambda: open(saved_path, "rb").read())

        try:
            audio_bytes = await preprocess_audio_ffmpeg(audio_bytes, ext)
        except Exception as e:
            logger.warning(f"ffmpeg 전처리 실패, 원본 사용: {e}")

        transcript = await stt_client.transcribe_bytes(audio_bytes, filename)
        if not transcript:
            raise HTTPException(status_code=400, detail="STT 결과가 비어있습니다.")

        summary = await _summarize_transcript(transcript)

        duration = time.time() - start_time
        return {
            "transcript": transcript,
            "summary": summary,
            "duration_seconds": round(duration, 2)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 청킹/요약 로직을 여기서만 처리, 진행률은 콜백으로 전달
async def _summarize_transcript(transcript: str, on_progress=None) -> str:
    """통화록 요약 (긴 텍스트는 청크 분할 처리)

    Args:
        transcript: 통화록 텍스트
        on_progress: 진행률 콜백 async(percent, stage, message) - SSE용, 없으면 진행률 생략
    """
    if callsummary_prompt_builder.needs_chunking(transcript):
        chunks = callsummary_prompt_builder.split_into_chunks(transcript)
        total_chunks = len(chunks)

        if on_progress:
            await on_progress(60, SummaryStage.SUMMARY_PROCESSING, "긴 통화 내용을 나누어 분석하고 있습니다")

        previous_summary = None
        chunk_summaries = []

        for i, chunk in enumerate(chunks):
            if on_progress:
                progress = 60 + int((i + 1) / total_chunks * 30)
                # 콜백 주입
                await on_progress(progress, SummaryStage.SUMMARY_PROCESSING, "통화 내용을 요약하고 있습니다")

            chunk_payload = callsummary_prompt_builder.build_chunk_payload(
                chunk, i, total_chunks, previous_summary
            )
            chunk_response = await llm_client.chat_completions(chunk_payload)
            previous_summary = llm_client.extract_content(chunk_response)
            chunk_summaries.append(previous_summary)

        if on_progress:
            # 콜백 주입
            await on_progress(92, SummaryStage.SUMMARY_PROCESSING, "요약을 정리하고 있습니다")

        combined = "\n\n---\n\n".join(chunk_summaries)
        final_payload = callsummary_prompt_builder.build_final_summary_payload(combined)
        final_response = await llm_client.chat_completions(final_payload)
        return llm_client.extract_content(final_response)
    else:
        if on_progress:
            await on_progress(70, SummaryStage.SUMMARY_PROCESSING, "통화 내용을 요약하고 있습니다")

        llm_payload = callsummary_prompt_builder.build_summary_payload(transcript)
        llm_response = await llm_client.chat_completions(llm_payload)
        return llm_client.extract_content(llm_response)


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
        content = await transcript_file.read()
        transcript = content.decode("utf-8")

        if not transcript.strip():
            raise HTTPException(status_code=400, detail="텍스트 파일이 비어있습니다.")

        summary = await _summarize_transcript(transcript)

        duration = time.time() - start_time
        return {
            "transcript": transcript,
            "summary": summary,
            "duration_seconds": round(duration, 2)
        }

    except HTTPException:
        raise
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="텍스트 파일 인코딩 오류. UTF-8 형식이어야 합니다.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
