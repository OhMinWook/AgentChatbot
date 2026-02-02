import asyncio
import logging
import time

from fastapi import APIRouter, HTTPException, File, UploadFile, Response

logger = logging.getLogger(__name__)

from app.services.utils.dialogue_converter_util import convert_csv_to_dialogue, split_dialogue_by_date
from app.services.prompt_builders.dialogue_prompt_builder import dialogue_prompt_builder
from app.services.api_clients.llm_client import llm_client
from app.services.utils.sse_utils import SSEType, create_sse_data, create_sse_response

router = APIRouter()

@router.post("/convert/dialogue", summary="CSV 채팅 로그를 대화록 텍스트로 변환")
async def convert_dialogue_from_csv(
        csv_file: UploadFile = File(..., description="업로드할 채팅 로그 CSV 파일")
):
    """
    특정 형식의 CSV 채팅 로그 파일을 업로드하면, 날짜별/사용자별로 메시지를 묶어
    가독성 좋은 대화록 텍스트 파일 형식으로 변환하여 반환합니다.

    - **csv_file**: `sTalkerName`, `formatted_date`, `sTalkerContent`, `sOriginalFileName` 컬럼을 포함하는 CSV 파일
    """
    if not csv_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드할 수 있습니다.")

    try:
        csv_content = await csv_file.read()

        # 유틸리티 함수를 호출하여 대화록 변환
        dialogue_text = convert_csv_to_dialogue(csv_content)

        # 결과를 일반 텍스트로 반환
        return Response(content=dialogue_text, media_type="text/plain; charset=utf-8")

    except ValueError as e:
        # 유틸리티에서 발생한 특정 오류 (예: 컬럼 없음, 인코딩)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 그 외 예상치 못한 오류
        logger.error(f"[Dialogue Convert Error] {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@router.post("/convert/dialogue-summary", summary="CSV 채팅 로그를 LLM으로 요약 (SSE)")
async def summarize_dialogue_from_csv(
        csv_file: UploadFile = File(..., description="업로드할 채팅 로그 CSV 파일")
):
    """
    CSV 채팅 로그를 업로드하면 날짜별 대화 요약과 전체 요약을 SSE 스트리밍으로 반환합니다.

    - **csv_file**: `sTalkerName`, `formatted_date`, `sTalkerContent`, `sOriginalFileName` 컬럼을 포함하는 CSV 파일

    SSE 이벤트:
    - progress: 진행 상황 (step, percent, message)
    - result: 최종 결과 (daily_summaries, overall_summary)
    - done: 완료
    - error: 에러 발생
    """
    if not csv_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드할 수 있습니다.")

    # CSV 파일 먼저 읽기 (SSE 스트림 시작 전)
    csv_content = await csv_file.read()

    async def stream_summary():
        try:
            # 1. CSV 파싱
            yield create_sse_data({
                "type": SSEType.PROGRESS,
                "step": "csv_parse",
                "message": "CSV 파일 분석 중..."
            })

            date_segments = split_dialogue_by_date(csv_content)
            logger.info(f"[Dialogue Summary] 세그먼트 분리 완료: {len(date_segments)}개")

            if not date_segments:
                yield create_sse_data({
                    "type": SSEType.ERROR,
                    "message": "대화 데이터가 비어있습니다."
                })
                return

            # 2. 날짜 기간 전송
            yield create_sse_data({
                "type": SSEType.PROGRESS,
                "start_date": date_segments[0]["date"],
                "end_date": date_segments[-1]["date"]
            })

            # 3. 세그먼트 준비 완료
            yield create_sse_data({
                "type": SSEType.PROGRESS,
                "step": "segments_ready",
                "message": f"{len(date_segments)}개 날짜별 대화 감지"
            })

            # 4. 날짜별 요약 (병렬 처리)
            yield create_sse_data({
                "type": SSEType.PROGRESS,
                "step": "daily_summary",
                "message": f"날짜별 요약 중... ({len(date_segments)}개)"
            })

            batch_start = time.time()

            async def summarize_one(segment: dict, idx: int) -> dict:
                start = time.time()
                logger.info(f"[LLM #{idx}] 시작 - {segment['date']}")

                payload = dialogue_prompt_builder.build_daily_summary_payload(
                    segment["date"], segment["text"]
                )
                response = await llm_client.chat_completions(payload)
                summary = llm_client.extract_content(response)

                elapsed = time.time() - start
                logger.info(f"[LLM #{idx}] 완료 - {segment['date']} ({elapsed:.2f}s)")
                return {"date": segment["date"], "summary": summary}

            daily_summaries = await asyncio.gather(
                *(summarize_one(seg, i) for i, seg in enumerate(date_segments))
            )
            daily_summaries = list(daily_summaries)
            logger.info(f"[Dialogue Summary] 날짜별 요약 완료: {len(daily_summaries)}개")

            # 5. 전체 요약
            yield create_sse_data({
                "type": SSEType.PROGRESS,
                "step": "overall_summary",
                "message": "전체 요약 생성 중..."
            })

            logger.info("[Dialogue Summary] 전체 요약 시작")
            overall_payload = dialogue_prompt_builder.build_overall_summary_payload(daily_summaries)
            overall_response = await llm_client.chat_completions(overall_payload)
            overall_summary = llm_client.extract_content(overall_response)
            logger.info("[Dialogue Summary] 전체 요약 완료")

            yield create_sse_data({
                "type": SSEType.RESULT,
                "daily_summaries": daily_summaries,
                "overall_summary": overall_summary
            })

            yield create_sse_data({
                "type": SSEType.DONE,
                "message": "요약 완료"
            })

        except ValueError as e:
            yield create_sse_data({
                "type": SSEType.ERROR,
                "message": str(e)
            })
        except Exception as e:
            logger.error(f"[Dialogue Summary Error] {e}")
            yield create_sse_data({
                "type": SSEType.ERROR,
                "message": f"서버 내부 오류: {e}"
            })

    return create_sse_response(stream_summary())
