import asyncio
import logging

from fastapi import APIRouter, HTTPException, File, UploadFile, Response

logger = logging.getLogger(__name__)

from app.services.utils.dialogue_converter_util import convert_csv_to_dialogue, split_dialogue_by_date
from app.services.prompt_builders.dialogue_prompt_builder import dialogue_prompt_builder
from app.services.clients.llm_client import llm_client

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


@router.post("/convert/dialogue-summary", summary="CSV 채팅 로그를 LLM으로 요약")
async def summarize_dialogue_from_csv(
        csv_file: UploadFile = File(..., description="업로드할 채팅 로그 CSV 파일")
):
    """
    CSV 채팅 로그를 업로드하면 날짜별 대화 요약과 전체 요약을 JSON으로 반환합니다.

    - **csv_file**: `sTalkerName`, `formatted_date`, `sTalkerContent`, `sOriginalFileName` 컬럼을 포함하는 CSV 파일

    응답 형식:
    ```json
    {
      "daily_summaries": [{"date": "2025-01-15", "summary": "..."}],
      "overall_summary": "전체 대화 내용 요약..."
    }
    ```
    """
    if not csv_file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드할 수 있습니다.")

    try:
        csv_content = await csv_file.read()
        date_segments = split_dialogue_by_date(csv_content)

        if not date_segments:
            raise HTTPException(status_code=400, detail="대화 데이터가 비어있습니다.")

        # 날짜별 LLM 요약 (병렬 처리)
        async def summarize_one(segment: dict) -> dict:
            payload = dialogue_prompt_builder.build_daily_summary_payload(
                segment["date"], segment["text"]
            )
            response = await llm_client.chat_completions(payload)
            summary = llm_client.extract_content(response)
            return {"date": segment["date"], "summary": summary}

        daily_summaries = await asyncio.gather(
            *(summarize_one(seg) for seg in date_segments)
        )
        daily_summaries = list(daily_summaries)

        # 전체 요약 LLM 호출
        overall_payload = dialogue_prompt_builder.build_overall_summary_payload(daily_summaries)
        overall_response = await llm_client.chat_completions(overall_payload)
        overall_summary = llm_client.extract_content(overall_response)

        return {
            "daily_summaries": daily_summaries,
            "overall_summary": overall_summary,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Dialogue Summary Error] {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")
