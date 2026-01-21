from fastapi import APIRouter, HTTPException, File, UploadFile, Response
from app.services.utils.dialogue_converter_util import convert_csv_to_dialogue

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
        print(f"대화록 변환 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")
