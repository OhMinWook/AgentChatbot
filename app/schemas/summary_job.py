from pydantic import BaseModel
from typing import Optional
from enum import Enum


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


class ProgressEvent(BaseModel):
    """SSE로 전송되는 진행률 이벤트"""
    percent: int
    stage: SummaryStage
    message: Optional[str] = None


class SummaryResult(BaseModel):
    """최종 요약 결과"""
    transcript: str  # STT 결과 (원본 텍스트)
    summary: str  # LLM 요약 결과
    duration_seconds: Optional[float] = None  # 처리 소요 시간
