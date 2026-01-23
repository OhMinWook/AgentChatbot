"""
Pydantic 스키마 정의
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """쿼리 분석 결과 스키마"""
    is_clear: bool = Field(description="질문이 명확한지 여부")
    clarification_message: Optional[str] = Field(default=None, description="명확화 요청 메시지")
    rewritten_questions: List[str] = Field(default_factory=list, description="재작성된 질문 목록")
    reasoning: str = Field(default="", description="분석 근거")


class AgentResponse(BaseModel):
    """개별 에이전트의 답변 스키마"""
    question_index: int = Field(description="처리한 질문의 인덱스")
    answer: str = Field(description="질문에 대한 답변")
    sources: List[dict] = Field(default_factory=list, description="참조한 문서 목록")
