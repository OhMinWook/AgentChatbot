from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    # 클라이언트가 보내주는 JSON 키값 그대로 매핑
    user_message: str = Field(..., alias="message", description="사용자의 질문 내용")

    # 파일 첨부 관련 필드
    attachment_name: Optional[str] = Field(None, alias="attachFile_name", description="첨부 파일명")
    file_extension: Optional[str] = Field(None, alias="attachFile_extension", description="파일 확장자 (예: hwp, pdf)")
    deep_research: bool = Field(False, alias="deepResearch")

    class Config:
        populate_by_name = True