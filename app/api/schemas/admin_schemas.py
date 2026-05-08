"""
관리자 문서 관리 API 스키마
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class ErrorCode(str, Enum):
    """관리자 API 공통 에러코드"""
    SUCCESS = "0000"
    BAD_REQUEST = "4000"
    NOT_FOUND = "4004"
    SERVER_ERROR = "5000"


class AdminBaseResponse(BaseModel):
    """관리자 API 기본 응답"""
    code: str = "0000"
    message: str = "success"


class DocumentItem(BaseModel):
    """문서 목록 아이템"""
    adminId: str
    adminName: str
    index: int
    key: str
    fileName: str
    length: int
    registDate: str
    isUse: bool


class DocumentListResponse(AdminBaseResponse):
    """문서 목록 응답"""
    totalCount: int = 0
    page: int = 1
    size: int = 10
    items: List[DocumentItem] = []


class DocumentCountResponse(AdminBaseResponse):
    """문서 통계 응답"""
    totalCount: int = 0
    useCount: int = 0
    todayCount: int = 0
    unUseCount: int = 0


