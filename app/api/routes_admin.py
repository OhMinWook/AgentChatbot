"""
관리자 문서 관리 API 라우터
"""

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.services.utils.profanity_filter import reload as reload_profanity

from app.api.schemas.admin_schemas import (
    AdminBaseResponse,
    DocumentItem,
    DocumentListResponse,
    DocumentCountResponse,
    ErrorCode,
)
from app.services.admin.admin_document_service import (
    admin_document_service,
    AdminDocumentInput,
    DocumentSearchQuery,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


def _server_error(ResponseClass, e: Exception, label: str):
    """예외를 로그에 기록하고 code=5000 응답 반환"""
    logger.exception(f"[Admin] {label}: {e}")
    return ResponseClass(code=ErrorCode.SERVER_ERROR, message=f"서버 오류: {str(e)}")


def _build_document_items(docs: list) -> list:
    return [
        DocumentItem(
            adminId=doc.get("admin_id", ""),
            adminName=doc.get("admin_name", ""),
            index=doc.get("index", 0),
            key=doc.get("key", ""),
            fileName=doc.get("file_name", ""),
            length=doc.get("file_size", 0),
            registDate=doc.get("regist_date", ""),
            isUse=doc.get("is_use", True),
        )
        for doc in docs
    ]


@router.post("/add_documents", response_model=AdminBaseResponse)
async def add_document(
    key: str = Form(..., description="문서 고유 키"),
    adminId: str = Form(..., description="관리자 ID"),
    adminName: str = Form(..., description="관리자 이름"),
    file: UploadFile = File(..., description="업로드할 파일"),
):
    """
    문서 등록

    - 파일을 업로드하고 청킹/임베딩 후 Qdrant에 저장
    - 동일한 key가 이미 존재하면 실패
    """
    # 임시 파일에 저장
    temp_file = None
    try:
        # 파일 확장자 추출
        _, ext = os.path.splitext(file.filename or "")

        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        content = await file.read()
        temp_file.write(content)
        temp_file.close()

        file_size = len(content)
        file_name = file.filename or "unknown"

        success = await admin_document_service.add_document(
            AdminDocumentInput(
                key=key,
                admin_id=adminId,
                admin_name=adminName,
                file_path=temp_file.name,
                file_name=file_name,
                file_size=file_size,
            )
        )

        if not success:
            return AdminBaseResponse(
                code=ErrorCode.BAD_REQUEST,
                message="문서 등록 실패 (중복 키 또는 파싱 실패)",
            )

        return AdminBaseResponse(
            code=ErrorCode.SUCCESS,
            message="문서가 성공적으로 등록되었습니다.",
        )

    except Exception as e:
        return _server_error(AdminBaseResponse, e, "Add document failed")
    finally:
        # 임시 파일 삭제
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@router.delete("/del_documents/{key}", response_model=AdminBaseResponse)
async def delete_document(key: str):
    """
    문서 삭제

    - key에 해당하는 모든 청크와 파일을 삭제
    """
    try:
        success = await admin_document_service.delete_document(key)

        if not success:
            return AdminBaseResponse(
                code=ErrorCode.NOT_FOUND,
                message="문서를 찾을 수 없거나 삭제에 실패했습니다.",
            )

        return AdminBaseResponse(
            code=ErrorCode.SUCCESS,
            message="문서가 성공적으로 삭제되었습니다.",
        )

    except Exception as e:
        return _server_error(AdminBaseResponse, e, "Delete document failed")


@router.get("/get_documents", response_model=DocumentListResponse)
async def get_documents(
    page: int = 1,
    size: int = 10,
    orderType: str = "registDate",
    order: str = "desc",
):
    """
    문서 목록 조회 (페이지네이션)

    - orderType: fileName (문서명) | registDate (등록일)
    - order: desc (내림차순) | asc (오름차순)
    """
    try:
        docs, total_count = await admin_document_service.get_document_list(
            page=page,
            size=size,
            order_type=orderType,
            order=order,
        )

        items = _build_document_items(docs)

        return DocumentListResponse(
            code=ErrorCode.SUCCESS,
            message="success",
            totalCount=total_count,
            page=page,
            size=size,
            items=items,
        )

    except Exception as e:
        return _server_error(DocumentListResponse, e, "Get documents failed")


@router.get("/documents/search", response_model=DocumentListResponse)
async def search_documents(
    searchType: str = "fileName",
    searchTerm: str = "",
    page: int = 1,
    size: int = 10,
    orderType: str = "registDate",
    order: str = "desc",
):
    """
    문서 검색

    - searchType: fileName, adminId, adminName
    - searchTerm: 검색어 (부분 일치)
    - orderType: fileName (문서명) | registDate (등록일)
    - order: desc (내림차순) | asc (오름차순)
    """
    try:
        docs, total_count = await admin_document_service.search_documents(
            DocumentSearchQuery(
                search_type=searchType,
                search_term=searchTerm,
                page=page,
                size=size,
                order_type=orderType,
                order=order,
            )
        )

        items = _build_document_items(docs)

        return DocumentListResponse(
            code=ErrorCode.SUCCESS,
            message="success",
            totalCount=total_count,
            page=page,
            size=size,
            items=items,
        )

    except Exception as e:
        return _server_error(DocumentListResponse, e, "Search documents failed")


@router.patch("/documents/{key}/toggle", response_model=AdminBaseResponse)
async def toggle_document_usage(key: str):
    """
    사용여부 토글

    - is_use 값을 반전시킴
    - is_use=False인 문서는 일반 사용자 검색에서 제외됨
    """
    try:
        new_value = await admin_document_service.toggle_usage(key)

        if new_value is None:
            return AdminBaseResponse(
                code=ErrorCode.NOT_FOUND,
                message="문서를 찾을 수 없습니다.",
            )

        status_text = "사용" if new_value else "미사용"
        return AdminBaseResponse(
            code=ErrorCode.SUCCESS,
            message=f"문서 상태가 '{status_text}'으로 변경되었습니다.",
        )

    except Exception as e:
        return _server_error(AdminBaseResponse, e, "Toggle usage failed")


@router.get("/documents/count", response_model=DocumentCountResponse)
async def get_document_count():
    """
    문서 통계 조회

    - totalCount: 전체 문서 수
    - useCount: 사용 중인 문서 수
    - unUseCount: 미사용 문서 수
    - todayCount: 오늘 등록된 문서 수
    """
    try:
        stats = await admin_document_service.get_statistics()

        return DocumentCountResponse(
            code=ErrorCode.SUCCESS,
            message="success",
            totalCount=stats.get("totalCount", 0),
            useCount=stats.get("useCount", 0),
            todayCount=stats.get("todayCount", 0),
            unUseCount=stats.get("unUseCount", 0),
        )

    except Exception as e:
        return _server_error(DocumentCountResponse, e, "Get count failed")


@router.post("/profanity/reload", summary="금칙어 목록 재로드")
async def reload_profanity_filter():
    """
    dataset.csv의 금칙어 목록을 서버 재시작 없이 즉시 갱신합니다.
    """
    try:
        reload_profanity()
        return {"code": "0000", "message": "금칙어 목록이 갱신되었습니다."}
    except Exception as e:
        logger.exception(f"[Admin] Profanity reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")
