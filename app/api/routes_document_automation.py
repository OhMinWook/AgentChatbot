from fastapi import APIRouter, HTTPException, Body, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from typing import Dict, Any, Optional
import json
import os

from app.services.documents.document_automater_service import document_automater_service
from app.services.utils.download_service import download_service

router = APIRouter()

@router.post("/documents/generate-hwpx", summary="HWPX 문서 자동 생성")
async def generate_hwpx_document_api(
        template_name: str = Form(..., description="사용할 HWPX 템플릿 파일명 (예: 'template.hwpx' 또는 'template2.hwpx')"),
        context_data_str: str = Form(..., description="문서에 채워 넣을 JSON 데이터 (문자열 형태)", alias="context_data"),
        expires_in: int = Form(3600, description="다운로드 링크 만료 시간 (초 단위, 기본 1시간)"),
        one_time: bool = Form(True, description="1회용 링크 여부 (기본 True)")
):
    """
    제공된 HWPX 템플릿과 JSON 데이터를 사용하여 새로운 HWPX 문서를 생성하고,
    1회용/기간제 다운로드 링크를 반환합니다.

    - **template_name**: 서버의 `app/templates/documents/` 폴더에 있는 HWPX 템플릿 파일명 (예: 'template.hwpx')
    - **context_data**: 문서의 플레이스홀더를 채울 JSON 형식의 데이터.
    - **expires_in**: 다운로드 링크 만료 시간 (초 단위, 기본 3600초 = 1시간)
    - **one_time**: True면 1회 다운로드 후 링크 무효화

    **반환값**: {"download_url": str, "expires_at": str, "one_time": bool}
    """
    try:
        # 문자열로 받은 context_data를 JSON(딕셔너리)으로 파싱
        try:
            context_data = json.loads(context_data_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="'context_data' 필드가 유효한 JSON 문자열이 아닙니다.")

        # 서비스 함수 호출
        generated_hwpx_bytes = document_automater_service.generate_hwpx_document(
            template_name=template_name,
            context_data=context_data
        )

        # HWPX 파일의 MIME 타입
        media_type = "application/haansofthwpml"
        filename = f"generated_{template_name}"

        # 다운로드 링크 생성
        link_info = download_service.create_download_link(
            file_bytes=generated_hwpx_bytes,
            filename=filename,
            expires_in_seconds=expires_in,
            one_time=one_time,
            media_type=media_type
        )

        return JSONResponse(content={
            "success": True,
            "download_url": link_info["download_url"],
            "expires_at": link_info["expires_at"],
            "one_time": link_info["one_time"],
            "filename": filename
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"HWPX 문서 생성 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@router.get("/documents/download/{token}", summary="토큰 기반 파일 다운로드")
async def download_document(token: str):
    """
    생성된 토큰을 사용하여 문서를 다운로드합니다.

    - **token**: 문서 생성 시 반환된 다운로드 토큰
    - 1회용 링크인 경우 다운로드 후 링크가 무효화됩니다.
    - 만료된 링크는 사용할 수 없습니다.
    """
    # 토큰으로 파일 정보 조회
    file_info = download_service.get_file_info(token)

    if not file_info:
        raise HTTPException(
            status_code=404,
            detail="다운로드 링크가 만료되었거나 유효하지 않습니다."
        )

    file_path = file_info["file_path"]
    filename = file_info["filename"]
    media_type = file_info["media_type"]
    is_one_time = file_info.get("one_time", True)

    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    # 파일 읽기
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # 1회용 링크면 파일 삭제
    if is_one_time:
        download_service.delete_file(file_path)

    return StreamingResponse(
        content=iter([file_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
