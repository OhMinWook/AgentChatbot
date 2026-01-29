"""파일 업로드/저장 관련 공통 유틸리티"""
import os
from typing import Tuple
from fastapi import UploadFile

from app.core.config import settings
from app.services.utils.path_validator import safe_join


async def save_upload_file(
    file: UploadFile,
    *path_parts: str,
    default_filename: str = "file"
) -> Tuple[str, bytes]:
    """업로드된 파일을 안전하게 저장

    Args:
        file: FastAPI UploadFile 객체
        *path_parts: UPLOAD_DIR 하위 경로 (예: "call_summary", invoke_id)
        default_filename: 파일명이 없을 경우 사용할 기본 이름

    Returns:
        (saved_path, content) 튜플

    Raises:
        ValueError: Path traversal 시도 시
    """
    # 저장 디렉토리 생성 (Path Traversal 방지)
    upload_dir = safe_join(settings.UPLOAD_DIR, *path_parts)
    os.makedirs(upload_dir, exist_ok=True)

    # 파일명 결정
    filename = file.filename or default_filename
    saved_path = safe_join(upload_dir, filename)

    # 파일 저장
    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    return saved_path, content
