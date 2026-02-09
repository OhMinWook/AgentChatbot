import secrets
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

import redis

from app.core.config import settings


class DownloadService:
    """
    1회용 또는 기간제 다운로드 링크를 생성하고 관리하는 서비스.
    Redis를 사용하여 토큰 -> 파일 정보 매핑을 저장합니다.
    """

    # 다운로드 파일 저장 경로
    DOWNLOAD_DIR = Path(settings.UPLOAD_DIR) / "downloads"

    # Redis 키 접두사
    TOKEN_PREFIX = "download_token:"

    def __init__(self):
        self.redis = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        # 다운로드 디렉토리 생성
        self.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def create_download_link(
        self,
        file_bytes: bytes,
        filename: str,
        expires_in_seconds: int = 3600,
        one_time: bool = True,
        media_type: str = "application/octet-stream"
    ) -> Dict[str, Any]:
        """
        다운로드 링크를 생성합니다.

        :param file_bytes: 다운로드할 파일의 바이트 데이터
        :param filename: 다운로드 시 사용할 파일명
        :param expires_in_seconds: 링크 만료 시간 (초 단위, 기본 1시간)
        :param one_time: True면 1회 다운로드 후 링크 무효화
        :param media_type: 파일의 MIME 타입
        :return: {"token": str, "download_url": str, "expires_at": str}
        """
        # 고유 토큰 생성 (URL-safe)
        token = secrets.token_urlsafe(32)

        # 파일을 디스크에 저장
        file_path = self.DOWNLOAD_DIR / f"{token}_{filename}"
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # 토큰 정보를 Redis에 저장
        token_data = {
            "file_path": str(file_path),
            "filename": filename,
            "media_type": media_type,
            "one_time": one_time,
            "created_at": datetime.now().isoformat()
        }

        redis_key = f"{self.TOKEN_PREFIX}{token}"
        self.redis.setex(
            redis_key,
            expires_in_seconds,
            json.dumps(token_data)
        )

        # 만료 시간 계산
        expires_at = datetime.now().timestamp() + expires_in_seconds

        return {
            "token": token,
            "download_url": f"/documents/download/{token}",
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "one_time": one_time
        }

    def create_download_link_from_path(
        self,
        file_path: str,
        filename: str,
        expires_in_seconds: int = 3600,
        one_time: bool = False,
        media_type: str = "application/octet-stream"
    ) -> Optional[Dict[str, Any]]:
        """
        기존 파일에 대한 다운로드 링크 생성 (파일 복사 없이)

        :param file_path: 다운로드할 파일의 실제 경로
        :param filename: 다운로드 시 사용할 파일명
        :param expires_in_seconds: 링크 만료 시간 (초 단위, 기본 1시간)
        :param one_time: True면 1회 다운로드 후 링크 무효화
        :param media_type: 파일의 MIME 타입
        :return: {"token": str, "download_url": str, "expires_at": str} 또는 None (파일 없음)
        """
        # 파일 존재 확인
        if not os.path.exists(file_path):
            return None

        # 고유 토큰 생성 (URL-safe)
        token = secrets.token_urlsafe(32)

        # 토큰 정보를 Redis에 저장 (원본 파일 경로 사용)
        token_data = {
            "file_path": file_path,
            "filename": filename,
            "media_type": media_type,
            "one_time": one_time,
            "is_reference": True,  # 원본 파일 참조 (삭제하지 않음)
            "created_at": datetime.now().isoformat()
        }

        redis_key = f"{self.TOKEN_PREFIX}{token}"
        self.redis.setex(
            redis_key,
            expires_in_seconds,
            json.dumps(token_data)
        )

        # 만료 시간 계산
        expires_at = datetime.now().timestamp() + expires_in_seconds

        return {
            "token": token,
            "download_url": f"/documents/download/{token}",
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "one_time": one_time
        }

    def get_file_info(self, token: str) -> Optional[Dict[str, Any]]:
        """
        토큰으로 파일 정보를 조회합니다.
        1회용 링크인 경우 조회 후 토큰을 삭제합니다.

        :param token: 다운로드 토큰
        :return: 파일 정보 딕셔너리 또는 None (유효하지 않은 토큰)
        """
        redis_key = f"{self.TOKEN_PREFIX}{token}"
        data = self.redis.get(redis_key)

        if not data:
            return None

        token_data = json.loads(data)

        # 파일 존재 확인
        if not os.path.exists(token_data["file_path"]):
            self.redis.delete(redis_key)
            return None

        # 1회용 링크면 토큰 삭제 (파일은 다운로드 후 삭제)
        if token_data.get("one_time", True):
            self.redis.delete(redis_key)

        return token_data

    def delete_file(self, file_path: str):
        """다운로드 완료 후 파일 삭제"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass



# 싱글톤 인스턴스
download_service = DownloadService()
