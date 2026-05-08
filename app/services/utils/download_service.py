import secrets
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from app.core.config import settings
from app.core.redis_client import sync_redis


@dataclass
class DownloadLinkConfig:
    """다운로드 링크 생성 옵션"""
    expires_in_seconds: int = 3600
    one_time: bool = True
    media_type: str = "application/octet-stream"


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
        self.redis = sync_redis
        # 다운로드 디렉토리 생성
        self.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def create_download_link(
        self,
        file_bytes: bytes,
        filename: str,
        config: Optional[DownloadLinkConfig] = None,
    ) -> Dict[str, Any]:
        """
        다운로드 링크를 생성합니다.

        :param file_bytes: 다운로드할 파일의 바이트 데이터
        :param filename: 다운로드 시 사용할 파일명
        :param config: 링크 옵션 (만료 시간, 1회용 여부, MIME 타입)
        :return: {"token": str, "download_url": str, "expires_at": str}
        """
        cfg = config or DownloadLinkConfig()
        token = secrets.token_urlsafe(32)

        file_path = self.DOWNLOAD_DIR / f"{token}_{filename}"
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        token_data = {
            "file_path": str(file_path),
            "filename": filename,
            "media_type": cfg.media_type,
            "one_time": cfg.one_time,
            "created_at": datetime.now().isoformat()
        }

        redis_key = f"{self.TOKEN_PREFIX}{token}"
        self.redis.setex(redis_key, cfg.expires_in_seconds, json.dumps(token_data))

        expires_at = datetime.now().timestamp() + cfg.expires_in_seconds
        return {
            "token": token,
            "download_url": f"{settings.DOWNLOAD_URL_PREFIX}/{token}",
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "one_time": cfg.one_time
        }

    def create_download_link_from_path(
        self,
        file_path: str,
        filename: str,
        config: Optional[DownloadLinkConfig] = None,
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

        cfg = config or DownloadLinkConfig(one_time=False)
        token = secrets.token_urlsafe(32)

        token_data = {
            "file_path": file_path,
            "filename": filename,
            "media_type": cfg.media_type,
            "one_time": cfg.one_time,
            "is_reference": True,  # 원본 파일 참조 (삭제하지 않음)
            "created_at": datetime.now().isoformat()
        }

        redis_key = f"{self.TOKEN_PREFIX}{token}"
        self.redis.setex(redis_key, cfg.expires_in_seconds, json.dumps(token_data))

        expires_at = datetime.now().timestamp() + cfg.expires_in_seconds
        return {
            "token": token,
            "download_url": f"{settings.DOWNLOAD_URL_PREFIX}/{token}",
            "expires_at": datetime.fromtimestamp(expires_at).isoformat(),
            "one_time": cfg.one_time
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
