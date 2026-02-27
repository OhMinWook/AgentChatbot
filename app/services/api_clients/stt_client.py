"""STT 서버와 통신하는 클라이언트"""
import logging
from pathlib import Path

import httpx

from app.core.config import settings
from app.services.api_clients.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class STTClient(BaseAPIClient):
    def __init__(self):
        super().__init__(
            base_url=settings.MODEL_SERVER_URL,
            timeout=settings.STT_TIMEOUT,
        )

    async def transcribe(self, file_path: str) -> str:
        """
        오디오 파일을 STT 서버로 전송하여 텍스트로 변환 (OpenAI 호환 API)

        :param file_path: 로컬에 저장된 오디오 파일 경로
        :return: 변환된 텍스트
        """
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
        return await self.transcribe_bytes(audio_bytes, Path(file_path).name)

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        오디오 바이트 데이터를 직접 STT 서버로 전송 (OpenAI 호환 API)

        :param audio_bytes: 오디오 파일의 바이트 데이터
        :param filename: 파일명 (확장자 포함)
        :return: 변환된 텍스트
        """
        url = f"{self.base_url}/v1/audio/transcriptions"
        files = {"file": (filename, audio_bytes)}
        data = {"model": "whisper-large-v3-turbo"}

        try:
            response = await self._request_with_retry("POST", url, files=files, data=data)
            return response.json().get("text", "")
        except httpx.HTTPStatusError as e:
            logger.error(f"STT Server Error: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"STT Connection Error: {e}")
            raise


# 싱글톤 인스턴스
stt_client = STTClient()
