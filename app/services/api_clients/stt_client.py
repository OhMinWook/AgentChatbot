"""STT 서버와 통신하는 클라이언트"""
import asyncio
import logging
from pathlib import Path

import httpx

from app.core.config import settings
from app.services.api_clients.base_client import BaseAPIClient
from app.services.utils.profanity_filter import filter_profanity

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
        def _read():
            with open(file_path, "rb") as f:
                return f.read()

        audio_bytes = await asyncio.to_thread(_read)
        return await self.transcribe_bytes(audio_bytes, Path(file_path).name)

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        오디오 바이트 데이터를 직접 STT 서버로 전송 (OpenAI 호환 API)

        :param audio_bytes: 오디오 파일의 바이트 데이터
        :param filename: 파일명 (확장자 포함)
        :return: 변환된 텍스트
        """
        _MIME_TYPES = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "mp4": "audio/mp4",
            "m4a": "audio/mp4",
            "ogg": "audio/ogg",
            "flac": "audio/flac",
            "webm": "audio/webm",
        }
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "wav"
        mime_type = _MIME_TYPES.get(ext, "audio/wav")

        url = f"{self.base_url}/v1/audio/transcriptions"
        files = {"file": (filename, audio_bytes, mime_type)}
        data = {"model": "whisper-large-v3-turbo"}

        try:
            response = await self._request_with_retry("POST", url, files=files, data=data)
            try:
                return filter_profanity(response.json().get("text", ""))
            except Exception:
                logger.error(f"STT 응답 JSON 파싱 실패: {response.text[:200]}")
                raise RuntimeError(f"STT 서버 응답을 파싱할 수 없습니다: {response.text[:200]}")
        except httpx.HTTPStatusError as e:
            logger.error(f"STT Server Error: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"STT Connection Error: {e}")
            raise


# 싱글톤 인스턴스
stt_client = STTClient()
