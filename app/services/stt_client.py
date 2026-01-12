"""STT 서버와 통신하는 클라이언트"""
import httpx
from typing import Optional
from app.core.config import settings


class STTClient:
    def __init__(self):
        self.base_url = settings.STT_BASE_URL
        self.timeout = 300.0  # STT는 오래 걸릴 수 있으므로 5분

    async def transcribe(self, file_path: str) -> str:
        """
        오디오 파일을 STT 서버로 전송하여 텍스트로 변환

        :param file_path: 로컬에 저장된 오디오 파일 경로
        :return: 변환된 텍스트
        """
        url = f"{self.base_url}/transcribe"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                # 파일을 multipart/form-data로 전송
                with open(file_path, "rb") as audio_file:
                    files = {"file": audio_file}
                    response = await client.post(url, files=files)
                    response.raise_for_status()

                result = response.json()
                # STT 서버 응답: {"transcription": "...", "detected_language": "...", ...}
                return result.get("transcription") or result.get("text") or result.get("transcript", "")

            except httpx.HTTPStatusError as e:
                print(f"STT Server Error: {e.response.text}")
                raise e
            except httpx.RequestError as e:
                print(f"STT Connection Error: {e}")
                raise e

    async def transcribe_bytes(self, audio_bytes: bytes, filename: str = "audio.wav") -> str:
        """
        오디오 바이트 데이터를 직접 STT 서버로 전송

        :param audio_bytes: 오디오 파일의 바이트 데이터
        :param filename: 파일명 (확장자 포함)
        :return: 변환된 텍스트
        """
        url = f"{self.base_url}/transcribe"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                files = {"file": (filename, audio_bytes)}
                response = await client.post(url, files=files)
                response.raise_for_status()

                result = response.json()
                return result.get("transcription") or result.get("text") or result.get("transcript", "")

            except httpx.HTTPStatusError as e:
                print(f"STT Server Error: {e.response.text}")
                raise e
            except httpx.RequestError as e:
                print(f"STT Connection Error: {e}")
                raise e


# 싱글톤 인스턴스
stt_client = STTClient()