"""
LLM 서버와 통신하는 클라이언트

- vLLM /v1/chat/completions 엔드포인트
- 일반 요청 및 SSE 스트리밍
"""
import logging
import time
import httpx
from typing import AsyncGenerator

from app.core.config import settings
from app.services.api_clients.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class LLMClient(BaseAPIClient):
    def __init__(self):
        super().__init__(
            base_url=settings.MODEL_SERVER_URL,
            headers={"Content-Type": "application/json"},
            timeout=settings.LLM_TIMEOUT,
            limits=httpx.Limits(max_connections=500, max_keepalive_connections=100),
        )
        self._stream_client: httpx.AsyncClient | None = None

    @property
    def stream_client(self) -> httpx.AsyncClient:
        """스트리밍 요청용 클라이언트 (timeout 없음, lazy initialization)"""
        if self._stream_client is None or self._stream_client.is_closed:
            self._stream_client = httpx.AsyncClient(timeout=None)
        return self._stream_client

    async def close(self):
        await super().close()
        if self._stream_client is not None and not self._stream_client.is_closed:
            await self._stream_client.aclose()
            self._stream_client = None

    @staticmethod
    def extract_content(response: dict) -> str:
        """LLM 응답에서 content를 안전하게 추출"""
        try:
            content = response["choices"][0]["message"]["content"]
            if content is None:
                logger.warning(f"LLM 응답 content가 None: {response}")
                return ""
            return content
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"LLM 응답 파싱 실패: {e}, response={response}")
            raise ValueError(f"LLM 응답 형식 오류: {e}")

    async def chat_completions(self, payload: dict, max_retries: int = 1) -> dict:
        """
        일반 대화 요청 (Non-streaming)

        :param payload: LLM 요청 payload
        :param max_retries: 실패 시 재시도 횟수 (기본 1회)
        :return: 파싱된 JSON dict
        """
        url = f"{self.base_url}/v1/chat/completions"
        logger.info(f"[LLM Client] 요청 시작 → {url}")
        start_time = time.time()

        response = await self._request_with_retry(
            "POST", url,
            max_retries=max_retries,
            json=payload, headers=self.headers,
        )

        logger.info(f"[LLM Client] 응답 완료 ({time.time() - start_time:.2f}s)")
        return response.json()

    async def chat_completions_stream(self, payload: dict, max_retries: int = 1) -> AsyncGenerator[bytes, None]:
        """
        vLLM 스트리밍 요청 (SSE)

        :param payload: LLM 요청 payload
        :param max_retries: 연결 실패 시 재시도 횟수 (기본 1회)
        :return: 바이트 스트림 제너레이터
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload["stream"] = True

        last_error = None
        r = None

        for attempt in range(max_retries + 1):
            try:
                req = self.stream_client.build_request("POST", url, json=payload, headers=self.headers)
                r = await self.stream_client.send(req, stream=True)
                r.raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"LLM Stream Server Error (attempt {attempt + 1}/{max_retries + 1}): {e.response.text}")
                if attempt < max_retries:
                    logger.info("Retrying LLM stream request...")
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"LLM Stream Connection Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    logger.info("Retrying LLM stream request...")

        if r is None:
            logger.error(f"LLM stream request failed after {max_retries + 1} attempts")
            raise last_error or RuntimeError(f"LLM stream request failed after {max_retries + 1} attempts")

        async def gen():
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            except Exception as stream_err:
                logger.warning(f"Streaming interrupted: {stream_err}")
                raise stream_err
            finally:
                await r.aclose()

        return gen()


# 싱글톤처럼 쓰기 위해 인스턴스 생성
llm_client = LLMClient()
