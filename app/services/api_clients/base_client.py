"""
API 클라이언트 베이스 클래스

- lazy-init httpx.AsyncClient
- exponential backoff 재시도
"""

import asyncio
import logging
from typing import Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseAPIClient:
    """HTTP API 클라이언트 공통 베이스"""

    def __init__(
        self,
        base_url: str,
        headers: Optional[dict] = None,
        timeout: float = 30.0,
        limits: Optional[httpx.Limits] = None,
    ):
        self.base_url = base_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.limits = limits or httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30.0,
        )
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-init + is_closed 자동 재생성"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=self.limits,
            )
        return self._client

    async def close(self):
        """클라이언트 정리"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # --------------------------------------------------
    # 재시도 로직
    # --------------------------------------------------

    @staticmethod
    def _is_retryable_error(exc: BaseException) -> bool:
        """재시도 가능한 에러인지 판별"""
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in (429, 502, 503)
        return False

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = None,
        **kwargs,
    ) -> httpx.Response:
        """
        재시도 로직이 포함된 HTTP 요청

        Args:
            method: HTTP 메서드 (GET, POST, DELETE 등)
            url: 요청 URL
            max_retries: 최대 재시도 횟수 (기본 settings.MAX_API_RETRIES)
            **kwargs: httpx 요청 인자 (json, headers, timeout 등)

        Returns:
            httpx.Response
        """
        if max_retries is None:
            max_retries = settings.MAX_API_RETRIES
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response

            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
                last_error = e
                if not self._is_retryable_error(e) or attempt >= max_retries:
                    raise

                wait_time = 2 ** attempt  # 1초, 2초, 4초...
                if isinstance(e, httpx.HTTPStatusError):
                    logger.warning(f"[Retry] {url} failed (attempt {attempt + 1}/{max_retries + 1}): HTTP {e.response.status_code}, retrying in {wait_time}s...")
                else:
                    logger.warning(f"[Retry] {url} failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}, retrying in {wait_time}s...")

                await asyncio.sleep(wait_time)

        raise last_error
