"""
모델 서버 클라이언트

- 텍스트 임베딩
- Reranking
- 키워드 추출
"""

import asyncio
import logging
import httpx
from typing import List, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def _is_retryable_error(exc: BaseException) -> bool:
    """재시도 가능한 에러인지 판별"""
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 502, 503)
    return False


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 2,
    **kwargs
) -> httpx.Response:
    """
    재시도 로직이 포함된 HTTP 요청

    Args:
        client: httpx.AsyncClient
        method: HTTP 메서드 (GET, POST, DELETE 등)
        url: 요청 URL
        max_retries: 최대 재시도 횟수 (기본 2회, 총 3회 시도)
        **kwargs: httpx 요청 인자 (json, headers, timeout 등)

    Returns:
        httpx.Response
    """
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_error = e
            if not _is_retryable_error(e) or attempt >= max_retries:
                raise

            wait_time = 2 ** attempt  # 1초, 2초, 4초...
            if isinstance(e, httpx.HTTPStatusError):
                logger.warning(f"[Retry] {url} failed (attempt {attempt + 1}/{max_retries + 1}): HTTP {e.response.status_code}, retrying in {wait_time}s...")
            else:
                logger.warning(f"[Retry] {url} failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}, retrying in {wait_time}s...")

            await asyncio.sleep(wait_time)

    raise last_error


class ModelServerClient:
    """
    외부 모델 서버와 통신

    - 텍스트 임베딩
    - Reranking
    - 키워드 추출
    """
    def __init__(self):
        self.base_url = settings.MODEL_SERVER_URL
        self.headers = {"Content-Type": "application/json"}
        self.timeout = settings.MODEL_SERVER_EMBED_TIMEOUT
        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            )
            self._async_client = httpx.AsyncClient(timeout=self.timeout, limits=limits)
        return self._async_client

    async def close(self):
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    # ========================================
    # 임베딩 API
    # ========================================

    async def embed_texts(
        self,
        texts: List[str],
        is_query: bool = False
    ) -> List[List[float]]:
        """
        텍스트 임베딩 생성 (순차 처리)

        POST /embed
        """
        if not texts:
            return []

        url = f"{self.base_url}/embed"
        payload = {"texts": texts, "is_query": is_query}

        try:
            response = await _request_with_retry(
                self.async_client, "POST", url,
                json=payload, headers=self.headers
            )
            return response.json().get("embeddings", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Embed): {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (Embed): {e}")
            raise

    # ========================================
    # Reranking API
    # ========================================

    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        문서 재정렬 (Cross-encoder Reranking)

        POST /rerank

        Returns:
            [{"index": 0, "score": 0.95, "content": "..."}, ...]
        """
        if not documents:
            return []

        url = f"{self.base_url}/rerank"
        payload = {"query": query, "documents": documents, "top_k": top_k}

        try:
            response = await _request_with_retry(
                self.async_client, "POST", url,
                json=payload, headers=self.headers,
                timeout=settings.MODEL_SERVER_QUERY_TIMEOUT
            )
            return response.json().get("results", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Rerank): {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (Rerank): {e}")
            raise

    # ========================================
    # 키워드 추출 API
    # ========================================

    async def extract_keywords_batch(self, texts: List[str]) -> List[str]:
        """
        텍스트 배치에서 키워드 추출
        """
        if not texts:
            return []

        url = f"{self.base_url}/extract_keywords"
        payload = {"texts": texts}

        try:
            response = await _request_with_retry(
                self.async_client, "POST", url,
                json=payload, headers=self.headers,
                timeout=settings.MODEL_SERVER_KEYWORD_TIMEOUT
            )
            return response.json()["keywords"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Keyword Extraction): {e.response.text}")
            raise
        except (httpx.RequestError, KeyError) as e:
            logger.error(f"Model Server Connection Error (Keyword Extraction): {e}")
            raise

    # ========================================
    # 헬스 체크
    # ========================================

    async def health_check(self) -> Dict:
        """모델 서버 상태 확인"""
        url = f"{self.base_url}/health"
        try:
            response = await self.async_client.get(url, timeout=settings.MODEL_SERVER_HEALTH_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# 싱글톤
model_server_client = ModelServerClient()
