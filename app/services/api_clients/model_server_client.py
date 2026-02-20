"""
모델 서버 클라이언트

- 텍스트 임베딩
- Reranking
"""

import logging
import httpx
from typing import List, Dict

from app.core.config import settings
from app.services.api_clients.base_client import BaseAPIClient

logger = logging.getLogger(__name__)


class ModelServerClient(BaseAPIClient):
    """
    외부 모델 서버와 통신

    - 텍스트 임베딩
    - Reranking
    """
    def __init__(self):
        super().__init__(
            base_url=settings.MODEL_SERVER_URL,
            headers={"Content-Type": "application/json"},
            timeout=settings.MODEL_SERVER_EMBED_TIMEOUT,
        )

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
            response = await self._request_with_retry(
                "POST", url,
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
            response = await self._request_with_retry(
                "POST", url,
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


# 싱글톤
model_server_client = ModelServerClient()
