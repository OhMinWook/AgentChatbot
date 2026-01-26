"""
모델 서버 클라이언트 (Stateless 버전)

- ColBERT 인코딩만 요청 (저장/검색은 게이트웨이 로컬에서)
- 키워드 추출
"""

import httpx
from typing import List, Dict, Optional
from app.core.config import settings


class ModelServerClient:
    """
    외부 모델 서버(ColBERT 인코딩, 키워드 추출)와 통신
    """
    def __init__(self):
        self.base_url = settings.MODEL_SERVER_URL
        self.headers = {"Content-Type": "application/json"}
        self.timeout = 600.0  # 10분 (대량 인코딩용)
        self._async_client: Optional[httpx.AsyncClient] = None

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    async def close(self):
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    # ========================================
    # ColBERT 인코딩 API
    # ========================================

    async def encode_documents(self, texts: List[str]) -> List[List[List[float]]]:
        """
        문서 텍스트를 ColBERT 임베딩으로 인코딩
        Returns: [batch, tokens, dim] 형태의 임베딩
        """
        if not texts:
            return []

        url = f"{self.base_url}/encode/documents"
        payload = {"texts": texts, "is_query": False}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Encode Documents): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection Error (Encode Documents): {e}")
            raise e

    async def encode_query(self, queries: List[str]) -> List[List[List[float]]]:
        """
        쿼리 텍스트를 ColBERT 임베딩으로 인코딩
        Returns: [batch, tokens, dim] 형태의 임베딩
        """
        if not queries:
            return []

        url = f"{self.base_url}/encode/query"
        payload = {"texts": queries, "is_query": True}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Encode Query): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection Error (Encode Query): {e}")
            raise e

    # ========================================
    # 키워드 추출 API
    # ========================================

    async def extract_keywords_batch(self, texts: List[str]) -> List[str]:
        """
        텍스트 배치에서 키워드 추출
        Returns: 각 청크의 키워드 문자열 목록
        """
        if not texts:
            return []

        url = f"{self.base_url}/extract_keywords"
        payload = {"texts": texts}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers, timeout=120.0)
            response.raise_for_status()
            return response.json()["keywords"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Keyword Extraction): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection Error (Keyword Extraction): {e}")
            raise e

    # ========================================
    # 지식 그래프 추출 API (LightRAG)
    # ========================================

    async def extract_graph_batch(self, texts: List[str]) -> List[Dict]:
        """
        텍스트 배치에서 엔티티 및 관계 추출 (LightRAG용)
        Returns: [{"entities": [...], "relationships": [...]}, ...]
        """
        if not texts:
            return []

        url = f"{self.base_url}/extract_graph"
        payload = {"texts": texts}

        try:
            # 배치 처리는 시간이 오래 걸릴 수 있으므로 넉넉한 타임아웃 설정
            response = await self.async_client.post(url, json=payload, headers=self.headers, timeout=300.0)
            response.raise_for_status()
            return response.json()["results"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Graph Extraction): {e.response.text}")
            return [{"entities": [], "relationships": []}] * len(texts)
        except Exception as e:
            print(f"Model Server Connection Error (Graph Extraction): {e}")
            return [{"entities": [], "relationships": []}] * len(texts)

    # ========================================
    # 헬스 체크
    # ========================================

    async def health_check(self) -> Dict:
        """모델 서버 상태 확인"""
        url = f"{self.base_url}/health"
        try:
            response = await self.async_client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "detail": str(e)}


# 싱글톤
model_server_client = ModelServerClient()
