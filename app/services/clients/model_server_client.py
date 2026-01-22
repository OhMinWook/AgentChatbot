import httpx
from typing import List, Dict, Tuple
from app.core.config import settings

class ModelServerClient:
    """
    외부 모델 서버(Embedding, Reranker 등)와 통신하기 위한 클라이언트
    httpx 클라이언트를 재사용하여 연결 효율성 향상
    """
    def __init__(self):
        self.base_url = settings.MODEL_SERVER_URL
        self.headers = {"Content-Type": "application/json"}
        self.timeout = 60.0

        # 클라이언트는 lazy initialization (첫 사용 시 생성)
        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    @property
    def async_client(self) -> httpx.AsyncClient:
        """비동기 클라이언트 (lazy initialization)"""
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client

    @property
    def sync_client(self) -> httpx.Client:
        """동기 클라이언트 (lazy initialization)"""
        if self._sync_client is None or self._sync_client.is_closed:
            self._sync_client = httpx.Client(timeout=self.timeout)
        return self._sync_client

    async def close(self):
        """비동기 클라이언트 정리"""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    def close_sync(self):
        """동기 클라이언트 정리"""
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()
            self._sync_client = None

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        [비동기] 텍스트 목록을 받아 임베딩 벡터 목록을 반환합니다.
        """
        url = f"{self.base_url}/embeddings"
        payload = {"texts": texts}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Embeddings): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Embeddings): {e}")
            raise e

    def get_embeddings_sync(self, texts: List[str]) -> List[List[float]]:
        """
        [동기] 텍스트 목록을 받아 임베딩 벡터 목록을 반환합니다.
        """
        url = f"{self.base_url}/embeddings"
        payload = {"texts": texts}

        try:
            response = self.sync_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Sync Embeddings): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Sync Embeddings): {e}")
            raise e

    async def rerank(self, query: str, documents: List[str]) -> List[dict]:
        """
        [비동기] 질의와 문서 목록을 받아 재랭킹된 결과를 (문서와 점수가 포함된 dict) 목록으로 반환합니다.
        """
        url = f"{self.base_url}/rerank"
        payload = {"query": query, "documents": documents}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["results"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Rerank): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Rerank): {e}")
            raise e

    async def get_sparse_embeddings(self, texts: List[str]) -> List[Dict[str, List]]:
        """
        [비동기] 텍스트 목록을 받아 sparse embedding (indices, values) 목록을 반환합니다.
        """
        url = f"{self.base_url}/sparse_embeddings"
        payload = {"texts": texts}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["sparse_embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Sparse Embeddings): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Sparse Embeddings): {e}")
            raise e

    def get_sparse_embeddings_sync(self, texts: List[str]) -> List[Dict[str, List]]:
        """
        [동기] 텍스트 목록을 받아 sparse embedding (indices, values) 목록을 반환합니다.
        """
        url = f"{self.base_url}/sparse_embeddings"
        payload = {"texts": texts}

        try:
            response = self.sync_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            return response.json()["sparse_embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Sync Sparse Embeddings): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Sync Sparse Embeddings): {e}")
            raise e

    async def get_hybrid_embeddings(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, List]]]:
        """
        [비동기] Dense + Sparse embedding을 동시에 반환합니다.
        Returns: (dense_embeddings, sparse_embeddings)
        """
        url = f"{self.base_url}/hybrid_embeddings"
        payload = {"texts": texts}

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data["dense_embeddings"], data["sparse_embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Hybrid Embeddings): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Hybrid Embeddings): {e}")
            raise e

    def get_hybrid_embeddings_sync(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, List]]]:
        """
        [동기] Dense + Sparse embedding을 동시에 반환합니다.
        Returns: (dense_embeddings, sparse_embeddings)
        """
        url = f"{self.base_url}/hybrid_embeddings"
        payload = {"texts": texts}

        try:
            response = self.sync_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data["dense_embeddings"], data["sparse_embeddings"]
        except httpx.HTTPStatusError as e:
            print(f"Model Server Error (Sync Hybrid Embeddings): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            print(f"Model Server Connection or a malformed response Error (Sync Hybrid Embeddings): {e}")
            raise e

# 싱글톤처럼 사용하기 위해 인스턴스 생성
model_server_client = ModelServerClient()
