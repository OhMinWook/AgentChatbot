"""
모델 서버 클라이언트

- ColBERT 인덱싱/검색 (원격 모델 서버에서 처리)
- 키워드 추출
- 지식 그래프 추출

인덱스와 메타데이터는 모델 서버 측에서 관리합니다.
"""

import logging
import time
import httpx
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """ColBERT 검색 결과"""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]


class ModelServerClient:
    """
    외부 모델 서버와 통신

    - ColBERT 인덱싱/검색 (원격)
    - 키워드 추출
    - 지식 그래프 추출
    """
    def __init__(self):
        self.base_url = settings.MODEL_SERVER_URL
        self.headers = {"Content-Type": "application/json"}
        self.timeout = settings.MODEL_SERVER_EMBED_TIMEOUT  # 10분 (대량 문서 임베딩용)
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
    # ColBERT 인덱싱/검색 API (원격)
    # ========================================

    async def colbert_index_documents(
        self,
        invoke_id: str,
        documents: List[Dict]
    ) -> Dict:
        """
        문서를 모델 서버의 ColBERT 인덱스에 저장

        POST /colbert/index
        Request: {"invoke_id": "...", "documents": [...]}
        Response: {"indexed_count": N, "total_tokens": N}
        """
        if not documents:
            return {"indexed_count": 0, "total_tokens": 0}

        url = f"{self.base_url}/colbert/index"
        payload = {"invoke_id": invoke_id, "documents": documents}

        logger.info(f"[ModelServer] colbert_index_documents: {len(documents)}개 문서, invoke_id={invoke_id}")

        try:
            start = time.time()
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            elapsed = time.time() - start

            result = response.json()
            logger.info(f"[ModelServer] colbert_index_documents 완료: {result.get('indexed_count', 0)}개 인덱싱 ({elapsed:.2f}s)")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (ColBERT Index): {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (ColBERT Index): {e}")
            raise e

    async def colbert_search(
        self,
        invoke_id: str,
        query: str,
        top_k: int = 6
    ) -> List[SearchResult]:
        """
        모델 서버에서 ColBERT 검색 수행

        POST /colbert/search
        Request: {"invoke_id": "...", "query": "...", "top_k": N}
        Response: {"results": [{"doc_id": "...", "score": N, "content": "...", "metadata": {...}}]}
        """
        url = f"{self.base_url}/colbert/search"
        payload = {"invoke_id": invoke_id, "query": query, "top_k": top_k}

        try:
            response = await self.async_client.post(
                url, json=payload, headers=self.headers,
                timeout=settings.MODEL_SERVER_QUERY_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    doc_id=item.get("doc_id", ""),
                    score=float(item.get("score", 0.0)),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {})
                ))
            return results
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (ColBERT Search): {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (ColBERT Search): {e}")
            raise e

    async def colbert_search_batch(
        self,
        invoke_id: str,
        queries: List[str],
        top_k: int = 6
    ) -> List[List[SearchResult]]:
        """
        모델 서버에서 ColBERT 배치 검색 수행

        POST /colbert/search/batch
        Request: {"invoke_id": "...", "queries": [...], "top_k": N}
        Response: {"results": [[...], [...]]}
        """
        if not queries:
            return []

        url = f"{self.base_url}/colbert/search/batch"
        payload = {"invoke_id": invoke_id, "queries": queries, "top_k": top_k}

        try:
            response = await self.async_client.post(
                url, json=payload, headers=self.headers,
                timeout=settings.MODEL_SERVER_QUERY_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            all_results = []
            for query_results in data.get("results", []):
                results = []
                for item in query_results:
                    results.append(SearchResult(
                        doc_id=item.get("doc_id", ""),
                        score=float(item.get("score", 0.0)),
                        content=item.get("content", ""),
                        metadata=item.get("metadata", {})
                    ))
                all_results.append(results)
            return all_results
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (ColBERT Batch Search): {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (ColBERT Batch Search): {e}")
            raise e

    async def colbert_store_document_metadata(
        self,
        invoke_id: str,
        file_name: str,
        total_pages: int,
        total_chunks: int
    ) -> None:
        """
        문서 메타데이터를 모델 서버에 저장

        POST /colbert/index/{invoke_id}/metadata
        """
        url = f"{self.base_url}/colbert/index/{invoke_id}/metadata"
        payload = {
            "file_name": file_name,
            "total_pages": total_pages,
            "total_chunks": total_chunks
        }

        try:
            response = await self.async_client.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            logger.info(f"[ModelServer] 문서 메타데이터 저장 완료: {file_name}")
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Store Metadata): {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (Store Metadata): {e}")
            raise e

    async def colbert_get_document_metadata(
        self,
        invoke_id: str,
        file_name: str
    ) -> Optional[Dict]:
        """
        문서 메타데이터 조회

        GET /colbert/index/{invoke_id}/metadata/{file_name}
        """
        url = f"{self.base_url}/colbert/index/{invoke_id}/metadata/{file_name}"

        try:
            response = await self.async_client.get(url, headers=self.headers)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            logger.error(f"Model Server Error (Get Metadata): {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (Get Metadata): {e}")
            raise e

    async def colbert_list_documents(
        self,
        invoke_id: str
    ) -> List[Dict]:
        """
        인덱싱된 문서 목록 조회

        GET /colbert/index/{invoke_id}/documents
        """
        url = f"{self.base_url}/colbert/index/{invoke_id}/documents"

        try:
            response = await self.async_client.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json().get("documents", [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (List Documents): {e.response.text}")
            return []
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (List Documents): {e}")
            return []

    async def colbert_delete_index(
        self,
        invoke_id: str
    ) -> Dict:
        """
        인덱스 삭제

        DELETE /colbert/index/{invoke_id}
        """
        url = f"{self.base_url}/colbert/index/{invoke_id}"

        try:
            response = await self.async_client.delete(url, headers=self.headers)
            response.raise_for_status()
            result = response.json()
            logger.info(f"[ModelServer] 인덱스 삭제 완료: {invoke_id}")
            return result
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Delete Index): {e.response.text}")
            raise e
        except httpx.RequestError as e:
            logger.error(f"Model Server Connection Error (Delete Index): {e}")
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
            response = await self.async_client.post(url, json=payload, headers=self.headers, timeout=settings.MODEL_SERVER_KEYWORD_TIMEOUT)
            response.raise_for_status()
            return response.json()["keywords"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Keyword Extraction): {e.response.text}")
            raise e
        except (httpx.RequestError, KeyError) as e:
            logger.error(f"Model Server Connection Error (Keyword Extraction): {e}")
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
            response = await self.async_client.post(url, json=payload, headers=self.headers, timeout=settings.MODEL_SERVER_GRAPH_TIMEOUT)
            response.raise_for_status()
            return response.json()["results"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Model Server Error (Graph Extraction): {e.response.text}")
            return [{"entities": [], "relationships": []}] * len(texts)
        except Exception as e:
            logger.error(f"Model Server Connection Error (Graph Extraction): {e}")
            return [{"entities": [], "relationships": []}] * len(texts)

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
