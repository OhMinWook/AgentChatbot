"""
관리자 문서 관리 전용 서비스

Qdrant를 직접 사용하여 관리자 문서의 CRUD 작업을 처리합니다.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.services.api_clients.model_server_client import model_server_client
from app.services.rag.chunker import IncrementalChunker
from app.services.rag.extractors import file_text_extractor
from app.services.rag.sparse_encoder import sparse_encoder
from app.services.rag.text_utils import add_source_prefix

logger = logging.getLogger(__name__)


@dataclass
class AdminDocumentInput:
    key: str
    admin_id: str
    admin_name: str
    file_path: str
    file_name: str
    file_size: int


@dataclass
class DocumentSearchQuery:
    search_type: str
    search_term: str
    page: int = 1
    size: int = 10
    order_type: str = "registDate"
    order: str = "desc"


class AdminDocumentService:
    """관리자 문서 관리 전용 서비스"""

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            self._client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
            )
        return self._client

    async def ensure_collection(self) -> None:
        """컬렉션 및 인덱스 확인/생성"""
        from app.services.rag.qdrant_service import qdrant_service
        await qdrant_service.ensure_collection()

        # 관리자 필드 인덱스 생성 (이미 있으면 스킵)
        try:
            await self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="key",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse:
            pass  # 이미 존재

        try:
            await self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="is_use",
                field_schema=models.PayloadSchemaType.BOOL,
            )
        except UnexpectedResponse:
            pass

        try:
            await self.client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name="regist_date",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse:
            pass

    async def add_document(self, inp: AdminDocumentInput) -> bool:
        """
        문서 등록 (청킹 + 임베딩 + Qdrant 저장)

        Returns:
            성공 여부
        """
        await self.ensure_collection()

        exists = await self._check_key_exists(inp.key)
        if exists:
            logger.info(f"[AdminDocument] Duplicate key, overwriting: {inp.key}")
            await self.delete_document(inp.key)

        admin_upload_dir, dest_file_path = self._prepare_storage(inp)

        chunks = await self._extract_chunks(dest_file_path, inp.file_name)
        if not chunks:
            logger.error(f"[AdminDocument] No chunks extracted: {inp.file_name}")
            shutil.rmtree(admin_upload_dir, ignore_errors=True)
            return False

        all_embeddings, all_sparse = await self._embed_chunks(chunks)
        if len(all_embeddings) != len(chunks):
            logger.error(f"[AdminDocument] Embedding count mismatch: expected {len(chunks)}, got {len(all_embeddings)}")
            shutil.rmtree(admin_upload_dir, ignore_errors=True)
            return False

        points = self._build_points(chunks, all_embeddings, all_sparse, inp)
        await self.client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)
        logger.info(f"[AdminDocument] Added {len(points)} chunks for key={inp.key}")
        return True

    def _prepare_storage(self, inp: AdminDocumentInput) -> Tuple[str, str]:
        """저장 디렉토리 생성 + 파일 복사.

        Returns:
            (admin_upload_dir, dest_file_path)
        """
        key_hash = hashlib.md5(inp.key.encode()).hexdigest()[:16]
        admin_upload_dir = os.path.join(settings.UPLOAD_DIR, "admin", key_hash)
        os.makedirs(admin_upload_dir, exist_ok=True)
        dest_file_path = os.path.join(admin_upload_dir, inp.file_name)
        shutil.copy(inp.file_path, dest_file_path)
        return admin_upload_dir, dest_file_path

    async def _embed_chunks(self, chunks: List[Dict]) -> Tuple[List, List]:
        """청크 리스트에 대해 dense + sparse 임베딩 생성.

        Returns:
            (all_embeddings, all_sparse)
        """
        all_texts = [
            add_source_prefix(c["content"], c.get("metadata", {}).get("source", ""))
            for c in chunks
        ]
        all_sparse = sparse_encoder.encode_batch(all_texts)
        all_embeddings = []
        for i in range(0, len(all_texts), settings.EMBED_BATCH_SIZE):
            batch = all_texts[i:i + settings.EMBED_BATCH_SIZE]
            batch_embeddings = await model_server_client.embed_texts(batch, is_query=False)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings, all_sparse

    def _build_points(
        self,
        chunks: List[Dict],
        all_embeddings: List,
        all_sparse: List,
        inp: AdminDocumentInput,
    ) -> List[models.PointStruct]:
        """청크 + 임베딩으로 Qdrant PointStruct 리스트 생성."""
        regist_date = datetime.now(timezone.utc).isoformat()
        invoke_id = settings.GLOBAL_INVOKE_ID
        points = []
        for chunk, emb, sparse in zip(chunks, all_embeddings, all_sparse):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{invoke_id}:{chunk['id']}"))
            payload = {
                "invoke_id": invoke_id,
                "doc_id": chunk["id"],
                "content": chunk["content"],
                "source": chunk.get("metadata", {}).get("source", ""),
                "page": chunk.get("metadata", {}).get("page", 0),
                "prev_chunk_id": chunk.get("metadata", {}).get("prev_chunk_id"),
                "next_chunk_id": chunk.get("metadata", {}).get("next_chunk_id"),
                "key": inp.key,
                "admin_id": inp.admin_id,
                "admin_name": inp.admin_name,
                "file_size": inp.file_size,
                "regist_date": regist_date,
                "is_use": True,
            }
            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": emb,
                    "sparse": models.SparseVector(indices=sparse["indices"], values=sparse["values"]),
                },
                payload=payload,
            ))
        return points

    async def _iter_pages(self, file_path: str, ext_lower: str) -> List[Tuple[int, str]]:
        """파일 타입별 텍스트 추출 → (page_num, text) 리스트 반환"""
        if ext_lower == '.pdf':
            return await asyncio.to_thread(
                lambda: list(file_text_extractor.iter_pdf_pages(file_path))
            )
        elif ext_lower in ['.hwp', '.hwpx']:
            temp_pdf_dir = tempfile.mkdtemp()
            try:
                converted_pdf = await asyncio.to_thread(
                    file_text_extractor.convert_hwp_to_pdf, file_path, temp_pdf_dir
                )
                if converted_pdf and os.path.exists(converted_pdf):
                    return await asyncio.to_thread(
                        lambda: list(file_text_extractor.iter_pdf_pages(converted_pdf))
                    )
            finally:
                shutil.rmtree(temp_pdf_dir, ignore_errors=True)
            return []
        else:
            text_content = await asyncio.to_thread(file_text_extractor.convert_sync, file_path)
            return [(1, text_content)] if text_content else []

    def _chunk_pages(self, pages: List[Tuple[int, str]], file_name: str) -> List[Dict]:
        """(page_num, text) 리스트 → 청크 리스트 (prev/next 링크 포함)"""
        chunker = IncrementalChunker(
            file_name=file_name,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = []
        for page_num, page_text in pages:
            chunks.extend(chunker.add_page(page_num, page_text))
        chunks.extend(chunker.flush())
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["prev_chunk_id"] = chunks[i - 1]["id"] if i > 0 else None
            chunk["metadata"]["next_chunk_id"] = chunks[i + 1]["id"] if i < len(chunks) - 1 else None
        return chunks

    async def _extract_chunks(self, file_path: str, file_name: str) -> List[Dict]:
        """파일에서 청크 추출"""
        _, ext = os.path.splitext(file_name)
        pages = await self._iter_pages(file_path, ext.lower())
        return self._chunk_pages(pages, file_name)

    async def _check_key_exists(self, key: str) -> bool:
        """key 중복 체크"""
        try:
            results, _ = await self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="key",
                            match=models.MatchValue(value=key),
                        ),
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False,
            )
            return len(results) > 0
        except UnexpectedResponse:
            return False

    async def delete_document(self, key: str) -> bool:
        """key로 문서 삭제"""
        try:
            await self.client.delete(
                collection_name=settings.QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="key",
                                match=models.MatchValue(value=key),
                            ),
                        ]
                    )
                ),
            )

            # 파일 삭제 (경로는 key 해시 사용)
            key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
            admin_upload_dir = os.path.join(settings.UPLOAD_DIR, "admin", key_hash)
            if os.path.exists(admin_upload_dir):
                try:
                    shutil.rmtree(admin_upload_dir)
                except OSError as e:
                    logger.warning(f"[AdminDocument] Failed to delete upload dir {admin_upload_dir}: {e}")

            logger.info(f"[AdminDocument] Deleted document: key={key}")
            return True
        except UnexpectedResponse as e:
            logger.error(f"[AdminDocument] Delete failed: {e}")
            return False

    async def get_document_list(
        self,
        page: int = 1,
        size: int = 10,
        order_type: str = "registDate",
        order: str = "desc",
    ) -> Tuple[List[Dict], int]:
        """
        문서 목록 조회 (페이지네이션)

        Args:
            page: 페이지 번호
            size: 페이지 크기
            order_type: 정렬 기준 (fileName, registDate)
            order: 정렬 방향 (asc, desc)

        Returns:
            (문서 목록, 총 개수)
        """
        try:
            # 전체 문서 조회 (key별 그룹화)
            all_docs = await self._get_all_documents()

            return self._sort_and_paginate(all_docs, order_type, order, page, size)
        except Exception as e:
            logger.error(f"[AdminDocument] Get list failed: {e}")
            return [], 0

    @staticmethod
    def _sort_and_paginate(
        docs: List[Dict], order_type: str, order: str, page: int, size: int
    ) -> tuple[List[Dict], int]:
        """정렬 + 페이지네이션 적용 후 (paged_docs, total_count) 반환."""
        total_count = len(docs)
        sort_field_map = {
            "fileName": "file_name",
            "registDate": "regist_date",
        }
        sort_key = sort_field_map.get(order_type, "regist_date")
        docs.sort(key=lambda x: x.get(sort_key, ""), reverse=(order == "desc"))
        start_idx = (page - 1) * size
        paged = docs[start_idx:start_idx + size]
        for i, doc in enumerate(paged):
            doc["index"] = start_idx + i + 1
        return paged, total_count

    async def _get_all_documents(self) -> List[Dict]:
        """모든 관리자 문서 조회 (key별 그룹화)"""
        try:
            doc_map: Dict[str, Dict] = {}
            offset = None
            
            while True:
                results, next_offset = await self.client.scroll(
                    collection_name=settings.QDRANT_COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="invoke_id",
                                match=models.MatchValue(value=settings.GLOBAL_INVOKE_ID),
                            ),
                            models.FieldCondition(
                                key="key",
                                match=models.MatchExcept(**{"except": [""]}),  # key가 비어있지 않은 것만
                            ),
                            models.IsNullCondition(
                                is_null=models.PayloadField(key="prev_chunk_id")
                            ),
                        ],
                    ),
                    limit=settings.QDRANT_SCROLL_BATCH_SIZE,
                    offset=offset,
                    with_payload=["key", "admin_id", "admin_name", "source", "file_size", "regist_date", "is_use"],
                    with_vectors=False,
                )

                for point in results:
                    payload = point.payload
                    key = payload.get("key", "")
                    if not key:
                        continue

                    if key not in doc_map:
                        doc_map[key] = {
                            "key": key,
                            "admin_id": payload.get("admin_id", ""),
                            "admin_name": payload.get("admin_name", ""),
                            "file_name": payload.get("source", ""),
                            "file_size": payload.get("file_size", 0),
                            "regist_date": payload.get("regist_date", ""),
                            "is_use": payload.get("is_use", True),
                            "chunk_count": 1,
                        }
                
                if next_offset is None:
                    break
                offset = next_offset

            return list(doc_map.values())
        except UnexpectedResponse as e:
            logger.error(f"[AdminDocument] Get all documents failed: {e}")
            return []

    async def search_documents(self, query: DocumentSearchQuery) -> Tuple[List[Dict], int]:
        """
        문서 검색

        Returns:
            (검색 결과, 총 개수)
        """
        try:
            all_docs = await self._get_all_documents()

            # 필터링
            if query.search_term:
                search_term_lower = query.search_term.lower()
                field_map = {
                    "fileName": "file_name",
                    "adminId": "admin_id",
                    "adminName": "admin_name",
                }
                field = field_map.get(query.search_type, "file_name")

                all_docs = [
                    doc for doc in all_docs
                    if search_term_lower in str(doc.get(field, "")).lower()
                ]

            return self._sort_and_paginate(all_docs, query.order_type, query.order, query.page, query.size)
        except Exception as e:
            logger.error(f"[AdminDocument] Search failed: {e}")
            return [], 0

    async def toggle_usage(self, key: str) -> Optional[bool]:
        """
        사용여부 토글

        Returns:
            새로운 is_use 값 (실패 시 None)
        """
        try:
            # 현재 상태 조회
            results, _ = await self.client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="key",
                            match=models.MatchValue(value=key),
                        ),
                    ]
                ),
                limit=1,
                with_payload=["is_use"],
                with_vectors=False,
            )

            if not results:
                logger.warning(f"[AdminDocument] Key not found: {key}")
                return None

            current_is_use = results[0].payload.get("is_use", True)
            new_is_use = not current_is_use

            # 해당 key의 모든 청크 업데이트
            # Qdrant에서 조건부 업데이트는 set_payload로 처리
            await self.client.set_payload(
                collection_name=settings.QDRANT_COLLECTION,
                payload={"is_use": new_is_use},
                points=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="key",
                                match=models.MatchValue(value=key),
                            ),
                        ]
                    )
                ),
            )

            logger.info(f"[AdminDocument] Toggled is_use: key={key}, new_value={new_is_use}")
            return new_is_use
        except UnexpectedResponse as e:
            logger.error(f"[AdminDocument] Toggle failed: {e}")
            return None

    async def get_statistics(self) -> Dict[str, int]:
        """
        문서 통계 조회

        Returns:
            {totalCount, useCount, todayCount, unUseCount}
        """
        try:
            all_docs = await self._get_all_documents()

            total_count = len(all_docs)
            use_count = sum(1 for doc in all_docs if doc.get("is_use", True))
            unuse_count = total_count - use_count

            # 오늘 등록된 문서 수
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            today_count = sum(
                1 for doc in all_docs
                if doc.get("regist_date", "").startswith(today_str)
            )

            return {
                "totalCount": total_count,
                "useCount": use_count,
                "todayCount": today_count,
                "unUseCount": unuse_count,
            }
        except Exception as e:
            logger.error(f"[AdminDocument] Get statistics failed: {e}")
            return {
                "totalCount": 0,
                "useCount": 0,
                "todayCount": 0,
                "unUseCount": 0,
            }

    async def close(self):
        """클라이언트 연결 종료"""
        if self._client is not None:
            await self._client.close()
            self._client = None


# 싱글톤 인스턴스
admin_document_service = AdminDocumentService()
