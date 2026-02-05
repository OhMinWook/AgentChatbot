"""
관리자 문서 관리 전용 서비스

Qdrant를 직접 사용하여 관리자 문서의 CRUD 작업을 처리합니다.
"""

import logging
import os
import uuid
import shutil
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.services.api_clients.model_server_client import model_server_client
from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.rag.sparse_encoder import sparse_encoder

logger = logging.getLogger(__name__)


class AdminDocumentService:
    """관리자 문서 관리 전용 서비스"""

    EMBED_BATCH_SIZE = 50

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

    async def add_document(
        self,
        key: str,
        admin_id: str,
        admin_name: str,
        file_path: str,
        file_name: str,
        file_size: int,
    ) -> bool:
        """
        문서 등록 (청킹 + 임베딩 + Qdrant 저장)

        Args:
            key: 문서 고유 키
            admin_id: 관리자 ID
            admin_name: 관리자 이름
            file_path: 업로드된 파일 경로
            file_name: 원본 파일명
            file_size: 파일 크기 (bytes)

        Returns:
            성공 여부
        """
        await self.ensure_collection()

        # 중복 key면 기존 삭제 후 덮어쓰기
        exists = await self._check_key_exists(key)
        if exists:
            logger.info(f"[AdminDocument] Duplicate key, overwriting: {key}")
            await self.delete_document(key)

        # 파일 저장 경로: uploaded_files/admin/{hash}/ (경로 길이 제한 방지)
        import hashlib
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        admin_upload_dir = os.path.join(settings.UPLOAD_DIR, "admin", key_hash)
        os.makedirs(admin_upload_dir, exist_ok=True)

        # 파일 복사
        dest_file_path = os.path.join(admin_upload_dir, file_name)
        shutil.copy(file_path, dest_file_path)

        # 청킹 수행 (rag_ingestion_service의 청킹 로직 재사용)
        chunks = await self._extract_chunks(dest_file_path, file_name)

        if not chunks:
            logger.error(f"[AdminDocument] No chunks extracted: {file_name}")
            # 실패 시 파일 삭제
            shutil.rmtree(admin_upload_dir, ignore_errors=True)
            return False

        # 임베딩 (dense + sparse)
        all_embeddings = []
        all_sparse = []
        for i in range(0, len(chunks), self.EMBED_BATCH_SIZE):
            batch_chunks = chunks[i:i + self.EMBED_BATCH_SIZE]
            texts = [c["content"] for c in batch_chunks]

            # Dense 임베딩
            embeddings = await model_server_client.embed_texts(texts, is_query=False)
            if len(embeddings) != len(batch_chunks):
                logger.error(f"[AdminDocument] Embedding count mismatch")
                shutil.rmtree(admin_upload_dir, ignore_errors=True)
                return False
            all_embeddings.extend(embeddings)

            # Sparse 임베딩
            sparse_vectors = sparse_encoder.encode_batch(texts)
            all_sparse.extend(sparse_vectors)

        # Qdrant에 저장 (관리자 메타데이터 포함)
        regist_date = datetime.now(timezone.utc).isoformat()
        invoke_id = settings.GLOBAL_INVOKE_ID

        points = []
        for chunk, emb, sparse in zip(chunks, all_embeddings, all_sparse):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{invoke_id}:{chunk['id']}"))

            payload = {
                # 기존 필드
                "invoke_id": invoke_id,
                "doc_id": chunk["id"],
                "content": chunk["content"],
                "source": chunk.get("metadata", {}).get("source", ""),
                "page": chunk.get("metadata", {}).get("page", 0),
                "prev_chunk_id": chunk.get("metadata", {}).get("prev_chunk_id"),
                "next_chunk_id": chunk.get("metadata", {}).get("next_chunk_id"),
                # 관리자 확장 필드
                "key": key,
                "admin_id": admin_id,
                "admin_name": admin_name,
                "file_size": file_size,
                "regist_date": regist_date,
                "is_use": True,
            }

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": emb,
                    "sparse": models.SparseVector(
                        indices=sparse["indices"],
                        values=sparse["values"],
                    ),
                },
                payload=payload,
            ))

        await self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
        )

        logger.info(f"[AdminDocument] Added {len(points)} chunks for key={key}")
        return True

    async def _extract_chunks(self, file_path: str, file_name: str) -> List[Dict]:
        """파일에서 청크 추출 (rag_ingestion_service 내부 로직 활용)"""
        import asyncio
        import tempfile

        _, ext = os.path.splitext(file_name)
        ext_lower = ext.lower()

        chunks = []

        if ext_lower == '.pdf':
            # PDF 직접 처리
            page_generator = await asyncio.to_thread(
                lambda: list(rag_ingestion_service._iter_pdf_pages(file_path))
            )

            from app.services.rag.rag_ingestion_service import IncrementalChunker
            chunker = IncrementalChunker(
                file_name=file_name,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
            )

            for page_num, page_text in page_generator:
                new_chunks = chunker.add_page(page_num, page_text)
                chunks.extend(new_chunks)

            remaining = chunker.flush()
            chunks.extend(remaining)

        elif ext_lower in ['.hwp', '.hwpx']:
            # HWP -> PDF 변환 후 처리
            temp_pdf_dir = tempfile.mkdtemp()
            try:
                converted_pdf = await asyncio.to_thread(
                    rag_ingestion_service._convert_hwp_to_pdf_with_win32com,
                    file_path,
                    temp_pdf_dir,
                )
                if converted_pdf and os.path.exists(converted_pdf):
                    page_generator = await asyncio.to_thread(
                        lambda: list(rag_ingestion_service._iter_pdf_pages(converted_pdf))
                    )

                    from app.services.rag.rag_ingestion_service import IncrementalChunker
                    chunker = IncrementalChunker(
                        file_name=file_name,
                        chunk_size=settings.CHUNK_SIZE,
                        chunk_overlap=settings.CHUNK_OVERLAP,
                    )

                    for page_num, page_text in page_generator:
                        new_chunks = chunker.add_page(page_num, page_text)
                        chunks.extend(new_chunks)

                    remaining = chunker.flush()
                    chunks.extend(remaining)
            finally:
                shutil.rmtree(temp_pdf_dir, ignore_errors=True)

        else:
            # 기타 문서 (MarkItDown 사용)
            if rag_ingestion_service._markitdown:
                try:
                    import asyncio
                    result = await asyncio.to_thread(
                        rag_ingestion_service._markitdown.convert, file_path
                    )
                    if result and result.text_content:
                        chunks = rag_ingestion_service._create_chunks_simple(
                            result.text_content, file_name
                        )
                except Exception as e:
                    logger.error(f"[AdminDocument] MarkItDown error: {e}")

        # prev/next 청크 ID 연결
        for i, chunk in enumerate(chunks):
            chunk["metadata"]["prev_chunk_id"] = chunks[i - 1]["id"] if i > 0 else None
            chunk["metadata"]["next_chunk_id"] = chunks[i + 1]["id"] if i < len(chunks) - 1 else None

        return chunks

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
            import hashlib
            key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
            admin_upload_dir = os.path.join(settings.UPLOAD_DIR, "admin", key_hash)
            if os.path.exists(admin_upload_dir):
                shutil.rmtree(admin_upload_dir, ignore_errors=True)

            logger.info(f"[AdminDocument] Deleted document: key={key}")
            return True
        except UnexpectedResponse as e:
            logger.error(f"[AdminDocument] Delete failed: {e}")
            return False

    async def get_document_list(
        self,
        page: int = 1,
        size: int = 10,
    ) -> Tuple[List[Dict], int]:
        """
        문서 목록 조회 (페이지네이션)

        Returns:
            (문서 목록, 총 개수)
        """
        try:
            # 전체 문서 조회 (key별 그룹화)
            all_docs = await self._get_all_documents()

            total_count = len(all_docs)

            # 등록일 기준 내림차순 정렬
            all_docs.sort(key=lambda x: x.get("regist_date", ""), reverse=True)

            # 페이지네이션 적용
            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paged_docs = all_docs[start_idx:end_idx]

            # index 재할당 (페이지 내 순번)
            for i, doc in enumerate(paged_docs):
                doc["index"] = start_idx + i + 1

            return paged_docs, total_count
        except Exception as e:
            logger.error(f"[AdminDocument] Get list failed: {e}")
            return [], 0

    async def _get_all_documents(self) -> List[Dict]:
        """모든 관리자 문서 조회 (key별 그룹화)"""
        try:
            results, _ = await self.client.scroll(
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
                    ]
                ),
                limit=100000,
                with_payload=["key", "admin_id", "admin_name", "source", "file_size", "regist_date", "is_use"],
                with_vectors=False,
            )

            # key별 그룹화
            doc_map: Dict[str, Dict] = {}
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
                        "chunk_count": 0,
                    }
                doc_map[key]["chunk_count"] += 1

            return list(doc_map.values())
        except UnexpectedResponse as e:
            logger.error(f"[AdminDocument] Get all documents failed: {e}")
            return []

    async def search_documents(
        self,
        search_type: str,
        search_term: str,
        page: int = 1,
        size: int = 10,
    ) -> Tuple[List[Dict], int]:
        """
        문서 검색

        Args:
            search_type: 검색 유형 (fileName, adminId, adminName)
            search_term: 검색어
            page: 페이지 번호
            size: 페이지 크기

        Returns:
            (검색 결과, 총 개수)
        """
        try:
            all_docs = await self._get_all_documents()

            # 필터링
            if search_term:
                search_term_lower = search_term.lower()
                field_map = {
                    "fileName": "file_name",
                    "adminId": "admin_id",
                    "adminName": "admin_name",
                }
                field = field_map.get(search_type, "file_name")

                all_docs = [
                    doc for doc in all_docs
                    if search_term_lower in str(doc.get(field, "")).lower()
                ]

            total_count = len(all_docs)

            # 정렬 및 페이지네이션
            all_docs.sort(key=lambda x: x.get("regist_date", ""), reverse=True)

            start_idx = (page - 1) * size
            end_idx = start_idx + size
            paged_docs = all_docs[start_idx:end_idx]

            for i, doc in enumerate(paged_docs):
                doc["index"] = start_idx + i + 1

            return paged_docs, total_count
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
