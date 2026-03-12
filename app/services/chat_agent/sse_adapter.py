"""
SSE 기반 Human-in-the-loop 어댑터 (토큰 스트리밍 지원)
"""

import hashlib
import json
import logging
import os
import re
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.services.chat_agent.graph import create_rag_graph
from app.services.chat_agent.node_utils import stream_llm_tokens
from app.services.rag.qdrant_service import qdrant_service
from app.services.utils.download_service import download_service
from app.services.utils.sse_utils import SSEType

logger = logging.getLogger(__name__)

# precomputed 답변을 작은 청크로 나눌 때 사용할 크기
_CHUNK_SIZE = 6

# pending_threads TTL (초)
class SSEGraphAdapter:
    """LangGraph와 SSE 스트리밍을 연결하는 어댑터"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = create_rag_graph(self.checkpointer)

    def _generate_thread_id(self, invoke_id: str) -> str:
        return f"{invoke_id}_{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # 마크다운 정규화
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_markdown(text: str) -> str:
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)  # --- 제거
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # 헤더 # 제거
        text = re.sub(r"\*+", "", text)  # ** 전부 제거
        text = re.sub(r"\n{3,}", "\n\n", text)  # 연속 빈 줄 → 최대 1줄
        text = text.strip()
        return text

    @staticmethod
    def _normalize_line(line: str) -> str:
        """한 줄에 대해 마크다운 정규화 적용"""
        line = re.sub(r"^-{3,}$", "", line)    # --- 제거
        line = re.sub(r"^#{1,6}\s+", "", line)  # 헤더 # 제거
        line = re.sub(r"\*+", "", line)          # ** 제거
        return line

    # ------------------------------------------------------------------
    # 토큰 스트리밍 헬퍼
    # ------------------------------------------------------------------
    async def _stream_answer_tokens(
        self, streaming_payload: dict, t_start: float = None
    ) -> AsyncGenerator[bytes, None]:
        """streaming_payload를 기반으로 answer 이벤트를 청크 단위로 전송"""
        first_token_logged = False

        def log_ttft():
            nonlocal first_token_logged
            if not first_token_logged and t_start:
                logger.info(f"[Timing] TTFT: {time.perf_counter() - t_start:.2f}s")
                first_token_logged = True

        def send_text(text: str):
            for i in range(0, len(text), _CHUNK_SIZE):
                yield self._format_sse({"type": SSEType.ANSWER, "content": text[i:i + _CHUNK_SIZE]})

        if streaming_payload.get("precomputed"):
            content = self._normalize_markdown(streaming_payload.get("content", ""))
            for i in range(0, len(content), _CHUNK_SIZE):
                log_ttft()
                yield self._format_sse({"type": SSEType.ANSWER, "content": content[i:i + _CHUNK_SIZE]})
        else:
            messages = streaming_payload.get("messages", [])
            max_tokens = streaming_payload.get("max_tokens", 2048)

            line_buffer = ""
            consecutive_newlines = 0
            at_line_start = True

            async for token in stream_llm_tokens(messages, max_tokens):
                line_buffer += token

                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    normalized = self._normalize_line(line)

                    if normalized.strip() == "":
                        consecutive_newlines += 1
                        if consecutive_newlines <= 2:
                            log_ttft()
                            yield self._format_sse({"type": SSEType.ANSWER, "content": "\n"})
                    else:
                        consecutive_newlines = 0
                        text_to_send = normalized + "\n"
                        for i in range(0, len(text_to_send), _CHUNK_SIZE):
                            log_ttft()
                            yield self._format_sse({"type": SSEType.ANSWER, "content": text_to_send[i:i + _CHUNK_SIZE]})
                    at_line_start = True

                # 줄 중간 토큰 즉시 전송 (헤더 줄은 \n 올 때까지 버퍼링)
                if line_buffer:
                    is_header = at_line_start and re.match(r'^#{1,6}', line_buffer)
                    if not is_header:
                        cleaned = re.sub(r"\*+", "", line_buffer)
                        if cleaned:
                            log_ttft()
                            yield self._format_sse({"type": SSEType.ANSWER, "content": cleaned})
                        line_buffer = ""
                        at_line_start = False

            # 마지막 줄 처리 (줄바꿈 없는 마지막 내용)
            if line_buffer.strip():
                normalized = self._normalize_line(line_buffer)
                if normalized:
                    for i in range(0, len(normalized), _CHUNK_SIZE):
                        log_ttft()
                        yield self._format_sse({"type": SSEType.ANSWER, "content": normalized[i:i + _CHUNK_SIZE]})

    # ------------------------------------------------------------------
    # 파일 경로 해석 유틸리티
    # ------------------------------------------------------------------
    def _resolve_file_path(
        self, invoke_id: str, source: str, metadata: Dict
    ) -> Optional[str]:
        """Admin/일반 문서 구분하여 실제 파일 경로 반환"""
        key = metadata.get("key")
        if key:
            # Admin 문서: uploaded_files/admin/{key_hash}/filename
            key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
            file_path = os.path.join(settings.UPLOAD_DIR, "admin", key_hash, source)
        else:
            # 일반 문서: uploaded_files/{invoke_id}/filename
            file_path = os.path.join(settings.UPLOAD_DIR, invoke_id, source)

        if os.path.exists(file_path):
            return file_path
        return None

    # ------------------------------------------------------------------
    # 공통: 그래프 실행 결과에서 답변/레퍼런스 SSE 전송
    # ------------------------------------------------------------------
    async def _emit_final_answer(
        self, final_state: dict, t_start: float = None
    ) -> AsyncGenerator[bytes, None]:
        """final_state에서 references + 토큰 스트리밍 답변을 SSE로 emit"""
        streaming_payload = final_state.get("streaming_payload")

        # 레퍼런스 전송
        agent_answers = final_state.get("agent_answers", [])
        all_refs = []
        all_rag_docs = []
        for ans in agent_answers:
            for ref in ans.get("sources", []):
                if ref not in all_refs:
                    all_refs.append(ref)
            # RAG 문서 수집 (디버깅용)
            for doc in ans.get("rag_docs", []):
                all_rag_docs.append(doc)

        # Open API인 경우에만 다운로드 URL 추가 (filter_filename=None)
        is_open_mode = final_state.get("filter_filename") is None
        invoke_id = final_state.get("invoke_id", "")

        if all_refs and is_open_mode:
            # source 목록 추출
            sources = [ref.get("source") for ref in all_refs if ref.get("source")]
            if sources:
                # 메타데이터 조회
                metadata_map = await qdrant_service.get_document_metadata_by_source(
                    invoke_id, sources
                )

                # 각 reference에 download_url 추가
                for ref in all_refs:
                    source = ref.get("source")
                    if source and source in metadata_map:
                        metadata = metadata_map[source]
                        file_path = self._resolve_file_path(invoke_id, source, metadata)
                        if file_path:
                            link_info = download_service.create_download_link_from_path(
                                file_path=file_path,
                                filename=source,
                                expires_in_seconds=3600,
                                one_time=False
                            )
                            if link_info:
                                ref["download_url"] = link_info["download_url"]

        if all_refs:
            yield self._format_sse({"type": SSEType.REFERENCES, "docs": all_refs})

        # RAG 문서 전체 전송 (디버깅용)
        if all_rag_docs:
            yield self._format_sse({"type": SSEType.RAG_DOCUMENTS, "documents": all_rag_docs})

        # 토큰 스트리밍 답변
        if streaming_payload:
            async for chunk in self._stream_answer_tokens(streaming_payload, t_start=t_start):
                yield chunk
        else:
            # fallback: streaming_payload 없으면 기존 방식
            messages = final_state.get("messages", [])
            final_answer = None
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    final_answer = msg.content
                    break
            if final_answer:
                for i in range(0, len(final_answer), _CHUNK_SIZE):
                    yield self._format_sse({"type": SSEType.ANSWER, "content": final_answer[i:i + _CHUNK_SIZE]})

    # ------------------------------------------------------------------
    # invoke_with_sse
    # ------------------------------------------------------------------
    async def invoke_with_sse(
        self,
        invoke_id: str,
        user_query: str,
        thread_id: Optional[str] = None,
        filter_filename: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """그래프 실행 및 SSE 스트리밍"""
        t_start = time.perf_counter()

        if not thread_id:
            thread_id = self._generate_thread_id(invoke_id)

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "invoke_id": invoke_id,
            "original_query": user_query,
            "messages": [HumanMessage(content=user_query)],
            "rewritten_questions": [],
            "agent_answers": [],
            "filter_filename": filter_filename,
            "streaming_payload": None
        }

        try:
            yield self._format_sse({"type": SSEType.PROGRESS, "step": "답변을 준비하고 있습니다"})

            final_state = None
            async for event in self.graph.astream(initial_state, config, stream_mode="values"):
                final_state = event

            if final_state:
                async for chunk in self._emit_final_answer(final_state, t_start=t_start):
                    yield chunk

            yield self._format_sse({"type": SSEType.DONE})

        except Exception as e:
            logger.exception(f"Graph execution error: {e}")
            yield self._format_sse({"type": SSEType.ERROR, "message": str(e)})

    def _format_sse(self, data: Dict[str, Any]) -> bytes:
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n".encode("utf-8")

sse_graph_adapter = SSEGraphAdapter()
