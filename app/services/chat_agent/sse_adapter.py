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

from langgraph.checkpoint.memory import MemorySaver
from app.core.langfuse_client import observe, langfuse, langfuse_context  # langfuse 비활성화 스텁
from app.core.config import settings
from app.services.chat_agent.graph import create_rag_graph
from app.services.chat_agent.node_utils import stream_llm_tokens, call_llm
from app.services.rag.qdrant_service import qdrant_service
from app.services.utils.answer_cache_service import answer_cache_service
from app.services.utils.download_service import download_service
from app.services.utils.sse_utils import SSEType, normalize_markdown, format_sse_bytes, TRANSLATE_LANG_NAMES

logger = logging.getLogger(__name__)

# precomputed 답변을 작은 청크로 나눌 때 사용할 크기 (settings.SSE_CHUNK_SIZE)

# pending_threads TTL (초)
class SSEGraphAdapter:
    """LangGraph와 SSE 스트리밍을 연결하는 어댑터"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = create_rag_graph(self.checkpointer)

    def _generate_thread_id(self, invoke_id: str) -> str:
        return f"{invoke_id}_{uuid.uuid4().hex[:8]}"




    # ------------------------------------------------------------------
    # 토큰 스트리밍 헬퍼
    # ------------------------------------------------------------------
    async def _emit_precomputed_chunks(
        self, content: str, translate_to: Optional[str]
    ) -> AsyncGenerator[bytes, None]:
        """precomputed 답변을 청크 단위로 전송. 번역 구분자가 있으면 ANSWER/TRANSLATION 분리."""
        DELIMITER = "[TRANSLATION]"
        content = normalize_markdown(content)
        if translate_to and DELIMITER in content:
            korean, _, translation = content.partition(DELIMITER)
            korean = korean.strip()
            translation = translation.strip()
            for i in range(0, len(korean), settings.SSE_CHUNK_SIZE):
                yield self._format_sse({"type": SSEType.ANSWER, "content": korean[i:i + settings.SSE_CHUNK_SIZE]})
            for i in range(0, len(translation), settings.SSE_CHUNK_SIZE):
                yield self._format_sse({"type": SSEType.TRANSLATION, "lang": translate_to, "content": translation[i:i + settings.SSE_CHUNK_SIZE]})
        else:
            for i in range(0, len(content), settings.SSE_CHUNK_SIZE):
                yield self._format_sse({"type": SSEType.ANSWER, "content": content[i:i + settings.SSE_CHUNK_SIZE]})

    async def _emit_streamed_tokens(
        self, messages: list, max_tokens: int, translate_to: Optional[str]
    ) -> AsyncGenerator[bytes, None]:
        """LLM 스트리밍으로 토큰 단위 전송. [TRANSLATION] 구분자 기준으로 ANSWER/TRANSLATION 이벤트 분리."""
        DELIMITER = "[TRANSLATION]"
        D_LEN = len(DELIMITER)
        accumulated = ""
        in_translation = False

        async for token in stream_llm_tokens(messages, max_tokens):
            if token:
                token = re.sub(r"\*+", "", token)
                accumulated += token

                if not in_translation:
                    idx = accumulated.find(DELIMITER)
                    if idx != -1:
                        before = accumulated[:idx].rstrip("\n")
                        after = accumulated[idx + D_LEN:].lstrip("\n")
                        if before:
                            yield self._format_sse({"type": SSEType.ANSWER, "content": before})
                        in_translation = True
                        accumulated = after
                        if accumulated and translate_to:
                            yield self._format_sse({"type": SSEType.TRANSLATION, "lang": translate_to, "content": accumulated})
                        accumulated = ""
                    else:
                        safe_len = max(0, len(accumulated) - D_LEN)
                        if safe_len > 0:
                            yield self._format_sse({"type": SSEType.ANSWER, "content": accumulated[:safe_len]})
                            accumulated = accumulated[safe_len:]
                else:
                    if translate_to and accumulated:
                        yield self._format_sse({"type": SSEType.TRANSLATION, "lang": translate_to, "content": accumulated})
                    accumulated = ""

        if accumulated:
            if in_translation and translate_to:
                yield self._format_sse({"type": SSEType.TRANSLATION, "lang": translate_to, "content": accumulated})
            else:
                yield self._format_sse({"type": SSEType.ANSWER, "content": accumulated})

    async def _stream_answer_tokens(
        self, streaming_payload: dict, t_start: float = None
    ) -> AsyncGenerator[bytes, None]:
        """streaming_payload 모드에 따라 precomputed 또는 스트리밍 경로로 분기."""
        translate_to = streaming_payload.get("translate_to")
        if streaming_payload.get("precomputed"):
            async for chunk in self._emit_precomputed_chunks(streaming_payload.get("content", ""), translate_to):
                yield chunk
        else:
            async for chunk in self._emit_streamed_tokens(
                streaming_payload.get("messages", []),
                streaming_payload.get("max_tokens", 2048),
                translate_to,
            ):
                yield chunk

    # ------------------------------------------------------------------
    # 레퍼런스/RAG 문서 수집 헬퍼
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_refs_and_docs(agent_answers: list) -> tuple:
        """agent_answers에서 중복 없이 refs와 rag_docs를 수집한다."""
        all_refs = []
        all_rag_docs = []
        for ans in agent_answers:
            for ref in ans.get("sources", []):
                if ref not in all_refs:
                    all_refs.append(ref)
            for doc in ans.get("rag_docs", []):
                all_rag_docs.append(doc)
        return all_refs, all_rag_docs

    async def _enhance_refs_with_download_urls(
        self, all_refs: list, invoke_id: str
    ) -> None:
        """all_refs에 다운로드 URL을 in-place로 추가한다 (Open 모드 전용)."""
        sources = [ref.get("source") for ref in all_refs if ref.get("source")]
        if not sources:
            return
        metadata_map = await qdrant_service.get_document_metadata_by_source(invoke_id, sources)
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
                        one_time=False,
                    )
                    if link_info:
                        ref["download_url"] = link_info["download_url"]

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
        all_refs, all_rag_docs = self._collect_refs_and_docs(agent_answers)

        # Open API인 경우에만 다운로드 URL 추가 (filter_filename=None)
        is_open_mode = final_state.get("filter_filename") is None
        invoke_id = final_state.get("invoke_id", "")

        if all_refs and is_open_mode:
            await self._enhance_refs_with_download_urls(all_refs, invoke_id)

        if all_refs:
            yield self._format_sse({"type": SSEType.REFERENCES, "docs": all_refs})

        # RAG 문서 전체 전송 (디버깅용)
        if all_rag_docs:
            yield self._format_sse({"type": SSEType.RAG_DOCUMENTS, "documents": all_rag_docs})

        # 토큰 스트리밍 답변
        if streaming_payload:
            async for chunk in self._stream_answer_tokens(streaming_payload, t_start=t_start):
                yield chunk

    # ------------------------------------------------------------------
    # references만 SSE로 emit (번역 모드용)
    # ------------------------------------------------------------------
    async def _emit_references(self, final_state: dict) -> AsyncGenerator[bytes, None]:
        """references + rag_documents SSE emit (한국어 스트리밍 없이)"""
        agent_answers = final_state.get("agent_answers", [])
        all_refs, all_rag_docs = self._collect_refs_and_docs(agent_answers)

        is_open_mode = final_state.get("filter_filename") is None
        invoke_id = final_state.get("invoke_id", "")

        if all_refs and is_open_mode:
            await self._enhance_refs_with_download_urls(all_refs, invoke_id)

        if all_refs:
            yield self._format_sse({"type": SSEType.REFERENCES, "docs": all_refs})
        if all_rag_docs:
            yield self._format_sse({"type": SSEType.RAG_DOCUMENTS, "documents": all_rag_docs})

    # ------------------------------------------------------------------
    # 번역 스트리밍 헬퍼
    # ------------------------------------------------------------------
    async def _stream_translation(self, text: str, translate_to: str) -> AsyncGenerator[bytes, None]:
        """한국어 답변을 지정 언어로 번역 후 SSE 스트리밍"""
        lang_name = TRANSLATE_LANG_NAMES.get(translate_to, translate_to)
        messages = [
            {
                "role": "system",
                "content": f"Translate the following Korean text to {lang_name}. Output only the translation without any explanations or additional text."
            },
            {"role": "user", "content": text}
        ]
        async for token in stream_llm_tokens(messages, max_tokens=settings.DEFAULT_MAX_TOKENS):
            if token:
                yield self._format_sse({"type": SSEType.TRANSLATION, "lang": translate_to, "content": token})

    async def _emit_cached_answer(
        self, cached: dict, user_query: str, translate_to: Optional[str], invoke_id: str
    ) -> AsyncGenerator[bytes, None]:
        """캐시 히트 시 저장된 답변/번역을 SSE로 emit."""
        if cached.get("references"):
            yield self._format_sse({"type": SSEType.REFERENCES, "docs": cached["references"]})
        content = cached.get("answer", "")
        try:
            langfuse_context.update_current_observation(
                input=user_query,
                output=content,
                metadata={"cache_hit": True},
            )
        except Exception:
            pass
        for i in range(0, len(content), settings.SSE_CHUNK_SIZE):
            yield self._format_sse({"type": SSEType.ANSWER, "content": content[i:i + settings.SSE_CHUNK_SIZE]})
        if translate_to:
            cached_translation = cached.get("translations", {}).get(translate_to)
            if cached_translation:
                for i in range(0, len(cached_translation), settings.SSE_CHUNK_SIZE):
                    yield self._format_sse({"type": SSEType.TRANSLATION, "lang": translate_to, "content": cached_translation[i:i + settings.SSE_CHUNK_SIZE]})
            else:
                translation_buffer = []
                async for chunk in self._stream_translation(content, translate_to):
                    data = json.loads(chunk.decode("utf-8").removeprefix("data: ").strip())
                    if data.get("type") == SSEType.TRANSLATION:
                        translation_buffer.append(data.get("content", ""))
                    yield chunk
                full_translation = "".join(translation_buffer)
                if full_translation:
                    await answer_cache_service.add_translation(invoke_id, user_query, translate_to, full_translation)
        yield self._format_sse({"type": SSEType.DONE})

    async def _run_graph(self, initial_state: dict, config: dict) -> Optional[dict]:
        """그래프를 실행하고 최종 상태를 반환한다."""
        final_state = None
        async for event in self.graph.astream(initial_state, config, stream_mode="values"):
            final_state = event
        return final_state

    # ------------------------------------------------------------------
    # invoke_with_sse
    # ------------------------------------------------------------------
    @observe(capture_input=False, capture_output=False)
    async def invoke_with_sse(
        self,
        invoke_id: str,
        user_query: str,
        thread_id: Optional[str] = None,
        filter_filename: Optional[str] = None,
        translate_to: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """그래프 실행 및 SSE 스트리밍"""
        t_start = time.perf_counter()

        if not thread_id:
            thread_id = self._generate_thread_id(invoke_id)

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "invoke_id": invoke_id,
            "original_query": user_query,
            "agent_answers": [],
            "filter_filename": filter_filename,
            "translate_to": translate_to,
            "streaming_payload": None
        }

        try:
            # 캐시 확인 (오픈 챗에서만)
            is_open = filter_filename is None
            cached = await answer_cache_service.get(invoke_id, user_query) if is_open else None
            if cached:
                async for chunk in self._emit_cached_answer(cached, user_query, translate_to, invoke_id):
                    yield chunk
                return

            yield self._format_sse({"type": SSEType.PROGRESS, "step": "답변을 준비하고 있습니다"})

            final_state = await self._run_graph(initial_state, config)

            if final_state:
                answer_buffer = []
                translation_buffer = []
                t_first_token = None

                async def _collecting_emit():
                    nonlocal t_first_token
                    async for chunk in self._emit_final_answer(final_state, t_start=t_start):
                        data = json.loads(chunk.decode("utf-8").removeprefix("data: ").strip())
                        if data.get("type") == SSEType.ANSWER:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                                logger.info(f"[Timing] TTFT: {t_first_token - t_start:.2f}s")
                            answer_buffer.append(data.get("content", ""))
                        elif data.get("type") == SSEType.TRANSLATION:
                            translation_buffer.append(data.get("content", ""))
                        yield chunk

                async for chunk in _collecting_emit():
                    yield chunk

                # 캐시 저장 (관련 문서가 실제로 사용된 경우에만)
                full_answer = "".join(answer_buffer)
                if full_answer:
                    try:
                        langfuse_context.update_current_observation(
                            input=user_query,
                            output=full_answer,
                        )
                    except Exception:
                        pass

                    refs = []
                    for ans in final_state.get("agent_answers", []):
                        for ref in ans.get("sources", []):
                            if ref not in refs:
                                refs.append(ref)
                    if is_open and refs:
                        await answer_cache_service.set(invoke_id, user_query, full_answer, refs)

                full_translation = "".join(translation_buffer)
                if full_translation and is_open and refs:
                    await answer_cache_service.add_translation(invoke_id, user_query, translate_to, full_translation)

            logger.info(f"[Timing] Total: {time.perf_counter() - t_start:.2f}s")
            yield self._format_sse({"type": SSEType.DONE})

        except Exception as e:
            logger.exception(f"Graph execution error: {e}")
            yield self._format_sse({"type": SSEType.ERROR, "message": str(e)})

    def _format_sse(self, data: Dict[str, Any]) -> bytes:
        return format_sse_bytes(data)

sse_graph_adapter = SSEGraphAdapter()
