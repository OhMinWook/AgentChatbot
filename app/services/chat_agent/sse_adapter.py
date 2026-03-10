"""
SSE 기반 Human-in-the-loop 어댑터 (토큰 스트리밍 지원)
"""

import hashlib
import json
import logging
import os
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, Tuple, List

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
_PENDING_THREAD_TTL = 600  # 10분


class SSEGraphAdapter:
    """LangGraph와 SSE 스트리밍을 연결하는 어댑터"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = create_rag_graph(self.checkpointer)
        self._pending_threads: Dict[str, Tuple[str, float]] = {}  # invoke_id → (thread_id, timestamp)

    def _generate_thread_id(self, invoke_id: str) -> str:
        return f"{invoke_id}_{uuid.uuid4().hex[:8]}"

    def cancel_pending(self, invoke_id: str):
        """대기 중인 명확화 세션을 폐기한다. 새 요청이 들어왔을 때 호출."""
        pending = self._pending_threads.pop(invoke_id, None)
        if pending:
            old_thread, _ = pending
            # MemorySaver 내부 체크포인트 정리
            self.checkpointer.storage.pop(old_thread, None)
            logger.info(f"[HITL] 대기 세션 폐기: {old_thread} (invokeId: {invoke_id})")
        # 만료된 세션들도 정리
        self._cleanup_expired_pending()

    def _cleanup_expired_pending(self):
        """TTL이 만료된 pending 세션들을 정리"""
        now = time.time()
        expired_keys = [
            invoke_id for invoke_id, (thread_id, ts) in self._pending_threads.items()
            if now - ts > _PENDING_THREAD_TTL
        ]
        for invoke_id in expired_keys:
            pending = self._pending_threads.pop(invoke_id, None)
            if pending:
                thread_id, _ = pending
                self.checkpointer.storage.pop(thread_id, None)
                logger.info(f"[HITL] 만료 세션 정리: {thread_id} (invokeId: {invoke_id})")

    # ------------------------------------------------------------------
    # 마크다운 정규화
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_markdown(text: str) -> str:
        import re
        text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)  # --- 제거
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # 헤더 # 제거
        text = re.sub(r"\*+", "", text)  # ** 전부 제거
        text = re.sub(r"\n{3,}", "\n\n", text)  # 연속 빈 줄 → 최대 1줄
        text = text.strip()
        return text

    # ------------------------------------------------------------------
    # 토큰 스트리밍 헬퍼
    # ------------------------------------------------------------------
    async def _stream_answer_tokens(
        self, streaming_payload: dict
    ) -> AsyncGenerator[bytes, None]:
        """streaming_payload를 기반으로 answer 이벤트를 청크 단위로 전송"""
        if streaming_payload.get("precomputed"):
            content = self._normalize_markdown(streaming_payload.get("content", ""))
            for i in range(0, len(content), _CHUNK_SIZE):
                yield self._format_sse({"type": SSEType.ANSWER, "content": content[i:i + _CHUNK_SIZE]})
        else:
            # 전체 토큰 수집 후 정규화하여 전송
            messages = streaming_payload.get("messages", [])
            max_tokens = streaming_payload.get("max_tokens", 2048)
            full_text = ""
            async for token in stream_llm_tokens(messages, max_tokens):
                full_text += token
            content = self._normalize_markdown(full_text)
            for i in range(0, len(content), _CHUNK_SIZE):
                yield self._format_sse({"type": SSEType.ANSWER, "content": content[i:i + _CHUNK_SIZE]})

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
        self, final_state: dict
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
            async for chunk in self._stream_answer_tokens(streaming_payload):
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
        # 기존 명확화 대기 세션이 있으면 폐기
        self.cancel_pending(invoke_id)

        if not thread_id:
            thread_id = self._generate_thread_id(invoke_id)

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "invoke_id": invoke_id,
            "original_query": user_query,
            "messages": [HumanMessage(content=user_query)],
            "question_is_clear": False,
            "conversation_summary": "",
            "rewritten_questions": [],
            "agent_answers": [],
            "clarification_message": None,
            "awaiting_human_input": False,
            "filter_filename": filter_filename,
            "streaming_payload": None
        }

        try:
            yield self._format_sse({"type": SSEType.PROGRESS, "step": "질문을 확인하고 있습니다"})

            final_state = None
            sent_progress = set()  # 중복 방지

            async for event in self.graph.astream(initial_state, config, stream_mode="values"):
                final_state = event

                # 대화 요약 진행 (1회만)
                if event.get("conversation_summary") and "summary" not in sent_progress:
                    sent_progress.add("summary")
                    yield self._format_sse({"type": SSEType.PROGRESS, "step": "이전 대화 내용을 확인하고 있습니다"})

                # 질문 분석 완료 (1회만)
                if event.get("rewritten_questions") and "analyzed" not in sent_progress:
                    sent_progress.add("analyzed")
                    yield self._format_sse({
                        "type": SSEType.PROGRESS,
                        "step": "질문 분석 완료"
                    })

            if final_state:
                if final_state.get("awaiting_human_input"):
                    async for chunk in self._handle_clarification(
                        invoke_id, thread_id, final_state, "질문을 더 구체적으로 해주세요."
                    ):
                        yield chunk
                    return

                async for chunk in self._emit_final_answer(final_state):
                    yield chunk

            yield self._format_sse({"type": SSEType.DONE})

        except Exception as e:
            logger.exception(f"Graph execution error: {e}")
            yield self._format_sse({"type": SSEType.ERROR, "message": str(e)})

    # ------------------------------------------------------------------
    # continue_with_sse
    # ------------------------------------------------------------------
    async def continue_with_sse(
        self,
        invoke_id: str,
        thread_id: str,
        human_response: str
    ) -> AsyncGenerator[bytes, None]:
        """Human-in-the-loop 후 그래프 재개"""
        config = {"configurable": {"thread_id": thread_id}}

        try:
            # 이미 다른 요청에 의해 폐기된 세션인지 확인
            pending = self._pending_threads.get(invoke_id)
            if not pending or pending[0] != thread_id:
                yield self._format_sse({
                    "type": SSEType.ERROR,
                    "message": "세션이 만료되었습니다. 새로 질문해 주세요."
                })
                return

            # TTL 만료 체크
            _, ts = pending
            if time.time() - ts > _PENDING_THREAD_TTL:
                self._pending_threads.pop(invoke_id, None)
                self.checkpointer.storage.pop(thread_id, None)
                yield self._format_sse({
                    "type": SSEType.ERROR,
                    "message": "세션이 만료되었습니다. 새로 질문해 주세요."
                })
                return

            # 정상 재개 - pending 해제
            self._pending_threads.pop(invoke_id, None)

            current_state = await self.graph.aget_state(config)

            if not current_state or not current_state.values:
                yield self._format_sse({
                    "type": SSEType.ERROR,
                    "message": "세션을 찾을 수 없습니다."
                })
                return

            # rewritten 쿼리 우선 사용, 없으면 original_query
            rewritten = current_state.values.get("rewritten_questions", [])
            base_query = rewritten[0] if rewritten else current_state.values.get("original_query", "")
            combined_query = f"{base_query} {human_response}" if base_query else human_response

            await self.graph.aupdate_state(
                config,
                {
                    "messages": [HumanMessage(content=combined_query)],
                    "original_query": combined_query,
                    "awaiting_human_input": False,
                    "streaming_payload": None
                }
            )

            yield self._format_sse({"type": SSEType.PROGRESS, "step": "답변을 준비하고 있습니다"})

            final_state = None
            async for event in self.graph.astream(None, config, stream_mode="values"):
                final_state = event

            if final_state: 
                if final_state.get("awaiting_human_input"):
                    async for chunk in self._handle_clarification(
                        invoke_id, thread_id, final_state, "조금 더 구체적으로 설명해 주세요."
                    ):
                        yield chunk
                    return

                async for chunk in self._emit_final_answer(final_state):
                    yield chunk

            yield self._format_sse({"type": SSEType.DONE})

        except Exception as e:
            logger.exception(f"Graph continuation error: {e}")
            yield self._format_sse({"type": SSEType.ERROR, "message": str(e)})

    # 사용자 입력이 필요한 상태를 등록하고 그 사실을 SSE로 클라이언트에게 알리는 역할 헬퍼
    async def _handle_clarification(
        self,
        invoke_id: str,
        thread_id: str,
        final_state: dict,
        default_message: str,
    ) -> AsyncGenerator[bytes, None]:
        """명확화 대기 상태 등록 및 SSE 이벤트 전송"""
        self._pending_threads[invoke_id] = (thread_id, time.time())
        clarification = final_state.get("clarification_message", default_message)
        yield self._format_sse({
            "type": SSEType.CLARIFICATION,
            "message": clarification,
            "thread_id": thread_id
        })

    def _format_sse(self, data: Dict[str, Any]) -> bytes:
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n".encode("utf-8")

sse_graph_adapter = SSEGraphAdapter()
