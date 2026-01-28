"""
SSE 기반 Human-in-the-loop 어댑터 (토큰 스트리밍 지원)
"""

import json
import logging
import uuid
from typing import AsyncGenerator, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.services.rag_agent.graph import create_rag_graph
from app.services.rag_agent.nodes import stream_llm_tokens

logger = logging.getLogger(__name__)

# precomputed 답변을 작은 청크로 나눌 때 사용할 크기
_CHUNK_SIZE = 6


class SSEGraphAdapter:
    """LangGraph와 SSE 스트리밍을 연결하는 어댑터"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = create_rag_graph(self.checkpointer)
        self._pending_threads: Dict[str, str] = {}  # invoke_id → thread_id (명확화 대기 중)

    def _generate_thread_id(self, invoke_id: str) -> str:
        return f"{invoke_id}_{uuid.uuid4().hex[:8]}"

    def cancel_pending(self, invoke_id: str):
        """대기 중인 명확화 세션을 폐기한다. 새 요청이 들어왔을 때 호출."""
        old_thread = self._pending_threads.pop(invoke_id, None)
        if old_thread:
            # MemorySaver 내부 체크포인트 정리
            self.checkpointer.storage.pop(old_thread, None)
            logger.info(f"[HITL] 대기 세션 폐기: {old_thread} (invokeId: {invoke_id})")

    # ------------------------------------------------------------------
    # 토큰 스트리밍 헬퍼
    # ------------------------------------------------------------------
    async def _stream_answer_tokens(
        self, streaming_payload: dict
    ) -> AsyncGenerator[bytes, None]:
        """streaming_payload를 기반으로 answer 이벤트를 청크 단위로 전송"""
        if streaming_payload.get("precomputed"):
            # 단일 답변: 이미 완성된 텍스트를 작은 청크로 나눠서 스트리밍 효과
            content = streaming_payload.get("content", "")
            for i in range(0, len(content), _CHUNK_SIZE):
                chunk = content[i:i + _CHUNK_SIZE]
                yield self._format_sse({"type": "answer", "content": chunk})
        else:
            # 복수 답변: vLLM SSE 스트리밍으로 실시간 토큰 전송
            messages = streaming_payload.get("messages", [])
            max_tokens = streaming_payload.get("max_tokens", 2048)
            async for token in stream_llm_tokens(messages, max_tokens):
                yield self._format_sse({"type": "answer", "content": token})

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
        for ans in agent_answers:
            for ref in ans.get("sources", []):
                if ref not in all_refs:
                    all_refs.append(ref)
        if all_refs:
            yield self._format_sse({"type": "references", "docs": all_refs})

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
                    yield self._format_sse({"type": "answer", "content": final_answer[i:i + _CHUNK_SIZE]})

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
            yield self._format_sse({"type": "progress", "step": "시작"})

            final_state = None
            sent_progress = set()  # 중복 방지

            async for event in self.graph.astream(initial_state, config, stream_mode="values"):
                final_state = event

                # 대화 요약 진행 (1회만)
                if event.get("conversation_summary") and "summary" not in sent_progress:
                    sent_progress.add("summary")
                    yield self._format_sse({"type": "progress", "step": "대화 맥락 분석 중"})

                # 질문 분석 완료 (1회만)
                if event.get("rewritten_questions") and "analyzed" not in sent_progress:
                    sent_progress.add("analyzed")
                    questions = event.get("rewritten_questions", [])
                    yield self._format_sse({
                        "type": "progress",
                        "step": f"질문 분석 완료 ({len(questions)}개 질문)"
                    })

            if final_state:
                if final_state.get("awaiting_human_input"):
                    # 명확화 대기 상태 등록
                    self._pending_threads[invoke_id] = thread_id
                    clarification = final_state.get("clarification_message", "질문을 더 구체적으로 해주세요.")
                    yield self._format_sse({
                        "type": "clarification_needed",
                        "message": clarification,
                        "thread_id": thread_id
                    })
                    return

                async for chunk in self._emit_final_answer(final_state):
                    yield chunk

            yield self._format_sse({"type": "done"})

        except Exception as e:
            logger.exception(f"Graph execution error: {e}")
            yield self._format_sse({"type": "error", "message": str(e)})

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
            if pending != thread_id:
                yield self._format_sse({
                    "type": "error",
                    "message": "세션이 만료되었습니다. 새로 질문해 주세요."
                })
                return

            # 정상 재개 - pending 해제
            self._pending_threads.pop(invoke_id, None)

            current_state = await self.graph.aget_state(config)

            if not current_state or not current_state.values:
                yield self._format_sse({
                    "type": "error",
                    "message": "세션을 찾을 수 없습니다."
                })
                return

            await self.graph.aupdate_state(
                config,
                {
                    "messages": [HumanMessage(content=human_response)],
                    "original_query": human_response,
                    "awaiting_human_input": False,
                    "streaming_payload": None
                }
            )

            yield self._format_sse({"type": "progress", "step": "명확화 응답 처리 중"})

            final_state = None
            async for event in self.graph.astream(None, config, stream_mode="values"):
                final_state = event

            if final_state:
                if final_state.get("awaiting_human_input"):
                    clarification = final_state.get("clarification_message", "조금 더 구체적으로 설명해 주세요.")
                    yield self._format_sse({
                        "type": "clarification_needed",
                        "message": clarification,
                        "thread_id": thread_id
                    })
                    return

                async for chunk in self._emit_final_answer(final_state):
                    yield chunk

            yield self._format_sse({"type": "done"})

        except Exception as e:
            logger.exception(f"Graph continuation error: {e}")
            yield self._format_sse({"type": "error", "message": str(e)})

    def _format_sse(self, data: Dict[str, Any]) -> bytes:
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n".encode("utf-8")


sse_graph_adapter = SSEGraphAdapter()
