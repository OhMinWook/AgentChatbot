"""
SSE 기반 Human-in-the-loop 어댑터
"""

import json
import logging
import uuid
from typing import AsyncGenerator, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.services.rag_agent.graph import create_rag_graph

logger = logging.getLogger(__name__)


class SSEGraphAdapter:
    """LangGraph와 SSE 스트리밍을 연결하는 어댑터"""

    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = create_rag_graph(self.checkpointer)

    def _generate_thread_id(self, invoke_id: str) -> str:
        return f"{invoke_id}_{uuid.uuid4().hex[:8]}"

    async def invoke_with_sse(
        self,
        invoke_id: str,
        user_query: str,
        thread_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """그래프 실행 및 SSE 스트리밍"""
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
            "awaiting_human_input": False
        }

        try:
            yield self._format_sse({"type": "progress", "step": "시작"})

            final_state = None
            async for event in self.graph.astream(initial_state, config, stream_mode="values"):
                final_state = event

                if event.get("conversation_summary") and not event.get("question_is_clear"):
                    yield self._format_sse({"type": "progress", "step": "대화 맥락 분석 중"})

                if event.get("rewritten_questions"):
                    questions = event.get("rewritten_questions", [])
                    yield self._format_sse({
                        "type": "progress",
                        "step": f"질문 분석 완료 ({len(questions)}개 질문)"
                    })

            if final_state:
                if final_state.get("awaiting_human_input"):
                    clarification = final_state.get("clarification_message", "질문을 더 구체적으로 해주세요.")
                    yield self._format_sse({
                        "type": "clarification_needed",
                        "message": clarification,
                        "thread_id": thread_id
                    })
                    return

                messages = final_state.get("messages", [])
                final_answer = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        final_answer = msg.content
                        break

                if final_answer:
                    agent_answers = final_state.get("agent_answers", [])
                    all_refs = []
                    for ans in agent_answers:
                        for ref in ans.get("sources", []):
                            if ref not in all_refs:
                                all_refs.append(ref)

                    if all_refs:
                        yield self._format_sse({"type": "references", "docs": all_refs})
                    yield self._format_sse({"type": "answer", "content": final_answer})

            yield self._format_sse({"type": "done"})

        except Exception as e:
            logger.exception(f"Graph execution error: {e}")
            yield self._format_sse({"type": "error", "message": str(e)})

    async def continue_with_sse(
        self,
        invoke_id: str,
        thread_id: str,
        human_response: str
    ) -> AsyncGenerator[bytes, None]:
        """Human-in-the-loop 후 그래프 재개"""
        config = {"configurable": {"thread_id": thread_id}}

        try:
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
                    "awaiting_human_input": False
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

                messages = final_state.get("messages", [])
                final_answer = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        final_answer = msg.content
                        break

                if final_answer:
                    agent_answers = final_state.get("agent_answers", [])
                    all_refs = []
                    for ans in agent_answers:
                        for ref in ans.get("sources", []):
                            if ref not in all_refs:
                                all_refs.append(ref)

                    if all_refs:
                        yield self._format_sse({"type": "references", "docs": all_refs})
                    yield self._format_sse({"type": "answer", "content": final_answer})

            yield self._format_sse({"type": "done"})

        except Exception as e:
            logger.exception(f"Graph continuation error: {e}")
            yield self._format_sse({"type": "error", "message": str(e)})

    def _format_sse(self, data: Dict[str, Any]) -> bytes:
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n".encode("utf-8")


sse_graph_adapter = SSEGraphAdapter()
