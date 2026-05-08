"""
LangGraph 노드 공용 유틸리티
- LLM 호출 헬퍼
- 스트리밍 토큰 생성기
- 할루시네이션 위험도 판단
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.langfuse_client import observe, langfuse_context, langfuse  # langfuse 비활성화 스텁
from app.services.api_clients.llm_client import llm_client
from app.core.config import settings
from app.services.utils.llm_payload import build_chat_payload

logger = logging.getLogger(__name__)

# ── JSON 스키마 ──────────────────────────────────────────────────────────────

VERIFY_ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["passed", "issues"]
}

# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class QuestionTask:
    """단일 질문 처리 작업 단위"""
    idx: int
    question: str
    doc_res: Dict
    max_tokens: int
    translate_to: Optional[str] = None


@dataclass
class HallucinationResult:
    """할루시네이션 검증 결과"""
    passed: bool
    issues: List[str]
    question: str


# ── 할루시네이션 위험도 ────────────────────────────────────────────────────────

_HIGH_RISK_QUESTION_PATTERN = re.compile(
    r"""
    몇|얼마|언제|기한|날짜|기간|기준|조건|요건|자격|대상  # 구체적 정보 요구
    | 비율|퍼센트|요율|[%％]                              # 수치 요구
    | 금액|수당|급여|임금|보수|한도|상한|하한              # 금액 요구
    | \d+                                               # 질문 자체에 숫자 포함
    | 이상|이하|초과|미만|이내                            # 범위 조건 질문
    """,
    re.VERBOSE
)


def is_high_risk_question(question: str) -> bool:
    """질문이 수치·날짜·조건 등 고위험 답변을 유도하는지 판단"""
    return bool(_HIGH_RISK_QUESTION_PATTERN.search(question))


# ── LLM 호출 헬퍼 ─────────────────────────────────────────────────────────────

from app.services.api_clients.llm_utils import call_llm, strip_think_blocks, strip_markdown_codeblock, stream_llm_tokens  # noqa: E402


# ── process_question 헬퍼 ────────────────────────────────────────────────────

def build_agent_messages(agent_prompt, question: str, doc_res: Dict, translate_to: str = None) -> List[Dict[str, str]] | None:
    """검색 결과로부터 LLM 메시지를 구성한다. context가 없으면 None."""
    doc_context = doc_res.get("context", "")
    if not doc_context:
        return None
    user_content = agent_prompt.user.format(context=doc_context, question=question)
    if translate_to:
        lang_name = _TRANSLATE_LANG_NAMES.get(translate_to, translate_to)
        user_content += f'\n\n답변 작성 후 반드시 새 줄에 "[TRANSLATION]"을 출력하고, 이어서 위 답변 전체를 {lang_name}으로 번역하여 출력하세요.'
    return [
        {"role": "system", "content": agent_prompt.system},
        {"role": "user", "content": user_content}
    ]


_TRANSLATE_LANG_NAMES = {
    "en": "English",
    "zh": "Chinese (Simplified)",
    "ja": "Japanese",
}


@observe()
async def generate_single_answer(agent_prompt, task: QuestionTask) -> Dict:
    """단일 질문에 대해 검색 결과 기반 답변을 생성한다.
    - 고위험 질문: LLM pre-generate (검증용)
    - 저위험 질문: 스트리밍 준비만 (answer="")
    """
    idx, question, doc_res, max_tokens, translate_to = (
        task.idx, task.question, task.doc_res, task.max_tokens, task.translate_to
    )
    references = doc_res.get("references", [])
    rag_docs = doc_res.get("results", [])
    context = doc_res.get("context", "")
    messages = build_agent_messages(agent_prompt, question, doc_res, translate_to=translate_to)

    if messages is None:
        return {
            "idx": idx, "question": question,
            "answer": f"'{question}'에 대한 관련 문서를 찾지 못했습니다.",
            "context": "", "messages": None, "sources": [], "rag_docs": []
        }

    high_risk = is_high_risk_question(question)
    try:
        langfuse_context.update_current_observation(metadata={"high_risk": high_risk})
    except Exception:
        pass
    if high_risk:
        logger.info(f"[Process] 고위험 질문 - pre-generate: {question[:50]}")
        answer = await call_llm(messages, max_tokens=max_tokens)
    else:
        logger.info(f"[Process] 저위험 질문 - 스트리밍 준비: {question[:50]}")
        answer = ""

    return {
        "idx": idx, "question": question,
        "answer": answer,
        "context": context if high_risk else "",
        "messages": messages,
        "sources": references,
        "rag_docs": rag_docs
    }


