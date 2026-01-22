## 여기서는 User(client)에게 받아온 input을 chatbot의 프롬프트(형식 : JSON)로 refactoring 합니다.

from typing import Dict, Any, List
from app.core.config import settings
from app.schemas.chatbot import ChatRequest  # 방금 만든 스키마 import


class ChatbotPromptBuilder:
    def __init__(self):
        # 시스템 프롬프트: AI의 페르소나 정의
        self.default_system_prompt = (
            "### 역할(Persona)\n"
            "당신은 '(주)울타리정보통신'의 사내 업무 지원을 위해 개발된 AI 비서입니다. "
            "제공된 자료를 바탕으로 사용자의 질문에 전문적이고 명확하며, 신뢰할 수 있는 답변을 제공해야 합니다.\n\n"

            "### 업무 원칙(Principles)\n"
            "1. **사실 중심 답변:** 제공된 <참고 자료>에 포함된 내용에만 근거하여 답변하십시오. 자료에 없는 내용은 절대 추측하거나 지어내지 마십시오.\n"
            "2. **데이터 무결성 판단:** OCR(광학 문자 인식)로 변환된 텍스트에는 오탈자나 서식 붕괴가 포함될 수 있습니다. 텍스트가 심하게 훼손되어 정보의 신뢰성을 담보할 수 없는 경우, 구체적인 수치 언급을 피하고 개요 위주로 설명하십시오.\n"
            "3. **정중한 태도:** 사용자와의 대화에서는 항상 예의 바르고 격식 있는 '해요체' 또는 '하십시오체'를 사용하십시오.\n"
        )

    def build_openai_payload(self, request_data: ChatRequest, history_override: List[Dict[str, str]] = None, rag_context: str = None) -> Dict[str, Any]:
        """
        ChatRequest(스키마)를 받아서 vLLM(OpenAI 호환) 포맷으로 변환
        """

        # 1. 시스템 프롬프트 (페르소나만 정의, RAG 컨텍스트 제외)
        messages = [{"role": "system", "content": self.default_system_prompt}]

        # 2. 대화의 연속성 구현 (과거 대화 기록)
        history_to_use = history_override if history_override is not None else request_data.history
        if history_to_use:
            messages.extend(history_to_use)

        # 3. 현재 턴의 user 메시지 (RAG 컨텍스트 + 질문)
        if rag_context:
            user_content = (
                f"### 사용자의 질문\n{request_data.user_message}\n\n"

                "### 참고 자료\n"
                "다음은 사내 문서에서 추출한 텍스트입니다. 각 문서는 <document> 태그로 구분되어 있으며, source(파일명)와 page(페이지 번호) 속성을 포함합니다.\n"
                "표나 서식이 일부 깨져 있을 수 있습니다.\n\n"
                f"{rag_context}\n\n"

                "### 지시사항(Instructions)\n"
                "위 <documents> 내의 자료를 면밀히 분석하여 다음 단계에 따라 답변을 생성하십시오.\n"
                "**중요: 제공된 참고 자료의 내용이 이전 대화에서 언급된 내용과 다를 경우, 반드시 참고 자료를 우선하여 답변하십시오.**\n\n"

                "**1단계: 자료 유효성 검토**\n"
                "- 각 <document>의 내용이 사용자의 질문과 관련이 있는지 확인하십시오.\n"
                "- 표나 데이터의 구조가 심하게 깨져 있어(예: 숫자와 특수문자의 무의미한 나열) 내용 파악이 불가능한지 판단하십시오.\n\n"

                "**2단계: 답변 전략 수립**\n"
                "- **자료가 명확한 경우:** 사용자의 질문에 대해 구체적이고 상세하게 답변하십시오.\n"
                "- **자료가 불명확한 경우(깨짐 발생):** 억지로 수치를 해석하지 말고, **\"~에 대한 표가 포함되어 있으나, 정확한 수치는 원본 문서를 확인해 주십시오.\"**와 같이 문서의 주제와 존재 여부만 안내하십시오.\n"
                "- **관련 자료가 아닌 경우:** 질문에 답변할 수 없음을 정중히 밝히고 거절하십시오.\n\n"

                "**3단계: 답변 출력**\n"
                "- 사용자가 이해하기 쉽도록 깔끔하게 정리하여 답변하십시오.\n\n"

                f"### 사용자의 질문\n{request_data.user_message}"  # 모델이 여기서부터 바로 답변을 시작하도록 유도
            )

        else:
            # 자료가 없을 때 (Hallucination 방지 최우선)
            user_content = (
                f"### 사용자의 질문\n{request_data.user_message}\n\n"

                "### 알림\n"
                "- 현재 질문과 연관된 <참고 자료>가 검색되지 않았습니다.\n\n"

                "### 지시사항\n"
                "1. **업무 관련 질문:** 근거 자료가 없으므로 답변할 수 없음을 정중히 안내하고, 문서를 업로드하거나 더 구체적인 키워드로 질문해 달라고 요청하십시오. (절대 지식을 날조하지 마십시오.)\n"
                "2. **일상 대화:** 인사나 가벼운 대화인 경우, AI 비서로서 자연스럽고 친절하게 응대하십시오.\n\n"

                f"### 사용자의 질문\n{request_data.user_message}"
            )

        messages.append({"role": "user", "content": user_content})

        # 2. 최종 Payload 조립 (LLM 모델 설정값 주입)
        payload = {
            "model": settings.VLLM_MODEL,  # config.py에서 가져옴
            "messages": messages,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }

        return payload


# 인스턴스 생성
prompt_builder = ChatbotPromptBuilder()