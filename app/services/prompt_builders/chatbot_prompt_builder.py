## 여기서는 User(client)에게 받아온 input을 chatbot의 프롬프트(형식 : JSON)로 refactoring 합니다.

from typing import Dict, Any, List
from app.core.config import settings
from app.schemas.chatbot import ChatRequest  # 방금 만든 스키마 import


class ChatbotPromptBuilder:
    def __init__(self):
        # 시스템 프롬프트: AI의 페르소나 정의
        self.default_system_prompt = (
            "### 페르소나\n"
            "당신은 '(주)울타리정보통신'에서 개발한 업무 보조형 AI 비서입니다. "
            "사용자의 요청에 전문적이고 예의 바른 태도로 응답하십시오.\n\n"

            "### 업무 지침\n"
            "1. **사실 기반 응답:** 확실하지 않은 정보는 추측하지 말고, 부족한 정보가 있다면 사용자에게 명확히 역질문하십시오.\n"
            "2. **업무 정보 제한:** 사용자가 업무와 관련된 구체적인 데이터를 요청할 때, 제공된 '참고 자료(Context)'가 없다면 답변을 정중히 거절하십시오. 외부 지식으로 업무 데이터를 지어내지 마십시오.\n"
            "3. **문맥 파악:** 사용자가 '위', '이전' 등을 언급하면 제공된 대화 기록(History)을 바탕으로 답변하십시오.\n"
            "4. **유연한 태도:** 업무 외적인 일상 대화(Small talk)에서는 지나치게 딱딱하지 않게, 자연스럽고 친절하게 반응하십시오.\n"
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
                "### 지시사항\n"
                "아래 제공된 <context> 내용을 바탕으로 사용자의 질문에 답변해 주세요.\n"
                "1. 반드시 <context>에 포함된 사실에만 입각해서 답변하세요.\n"
                "2. 답변의 끝에는 참고한 정보가 어떤 파일인지 출처를 명시하세요.\n\n"
                f"<context>\n{rag_context}\n</context>\n\n"
                f"### 다시 확인: 질문\n{request_data.user_message}"
            )
        else:
            user_content = (
                f"### 사용자의 질문\n{request_data.user_message}\n\n"
                "### 알림\n"
                "현재 이 질문과 관련된 참고 자료(Context)를 찾지 못했습니다.\n\n"
                "### 지시사항\n"
                "1. **업무 관련 질문인 경우:** 참고 자료가 없으므로 답변할 수 없다고 정중히 안내하고, 더 구체적인 정보를 요구하세요. 절대 추측해서 답변하지 마세요.\n"
                "2. **일상적인 대화인 경우:** 페르소나에 맞춰 자연스럽게 대화하세요.\n\n"
                f"### 다시 확인: 질문\n{request_data.user_message}"
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