## 여기서는 User(client)에게 받아온 input을 chatbot의 프롬프트(형식 : JSON)로 refactoring 합니다.

from typing import Dict, Any, List
from app.core.config import settings
from app.schemas.chatbot import ChatRequest  # 방금 만든 스키마 import


class ChatbotPromptBuilder:
    def __init__(self):
        # 시스템 프롬프트: AI의 페르소나 정의
        self.default_system_prompt = (
            "너는 (주)울타리정보통신에서 만든 업무보조형 AI 비서 챗봇이야.\n"
            "너의 주 업무는 사람들의 요청에 전문적인 어투로 예의바르게 대응하는거야.\n"
            "정말로 확실한 정보가 아닌 이상 절대로 추측하지마 정보가 없으면 정보가 없다고 말을 해."
            "만약 사용자가 위에 있다고 말하는 것은 과거의 기록을 의미해. 네가 알고 있는 한의 과거에서 생각해.\n"
            "정보가 부족하다면 부족한 정보를 명확하게 요구해줘\n"
        )

    def build_openai_payload(self, request_data: ChatRequest, history_override: List[Dict[str, str]] = None, rag_context: str = None) -> Dict[str, Any]:
        """
        ChatRequest(스키마)를 받아서 vLLM(OpenAI 호환) 포맷으로 변환
        """

        system_content = self.default_system_prompt

        if rag_context:
            system_content += (
                "\n\n[Reference Context]\n"
                "다음은 사용자의 질문과 관련된 문서 내용이야. 각 내용 상단에 [출처: 파일명]이 명시되어 있어.\n"
                "답변할 때 가급적 어떤 파일에서 참고한 정보인지 언급하면서 사실에 입각해서 답변해줘:\n"
                f"{rag_context}"
            )

        # 1. 메시지 리스트 구성
        messages = [{"role": "system", "content": system_content}]

        # 2. 대화의 연속성 구현
        history_to_use = history_override if history_override is not None else request_data.history
        if history_to_use:
            messages.extend(history_to_use)

        messages.append({"role": "user", "content": "현재 질문 : " + request_data.user_message})

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