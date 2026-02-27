"""대화록 요약용 프롬프트 빌더 - 날짜별/전체 대화 요약 LLM 요청 payload 생성"""

from typing import Dict, Any, List
from app.services.utils.llm_payload import build_chat_payload


class DialoguePromptBuilder:
    def __init__(self):
        # 출력 형식 (공통)
        self.output_format = (
            "### 대화 주제\n"
            "- 주제1\n"
            "  - 세부 내용\n"
            "  - 세부 내용\n"
            "- 주제2\n"
            "  - 세부 내용\n\n"
            "### 주요 대화 흐름\n"
            "- 논의1\n"
            "  - 세부 내용\n"
            "- 논의2\n"
            "  - 세부 내용\n\n"
        )

        self.daily_system_prompt = (
            "너는 대화 내용을 분석하고 요약하는 전문가야.\n"
            "주어진 대화 내용을 분석하여 대화 주제를 추출해줘.\n\n"
            "### [분석 방법]\n"
            "1. 화자를 구분하고 각자의 입장을 파악해\n"
            "2. 논의된 주제와 세부내용을 정리해\n"
            "### [절대 원칙]\n"
            "1. 없는 내용 창조 금지 - 대화에 없는 내용은 절대 추가하지 마\n"
            "2. 구체적 정보 보존 - 날짜/시간/금액/이름은 정확히 기재\n\n"
            "### [출력 형식]\n"
            "### 대화 주제\n"
            "- 주제1\n"
            "  - 세부 내용\n"
            "  - 세부 내용\n"
            "- 주제2\n"
            "  - 세부 내용"
        )

        self.overall_system_prompt = (
            "너는 대화 내용을 종합 분석하는 전문가야.\n"
            "여러 날에 걸친 대화 요약들을 종합하여 전체 흐름과 핵심 내용을 정리해줘.\n\n"
            "### [분석 방법]\n"
            "1. 각 요일의 대화의 흐름과 맥락을 파악해\n"
            "2. 반복적으로 등장하는 주제나 핵심 사안을 식별해\n"
            "3. 대화 주제나 대화의 흐름을 정리해\n\n"
            "### [절대 원칙]\n"
            "1. 없는 내용 창조 금지\n"
            "2. 핵심 주제 누락 금지\n"
            "3. 날짜별 요약에 기반하여 종합적으로 정리\n"
            "4. 언제부터 언제까지 논의 된 내용인지 명시\n\n"
            f"### [출력 형식]\n{self.output_format}"
        )

    def build_daily_summary_payload(self, date: str, dialogue_text: str) -> Dict[str, Any]:
        """
        날짜별 대화 텍스트를 vLLM 요약 요청 payload로 변환

        :param date: 대화 날짜 (예: "2025-01-15")
        :param dialogue_text: 해당 날짜의 대화 텍스트
        :return: vLLM API 요청용 payload
        """
        user_content = f"다음은 {date} 날짜의 대화 내용이야. 요약해줘:\n\n{dialogue_text}"

        messages = [
            {"role": "system", "content": self.daily_system_prompt},
            {"role": "user", "content": user_content}
        ]

        return build_chat_payload(messages)

    def build_overall_summary_payload(self, daily_summaries: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        날짜별 요약들을 종합하여 전체 요약 요청 payload로 변환

        :param daily_summaries: [{"date": "2025-01-15", "summary": "..."}, ...] 형태의 리스트
        :return: vLLM API 요청용 payload
        """
        summaries_text = "\n\n".join(
            f"[{s['date']}]\n{s['summary']}" for s in daily_summaries
        )
        user_content = f"다음은 날짜별 대화 요약이야. 전체 내용을 종합적으로 요약해줘:\n\n{summaries_text}"

        messages = [
            {"role": "system", "content": self.overall_system_prompt},
            {"role": "user", "content": user_content}
        ]

        return build_chat_payload(messages)


# 싱글톤 인스턴스
dialogue_prompt_builder = DialoguePromptBuilder()
