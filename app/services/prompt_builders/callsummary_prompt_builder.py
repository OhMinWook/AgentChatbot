"""통화 요약용 프롬프트 빌더 - STT 결과를 LLM 요약 요청으로 변환"""

from typing import Dict, Any, Optional
from app.core.config import settings


class CallSummaryPromptBuilder:
    def __init__(self):
        # 기본 시스템 프롬프트: 통화 요약 전문가
        self.default_system_prompt = (
            "너는 통화 내용을 분석하는 전문가야.\n"
      "주어진 통화 녹취록을 **빠짐없이** 분석하여 정리해줘.\n\n"

      "### [분석 방법]\n"
      "1. 먼저 전체 통화를 처음부터 끝까지 읽어\n"
      "2. 누가 누구에게 말하는지 화자를 구분해\n"
      "3. 언급된 모든 주제/사안을 나열해\n"
      "4. 날짜, 시간, 장소, 금액, 이름 등 구체적 정보를 추출해\n"
      "5. 약속이나 합의 사항을 찾아\n\n"

      "### [절대 원칙]\n"
      "1. 없는 내용 창조 금지 - 통화에 없는 내용은 절대 추가하지 마\n"
      "2. 누락 금지 - 언급된 내용은 사소해도 기록해\n"
      "3. 구체적 정보 보존 - 날짜/시간/금액/이름은 정확히 기재\n"
      "4. 정보 부족 시 '알 수 없음' 또는 '언급 없음'으로 표기\n"
        )

        # 기본 출력 포맷
        self.default_output_format = """
다음 형식으로 요약해줘:

## 인물 관계
**추론 결과는 군더더기 없이 명사형 단어 하나 혹은 구문으로만 출력하세요.정보가 부족하면 '알 수 없음' 으로 출력**
(예: 직장 동료, 친구, 비즈니스 파트너, 점원, 손님)

## 📞 통화 한 줄 요약
이 통화의 핵심 목적/결론 (30자 내외)

## 📝 상세 내용
### 주요 대화 흐름
1. [첫 번째 주제] - 내용 정리
2. [두 번째 주제] - 내용 정리
3. ...

### 언급된 구체적 정보
- 날짜/시간: (언급된 경우)
- 장소: (언급된 경우)
- 금액/수량: (언급된 경우)
- 인물/기관: (언급된 경우)
- 기타 중요 정보: (언급된 경우)

## 📅 일정 및 할 일
(※ 통화에서 **명확히 합의된** 내용만 작성)
- [ ] 할 일 내용 (담당자, 기한 있으면 포함)
- 없으면 '합의된 일정 없음'

## 🏷️ 태그
관련 키워드 3~5개
"""

    def build_summary_payload(
        self,
        transcript: str,
        custom_system_prompt: Optional[str] = None,
        custom_output_format: Optional[str] = None,
        additional_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        STT 결과(transcript)를 vLLM 요청 payload로 변환

        :param transcript: STT로 변환된 통화 텍스트
        :param custom_system_prompt: 커스텀 시스템 프롬프트 (None이면 기본값 사용)
        :param custom_output_format: 커스텀 출력 포맷 (None이면 기본값 사용)
        :param additional_instructions: 추가 지시사항
        :return: vLLM API 요청용 payload
        """
        system_prompt = custom_system_prompt or self.default_system_prompt
        output_format = custom_output_format or self.default_output_format

        # 시스템 프롬프트 + 출력 포맷 결합
        full_system_prompt = system_prompt + "\n" + output_format

        if additional_instructions:
            full_system_prompt += f"\n\n[추가 지시사항]\n{additional_instructions}"

        # 사용자 메시지 구성
        user_content = f"다음 통화 내용을 요약해줘:\n\n{transcript}"

        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_content}
        ]

        payload = {
            "model": settings.VLLM_MODEL,
            "messages": messages,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }

        return payload


# 싱글톤 인스턴스
callsummary_prompt_builder = CallSummaryPromptBuilder()