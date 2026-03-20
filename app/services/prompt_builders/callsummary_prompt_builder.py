"""통화 요약용 프롬프트 빌더 - STT 결과를 LLM 요약 요청으로 변환"""

import tiktoken
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.services.utils.llm_payload import build_chat_payload


class CallSummaryPromptBuilder:

    def __init__(self):
        # tiktoken 인코더 초기화 (cl100k_base: GPT-4 기준)
        self._encoder = tiktoken.get_encoding("cl100k_base")
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

        # 청크별 요약용 시스템 프롬프트 (간결하게)
        self.chunk_system_prompt = (
            "너는 통화 내용을 분석하는 전문가야.\n"
            "주어진 통화 녹취록 일부를 요약해줘.\n\n"
            "### [절대 원칙]\n"
            "1. 없는 내용 창조 금지\n"
            "2. 핵심 내용, 구체적 정보(날짜/시간/금액/이름), 합의 사항 포함\n"
            "3. 이전 요약이 제공되면 문맥을 이어서 요약\n"
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

        return build_chat_payload(messages)

    def count_tokens(self, text: str) -> int:
        """tiktoken으로 토큰 수 측정"""
        return len(self._encoder.encode(text))

    def needs_chunking(self, transcript: str) -> bool:
        """청크 분할이 필요한지 확인"""
        return self.count_tokens(transcript) >= settings.CALLSUMMARY_CHUNK_THRESHOLD

    def split_into_chunks(self, transcript: str) -> List[str]:
        """
        통화록을 오버랩이 있는 청크로 분할 (토큰 기준)

        Returns: 청크 리스트
        """
        tokens = self._encoder.encode(transcript)
        chunks = []
        start = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = min(start + settings.CALLSUMMARY_CHUNK_SIZE, total_tokens)

            # 청크 토큰을 텍스트로 디코딩
            chunk_tokens = tokens[start:end]
            chunk_text = self._encoder.decode(chunk_tokens)

            # 마지막 청크가 아니면 문장 경계에서 자르기 시도
            if end < total_tokens:
                # 청크 끝 부근에서 문장 끝(. ! ? 줄바꿈) 찾기
                search_start = max(len(chunk_text) - 300, 0)
                last_break = -1
                for i in range(len(chunk_text) - 1, search_start, -1):
                    if chunk_text[i] in '.!?\n':
                        last_break = i + 1
                        break
                if last_break > 0:
                    chunk_text = chunk_text[:last_break]
                    # 실제 사용된 토큰 수 재계산
                    end = start + len(self._encoder.encode(chunk_text))

            chunks.append(chunk_text.strip())

            # 다음 시작점 (오버랩 적용)
            start = end - settings.CALLSUMMARY_CHUNK_OVERLAP
            if start >= total_tokens or end >= total_tokens:
                break

        return chunks

    def build_chunk_payload(
        self,
        chunk: str,
        chunk_index: int,
        total_chunks: int,
        previous_summary: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        청크별 요약 요청 payload 생성

        :param chunk: 현재 청크 텍스트
        :param chunk_index: 청크 인덱스 (0부터 시작)
        :param total_chunks: 전체 청크 수
        :param previous_summary: 이전 청크의 요약 (문맥 연결용)
        """
        if previous_summary:
            user_content = (
                f"[이전 내용 요약]\n{previous_summary}\n\n"
                f"---\n\n"
                f"[통화 내용 ({chunk_index + 1}/{total_chunks} 부분)]\n{chunk}\n\n"
                f"위 내용을 이전 요약과 문맥이 이어지도록 요약해줘."
            )
        else:
            user_content = (
                f"[통화 내용 ({chunk_index + 1}/{total_chunks} 부분)]\n{chunk}\n\n"
                f"위 내용을 요약해줘."
            )

        messages = [
            {"role": "system", "content": self.chunk_system_prompt},
            {"role": "user", "content": user_content}
        ]

        return build_chat_payload(messages, max_tokens=1024)

    def build_final_summary_payload(self, combined_summary: str) -> Dict[str, Any]:
        """
        청크별 요약을 통합한 최종 요약 요청 payload 생성
        """
        user_content = (
            f"다음은 긴 통화를 여러 부분으로 나눠 요약한 내용이야.\n"
            f"이를 하나의 완성된 요약으로 정리해줘:\n\n{combined_summary}"
        )

        messages = [
            {"role": "system", "content": self.default_system_prompt + "\n" + self.default_output_format},
            {"role": "user", "content": user_content}
        ]

        return build_chat_payload(messages)


# 싱글톤 인스턴스
callsummary_prompt_builder = CallSummaryPromptBuilder()