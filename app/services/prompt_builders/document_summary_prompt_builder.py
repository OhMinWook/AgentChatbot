"""문서 요약용 프롬프트 빌더 - 질문 분해 기반 체계적 요약"""

from typing import Dict, Any, List
from app.core.config import settings


class DocumentSummaryPromptBuilder:
    # 20페이지 이상이면 질문 분해 방식, 미만이면 단순 검색 방식
    DECOMPOSE_THRESHOLD = 20

    def __init__(self):
        # 질문 분해 방식용 질문 (20페이지 이상)
        self.decomposed_questions = [
            {
                "key": "purpose",
                "question": "이 문서는 왜 작성되었고, 무엇을 위한 문서인가?",
                "description": "문서 목적"
            },
            {
                "key": "main_points",
                "question": "이 문서에서 가장 중요한 내용들은 무엇인가?",
                "description": "주요 내용"
            },
            {
                "key": "structure",
                "question": "이 문서의 전체 구조와 흐름은 어떻게 되는가?",
                "description": "문서 구조"
            }
        ]

        # 개별 질문 답변용 프롬프트
        self.qa_system_prompt = (
            "너는 문서 분석 전문가야.\n"
            "주어진 문서 내용을 바탕으로 질문에 정확히 답변해.\n\n"
            "### 원칙\n"
            "1. 문서에 있는 내용만 답변해 (없으면 '해당 정보 없음')\n"
            "2. 구체적인 정보(이름, 수치, 날짜 등)는 정확히 기재\n"
            "3. 간결하게 핵심만 답변 (3~5문장 이내)\n"
        )

        # 출력 형식 (공통)
        self.output_format = (
            "### 문서 요약\n"
            "(문서 핵심을 2~4문장으로)\n\n"
            "### 문서의 목적\n"
            "(이 문서가 왜 작성되었는지)\n\n"
            "### 주요 내용\n"
            "- 항목1\n"
            "- 항목2\n"
            "- ..."
        )

        # 최종 통합 요약 프롬프트 (질문 분해 방식용)
        self.merge_system_prompt = (
            "너는 문서 요약 전문가야.\n"
            "아래는 하나의 문서에 대해 분석한 결과야.\n"
            "이를 바탕으로 문서 전체를 요약해줘.\n\n"
            "### 원칙\n"
            "1. 최대한 자세한 부분까지 설명해\n"
            "2. 이 문서를 읽지 않은 사람도 핵심을 파악할 수 있게\n\n"
            f"### 출력 형식\n{self.output_format}"
        )

        # 단순 요약 프롬프트 (20페이지 미만용)
        self.simple_summary_prompt = (
            "너는 문서 요약 전문가야.\n"
            "주어진 문서 내용을 바탕으로 요약해줘.\n\n"
            "### 원칙\n"
            "1. 최대한 자세한 부분까지 설명해\n"
            "2. 이 문서를 읽지 않은 사람도 핵심을 파악할 수 있게\n\n"
            f"### 출력 형식\n{self.output_format}"
        )

    def needs_decomposition(self, total_pages: int) -> bool:
        """질문 분해가 필요한지 확인 (20페이지 이상)"""
        return total_pages >= self.DECOMPOSE_THRESHOLD

    def get_questions(self) -> List[Dict[str, str]]:
        """질문 분해 방식용 질문 목록 반환"""
        return self.decomposed_questions

    def build_simple_summary_payload(self, context: str, file_name: str) -> Dict[str, Any]:
        """단순 요약용 payload 생성 (20페이지 미만)"""
        user_content = f"[문서: {file_name}]\n\n{context}\n\n위 문서를 요약해줘."

        return {
            "model": settings.VLLM_MODEL,
            "messages": [
                {"role": "system", "content": self.simple_summary_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }

    def build_qa_payload(self, question: str, context: str) -> Dict[str, Any]:
        """개별 질문-답변용 payload 생성"""
        user_content = f"[문서 내용]\n{context}\n\n[질문]\n{question}"

        return {
            "model": settings.VLLM_MODEL,
            "messages": [
                {"role": "system", "content": self.qa_system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 512,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }

    def build_merge_payload(self, qa_results: Dict[str, str], file_name: str) -> Dict[str, Any]:
        """분석 결과 통합용 payload 생성"""
        # 질문 분해 방식에서 사용한 질문 목록
        questions = self.get_questions()

        # QA 결과를 포맷팅
        formatted_results = []
        for q in questions:
            key = q["key"]
            desc = q["description"]
            answer = qa_results.get(key, "정보 없음")
            formatted_results.append(f"### {desc}\n{answer}")

        combined = "\n\n".join(formatted_results)
        user_content = f"[문서명: {file_name}]\n\n{combined}\n\n위 분석 결과를 하나의 요약으로 통합해줘."

        return {
            "model": settings.VLLM_MODEL,
            "messages": [
                {"role": "system", "content": self.merge_system_prompt},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
            "temperature": settings.DEFAULT_TEMPERATURE,
        }

# 싱글톤 인스턴스
document_summary_prompt_builder = DocumentSummaryPromptBuilder()
