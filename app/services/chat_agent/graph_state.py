"""
LangGraph State 클래스 정의
"""

from typing import List, Optional, Annotated
from typing_extensions import TypedDict


def accumulate_or_reset(left: List[dict], right: List[dict]) -> List[dict]:
    """에이전트 답변을 누적하는 리듀서 함수"""
    if not right:
        return []
    return left + right


class MainState(TypedDict, total=False):
    """메인 그래프 상태"""
    invoke_id: str
    original_query: str
    agent_answers: Annotated[List[dict], accumulate_or_reset]
    filter_filename: Optional[str]  # 특정 파일 검색 필터
    translate_to: Optional[str]  # 번역 언어 코드 (en/zh/ja), None이면 번역 없음
    streaming_payload: Optional[dict]  # SSE adapter에서 스트리밍 생성에 사용
    # 할루시네이션 검증
    verification_passed: Optional[bool]  # None=미검증, True=통과, False=실패
    retry_count: int  # 검증 실패 후 재시도 횟수

