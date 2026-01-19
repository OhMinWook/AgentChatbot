"""
사용 예시 - 이 파일을 수정해서 테스트하세요
"""
import asyncio
from debug_caller import call_ai, call_ai_with_history, call_ai_stream


# 예시 1: 단순 호출
async def test_simple():
    system_prompt = """당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 간결하게 답변해주세요."""

    user_message = "안녕하세요! 오늘 날씨가 어떤가요?"

    response = await call_ai(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.0,
        max_tokens=1024
    )

    print("=" * 50)
    print("[단순 호출 테스트]")
    print(f"System: {system_prompt[:50]}...")
    print(f"User: {user_message}")
    print(f"AI: {response}")
    print("=" * 50)


# 예시 2: 멀티턴 대화
async def test_multi_turn():
    system_prompt = "당신은 수학 선생님입니다. 학생의 질문에 친절하게 답변해주세요."

    messages = [
        {"role": "user", "content": "피타고라스 정리가 뭐에요?"},
        {"role": "assistant", "content": "피타고라스 정리는 직각삼각형에서 빗변의 제곱이 나머지 두 변의 제곱의 합과 같다는 정리입니다. a² + b² = c² 로 표현됩니다."},
        {"role": "user", "content": "실생활에서 어디에 쓰이나요?"},
    ]

    response = await call_ai_with_history(
        system_prompt=system_prompt,
        messages=messages,
        temperature=0.0
    )

    print("=" * 50)
    print("[멀티턴 대화 테스트]")
    print(f"System: {system_prompt}")
    for msg in messages:
        print(f"{msg['role'].upper()}: {msg['content'][:100]}...")
    print(f"AI: {response}")
    print("=" * 50)


# 파일 읽기 헬퍼 함수
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """파일 내용을 읽어서 반환"""
    with open(file_path, "r", encoding=encoding) as f:
        return f.read()


# 예시 3: 커스텀 프롬프트 테스트 (파일 첨부 가능)
async def test_custom():
    # ========== 여기만 수정하세요 ==========

    # 첨부할 파일 경로 (None이면 파일 첨부 안함)
    FILE_PATH = r"C:\Users\user\PycharmProjects\PythonProject1\chat_debug_0325_0326.txt"

    system_prompt = """
당신은 기업의 **전문 회의록 작성자(Professional Scribe)**입니다.
사용자가 제공하는 [업무 대화 기록]을 바탕으로, 주관적인 해석이나 감상 없이 **오직 팩트(Fact)에 기반한 상세 회의록**을 작성하십시오.

**[작성 원칙]**
1. **객관성 유지 (Strictly Objective)**: 대화의 분위기나 의도를 추측하지 말고, 명시적으로 발언된 내용만 기록하십시오.
2. **정보의 누락 방지 (Comprehensive Details)**: 단순히 내용을 압축하지 마십시오. 주요 안건에 대해 누가 어떤 구체적인 근거(숫자, 데이터, 사례 등)를 들어 주장했는지 상세히 기술하십시오.
3. **구조화된 정리**: 시간 순서나 단순 나열보다는, **'안건(Topic)'** 중심으로 내용을 재구성하십시오.

**[출력 양식]**
반드시 아래 Markdown 형식을 따르십시오.

# 📑 상세 업무 회의록

## 1. 개요
* **목적**: [대화의 핵심 목적 1문장 정리]
* **참여자**: [대화에 참여한 인원 목록]

## 2. 안건별 상세 논의 내용
*(대화에서 다뤄진 주제별로 섹션을 나누고, 구체적인 발언 내용을 정리하세요)*

### [안건 1: 주제명]
* **현황/문제점**: [언급된 문제 상황이나 배경 사실]
* **주요 의견**:
  * **[참여자 A]**: [구체적인 주장 및 근거]
  * **[참여자 B]**: [구체적인 주장 및 근거]
* **결정 사항**: [최종 합의된 내용 또는 보류 여부]

### [안건 2: 주제명]
*(위와 동일한 형식으로 반복)*

## 3. 미결정 및 추후 논의 사항
* [결론이 나지 않았거나, 추후 다시 확인하기로 한 팩트들]
"""

    # 파일이 있으면 파일 내용을, 없으면 직접 입력한 메시지를 사용
    if FILE_PATH:
        file_content = read_file(FILE_PATH)
        user_message = f"""
아래 업무 대화 로그를 바탕으로, 위 시스템 프롬프트의 양식에 맞춰 회의록을 작성해 주세요.
모델의 주관적인 생각이나 요약은 배제하고, 대화에 있는 '사실'과 '세부 내용(수치, 조건 등)'을 있는 그대로 상세하게 정리해 주시기 바랍니다.

[채팅 기록 시작]
{file_content}
[채팅 기록 끝]
"""
        print(f"[파일 로드됨] {FILE_PATH} ({len(file_content)} 글자)")
    else:
        user_message = "채팅 기록을 자세히 요약해줄래"

    # ========== 수정 끝 ==========

    print("=" * 50)
    print("[스트리밍 응답]")
    print("=" * 50)

    # 스트리밍으로 실시간 출력 (토큰이 생성될 때마다 바로 출력됨)
    response = await call_ai_stream(
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0,           # 0이면 반복 발생 가능, 0.3~0.7 권장
        max_tokens=1500,
        repetition_penalty=1.1,    # 반복 방지 (1.1~1.2 권장)
    )

    print("=" * 50)


if __name__ == "__main__":
    # 원하는 테스트 함수를 실행하세요
    # asyncio.run(test_simple())
    # asyncio.run(test_multi_turn())
    asyncio.run(test_custom())
