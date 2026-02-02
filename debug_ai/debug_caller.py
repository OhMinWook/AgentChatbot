"""
디버그용 AI 호출기
- 시스템 프롬프트와 사용자 질문을 직접 입력해서 AI 응답을 테스트할 수 있습니다.
"""
import httpx
import asyncio
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings


async def call_ai(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    AI 호출 함수

    Args:
        system_prompt: 시스템 프롬프트
        user_message: 사용자 메시지
        temperature: 창의성 조절 (0.0 ~ 1.0)
        max_tokens: 최대 토큰 수

    Returns:
        AI 응답 텍스트
    """
    url = f"{settings.MODEL_SERVER_URL}/v1/chat/completions"

    payload = {
        "model": settings.VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:  # 10분
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]


async def call_ai_with_history(
    system_prompt: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """
    대화 히스토리를 포함한 AI 호출

    Args:
        system_prompt: 시스템 프롬프트
        messages: [{"role": "user/assistant", "content": "..."}] 형태의 대화 기록
        temperature: 창의성 조절
        max_tokens: 최대 토큰 수

    Returns:
        AI 응답 텍스트
    """
    url = f"{settings.MODEL_SERVER_URL}/v1/chat/completions"

    all_messages = [{"role": "system", "content": system_prompt}] + messages

    payload = {
        "model": settings.VLLM_MODEL,
        "messages": all_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=600.0) as client:  # 10분
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]


async def call_ai_stream(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    repetition_penalty: float = 1.1,
):
    """
    스트리밍 AI 호출 - 토큰이 생성될 때마다 실시간 출력

    Args:
        system_prompt: 시스템 프롬프트
        user_message: 사용자 메시지
        temperature: 창의성 조절 (0.0 ~ 1.0)
        max_tokens: 최대 토큰 수
    """
    import json

    url = f"{settings.MODEL_SERVER_URL}/v1/chat/completions"


    payload = {
        "model": settings.VLLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "repetition_penalty": repetition_penalty,  # 반복 방지 (1.0 = 없음, 1.1~1.2 권장)
    }

    full_response = ""

    async with httpx.AsyncClient(timeout=None) as client:  # 스트리밍은 timeout 없음
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # "data: " 제거
                    if data.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                            full_response += content
                    except json.JSONDecodeError:
                        pass

    print()  # 마지막 줄바꿈
    return full_response


def interactive_mode():
    """대화형 모드로 AI와 대화"""
    print("=" * 60)
    print("디버그용 AI 호출기 - 대화형 모드")
    print("=" * 60)
    print(f"Model: {settings.VLLM_MODEL}")
    print(f"Server: {settings.MODEL_SERVER_URL}")
    print("=" * 60)

    # 시스템 프롬프트 입력
    print("\n[시스템 프롬프트 입력] (빈 줄 입력시 기본값 사용)")
    print("여러 줄 입력 가능, 'END'를 입력하면 완료:")

    system_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        system_lines.append(line)

    system_prompt = "\n".join(system_lines) if system_lines else "You are a helpful assistant."
    print(f"\n[시스템 프롬프트 설정됨] ({len(system_prompt)} 글자)")

    # 대화 루프
    messages = []
    print("\n질문을 입력하세요 ('quit' 입력시 종료, 'reset' 입력시 대화 초기화):\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("종료합니다.")
            break
        elif user_input.lower() == "reset":
            messages = []
            print("[대화 기록 초기화됨]\n")
            continue
        elif not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = asyncio.run(call_ai_with_history(
                system_prompt=system_prompt,
                messages=messages,
                temperature=0.0
            ))
            messages.append({"role": "assistant", "content": response})
            print(f"\nAI: {response}\n")
        except Exception as e:
            print(f"\n[오류 발생] {e}\n")
            messages.pop()  # 실패한 메시지 제거


async def quick_test(system_prompt: str, user_message: str):
    """빠른 테스트용 함수"""
    print("=" * 60)
    print("디버그용 AI 호출기 - 빠른 테스트")
    print("=" * 60)
    print(f"Model: {settings.VLLM_MODEL}")
    print(f"Server: {settings.MODEL_SERVER_URL}")
    print("=" * 60)
    print(f"\n[System Prompt]\n{system_prompt}\n")
    print(f"[User Message]\n{user_message}\n")
    print("=" * 60)
    print("[AI Response]")

    response = await call_ai(system_prompt, user_message)
    print(response)
    print("=" * 60)
    return response


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # 커맨드라인 인자가 있으면 빠른 테스트
        # 사용법: python debug_caller.py "시스템프롬프트" "유저메시지"
        system = sys.argv[1] if len(sys.argv) > 1 else "You are a helpful assistant."
        user = sys.argv[2] if len(sys.argv) > 2 else "Hello!"
        asyncio.run(quick_test(system, user))
    else:
        # 인자 없으면 대화형 모드
        interactive_mode()
