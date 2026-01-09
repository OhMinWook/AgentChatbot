## 여기서는 LLM server와 통신을 하는 곳입니다. LLM server에게 직접 메세지를 보내는 소통 창구입니다.
import httpx
import json
from typing import AsyncGenerator
from app.core.config import settings


class LLMClient:
    def __init__(self):
        self.base_url = settings.VLLM_BASE_URL
        self.headers = {
            "Content-Type": "application/json",
            # 만약 vLLM에 API 키를 걸었다면 여기에 추가: "Authorization": f"Bearer {KEY}"
        }

    async def chat_completions(self, payload: dict) -> dict:
        """
        일반 대화 요청 (Non-streaming)
        :return: 파싱된 JSON dict (Response 객체 아님)
        """
        url = f"{self.base_url}/v1/chat/completions"

        # timeout은 넉넉하게, 하지만 무한대는 위험함.
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()  # 4xx, 5xx 에러 시 즉시 예외 발생
                return response.json()  # 받는 쪽 편하라고 아예 JSON으로 까서 리턴

            except httpx.HTTPStatusError as e:
                # vLLM이 뱉은 에러 메시지를 로그로 남기거나 확인하기 좋음
                print(f"LLM Server Error: {e.response.text}")
                raise e
            except httpx.RequestError as e:
                print(f"LLM Connection Error: {e}")
                raise e

    async def chat_completions_stream(self, payload: dict) -> AsyncGenerator[bytes, None]:
        """
        vLLM 스트리밍 요청 (SSE)
        :return: 바이트 스트림 제너레이터
        """
        url = f"{self.base_url}/v1/chat/completions"

        # 스트리밍을 켜달라는 옵션을 강제로 주입 (실수 방지)
        payload["stream"] = True

        # 스트리밍은 연결을 오래 유지해야 하므로 timeout을 길게 잡거나 없앰
        client = httpx.AsyncClient(timeout=None)

        try:
            req = client.build_request("POST", url, json=payload, headers=self.headers)
            r = await client.send(req, stream=True)
            r.raise_for_status()

        except Exception as e:
            # 연결 단계에서 실패하면 클라이언트를 닫고 에러를 던짐
            await client.aclose()
            raise e

        # 내부 함수 정의 (Closure)
        async def gen():
            try:
                # aiter_bytes()는 Raw Byte를 줌.
                # SSE 포맷(data: ...)을 그대로 프론트로 토스하기엔 이게 제일 효율적임.
                async for chunk in r.aiter_bytes():
                    yield chunk
            except Exception as stream_err:
                print(f"Streaming interrupted: {stream_err}")
                raise stream_err
            finally:
                # 스트리밍이 끝나거나 중간에 끊겨도 무조건 자원 해제
                await r.aclose()
                await client.aclose()

        return gen()


# 싱글톤처럼 쓰기 위해 인스턴스 생성
llm_client = LLMClient()