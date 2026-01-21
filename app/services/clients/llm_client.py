## 여기서는 LLM server와 통신을 하는 곳입니다. LLM server에게 직접 메세지를 보내는 소통 창구입니다.
import httpx
from typing import AsyncGenerator
from app.core.config import settings


class LLMClient:
    def __init__(self):
        self.base_url = settings.VLLM_BASE_URL
        self.headers = {
            "Content-Type": "application/json",
            # 만약 vLLM에 API 키를 걸었다면 여기에 추가: "Authorization": f"Bearer {KEY}"
        }

        # 클라이언트는 lazy initialization (첫 사용 시 생성)
        self._client: httpx.AsyncClient | None = None
        # 스트리밍용 클라이언트 (timeout 없음)
        self._stream_client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """일반 요청용 클라이언트 (lazy initialization)"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=90.0)
        return self._client

    @property
    def stream_client(self) -> httpx.AsyncClient:
        """스트리밍 요청용 클라이언트 (timeout 없음, lazy initialization)"""
        if self._stream_client is None or self._stream_client.is_closed:
            self._stream_client = httpx.AsyncClient(timeout=None)
        return self._stream_client

    async def close(self):
        """클라이언트 정리"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
        if self._stream_client is not None and not self._stream_client.is_closed:
            await self._stream_client.aclose()
            self._stream_client = None

    async def chat_completions(self, payload: dict) -> dict:
        """
        일반 대화 요청 (Non-streaming)
        :return: 파싱된 JSON dict (Response 객체 아님)
        """
        url = f"{self.base_url}/v1/chat/completions"

        try:
            response = await self.client.post(url, json=payload, headers=self.headers)
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

        try:
            req = self.stream_client.build_request("POST", url, json=payload, headers=self.headers)
            r = await self.stream_client.send(req, stream=True)
            r.raise_for_status()

        except Exception as e:
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
                # 스트리밍이 끝나거나 중간에 끊겨도 응답만 닫음 (클라이언트는 재사용)
                await r.aclose()

        return gen()


# 싱글톤처럼 쓰기 위해 인스턴스 생성
llm_client = LLMClient()
