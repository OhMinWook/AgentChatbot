## 여기서는 LLM server와 통신을 하는 곳입니다. LLM server에게 직접 메세지를 보내는 소통 창구입니다.
import logging
import httpx
from typing import AsyncGenerator
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        self.base_url = settings.MODEL_SERVER_URL
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
            # 동시 연결 제한 늘리기 (기본값: 100)
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            self._client = httpx.AsyncClient(timeout=settings.LLM_TIMEOUT, limits=limits)
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

    @staticmethod
    def extract_content(response: dict) -> str:
        """LLM 응답에서 content를 안전하게 추출"""
        try:
            content = response["choices"][0]["message"]["content"]
            if content is None:
                logger.warning(f"LLM 응답 content가 None: {response}")
                return ""
            return content
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"LLM 응답 파싱 실패: {e}, response={response}")
            raise ValueError(f"LLM 응답 형식 오류: {e}")

    async def chat_completions(self, payload: dict, max_retries: int = 1) -> dict:
        """
        일반 대화 요청 (Non-streaming)
        :param payload: LLM 요청 payload
        :param max_retries: 실패 시 재시도 횟수 (기본 1회)
        :return: 파싱된 JSON dict (Response 객체 아님)
        """
        import time
        url = f"{self.base_url}/v1/chat/completions"
        last_error = None

        logger.info(f"[LLM Client] 요청 시작 → {url}")
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                response = await self.client.post(url, json=payload, headers=self.headers)
                response.raise_for_status()
                duration = time.time() - start_time
                logger.info(f"[LLM Client] 응답 완료 ({duration:.2f}s)")
                return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"LLM Server Error (attempt {attempt + 1}/{max_retries + 1}): {e.response.text}")
                if attempt < max_retries:
                    logger.info("Retrying LLM request...")
                    continue
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"LLM Connection Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    logger.info("Retrying LLM request...")
                    continue

        # 모든 재시도 실패
        logger.error(f"LLM request failed after {max_retries + 1} attempts")
        raise last_error

    async def chat_completions_stream(self, payload: dict, max_retries: int = 1) -> AsyncGenerator[bytes, None]:
        """
        vLLM 스트리밍 요청 (SSE)
        :param payload: LLM 요청 payload
        :param max_retries: 연결 실패 시 재시도 횟수 (기본 1회)
        :return: 바이트 스트림 제너레이터
        """
        url = f"{self.base_url}/v1/chat/completions"

        # 스트리밍을 켜달라는 옵션을 강제로 주입 (실수 방지)
        payload["stream"] = True

        last_error = None
        r = None

        # 연결 단계에서만 재시도
        for attempt in range(max_retries + 1):
            try:
                req = self.stream_client.build_request("POST", url, json=payload, headers=self.headers)
                r = await self.stream_client.send(req, stream=True)
                r.raise_for_status()
                break  # 연결 성공

            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"LLM Stream Server Error (attempt {attempt + 1}/{max_retries + 1}): {e.response.text}")
                if attempt < max_retries:
                    logger.info("Retrying LLM stream request...")
                    continue
            except httpx.RequestError as e:
                last_error = e
                logger.warning(f"LLM Stream Connection Error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                if attempt < max_retries:
                    logger.info("Retrying LLM stream request...")
                    continue

        if r is None:
            logger.error(f"LLM stream request failed after {max_retries + 1} attempts")
            raise last_error

        # 내부 함수 정의 (Closure)
        async def gen():
            try:
                # aiter_bytes()는 Raw Byte를 줌.
                # SSE 포맷(data: ...)을 그대로 프론트로 토스하기엔 이게 제일 효율적임.
                async for chunk in r.aiter_bytes():
                    yield chunk
            except Exception as stream_err:
                logger.warning(f"Streaming interrupted: {stream_err}")
                raise stream_err
            finally:
                # 스트리밍이 끝나거나 중간에 끊겨도 응답만 닫음 (클라이언트는 재사용)
                await r.aclose()

        return gen()


# 싱글톤처럼 쓰기 위해 인스턴스 생성
llm_client = LLMClient()
