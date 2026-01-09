from pydantic import BaseModel
import os

class Settings(BaseModel):
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://198.13.252.3:27478")
    VLLM_MODEL: str = os.getenv("VLLM_MODEL", "QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ")

    # 운영에서 흔히 필요한 제한값들 (MVP 기본)
    MAX_INPUT_CHARS: int = int(os.getenv("MAX_INPUT_CHARS", "20000"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0"))

    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))

    # 대화 내용을 며칠 뒤에 자동 삭제할지 (초 단위, 3일 = 259200초)
    # 이게 있어서 용량 폭발 걱정이 없는 거야!
    CHAT_HISTORY_TTL: int = 60 * 60 * 24 * 7

    # 기억할 대화 턴 수 (질문+답변 한 쌍 기준 10개)
    MAX_HISTORY_COUNT: int = 5

    # [파일 업로드 경로] 프로젝트 루트에 'uploaded_files' 라는 폴더를 기본값으로 사용
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploaded_files")

settings = Settings()