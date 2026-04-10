from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # [외부 인프라 - .env에서 관리]
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "")
    VLLM_MODEL: str = os.getenv("VLLM_MODEL", "")
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_URL: str = os.getenv("REDIS_URL", f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0")
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploaded_files")
    RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.55"))
    RERANK_MIN_RESULTS: int = int(os.getenv("RERANK_MIN_RESULTS", "2"))

    # [LLM 기본 설정]
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0"))

    # [대화 설정]
    CHAT_HISTORY_TTL: int = 60 * 60 * 24 * 7  # 7일

    # [답변 캐시 설정]
    ANSWER_CACHE_TTL: int = int(os.getenv("ANSWER_CACHE_TTL", "3600"))  # 1시간
    ANSWER_CACHE_ENABLED: bool = os.getenv("ANSWER_CACHE_ENABLED", "true").lower() == "true"

    # [검색 설정]
    SEARCH_TOP_K: int = 4

    # [청크 설정 - RAG 문서 분할 (문자 기준)]
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 150
    CONTEXT_EXPAND_SIZE: int = 800

    # [통화 요약 청크 설정 - tiktoken 토큰 기준]
    CALLSUMMARY_CHUNK_THRESHOLD: int = 10000  # 이 토큰 수 이상이면 청크 분할
    CALLSUMMARY_CHUNK_SIZE: int = 6500
    CALLSUMMARY_CHUNK_OVERLAP: int = 500

    # [모델 스펙]
    EMBEDDING_DIM: int = 1024  # Qwen3-Embedding-0.6B

    # [Polaris 설정]
    POLARIS_ENABLED: bool = False

    # [Contextual Retrieval]
    CONTEXTUAL_RETRIEVAL_ENABLED: bool = os.getenv("CONTEXTUAL_RETRIEVAL_ENABLED", "false").lower() == "true"
    CONTEXTUAL_RETRIEVAL_MAX_DOC_CHARS: int = int(os.getenv("CONTEXTUAL_RETRIEVAL_MAX_DOC_CHARS", "8000"))
    CONTEXTUAL_RETRIEVAL_BATCH_SIZE: int = int(os.getenv("CONTEXTUAL_RETRIEVAL_BATCH_SIZE", "32"))

    # [글로벌 문서 설정]
    GLOBAL_INVOKE_ID: str = os.getenv("GLOBAL_INVOKE_ID", "__global__")

    # [DB Agent - MariaDB]
    DB_HOST: str = os.getenv("DB_HOST", "")
    DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
    DB_NAME: str = os.getenv("DB_NAME", "")
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")

    # [Langfuse]
    LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

    # [ffmpeg 경로]
    FFMPEG_PATH: str = os.getenv("FFMPEG_PATH", "ffmpeg")

    # [타임아웃 설정 (초)]
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "180"))
    STT_TIMEOUT: float = float(os.getenv("STT_TIMEOUT", "300"))
    MODEL_SERVER_EMBED_TIMEOUT: float = float(os.getenv("MODEL_SERVER_EMBED_TIMEOUT", "600"))
    MODEL_SERVER_QUERY_TIMEOUT: float = float(os.getenv("MODEL_SERVER_QUERY_TIMEOUT", "30"))
    SSE_QUEUE_TIMEOUT: float = float(os.getenv("SSE_QUEUE_TIMEOUT", "600"))

settings = Settings()
