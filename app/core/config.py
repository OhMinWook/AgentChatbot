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

    # [LLM 기본 설정]
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0"))

    # [대화 설정]
    CHAT_HISTORY_TTL: int = 60 * 60 * 24 * 7  # 7일

    # [검색 설정]
    SEARCH_TOP_K: int = 6

    # [청크 설정]
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    CONTEXT_EXPAND_SIZE: int = 800

    # [모델 스펙]
    EMBEDDING_DIM: int = 1024  # Qwen3-Embedding-0.6B

    # [Polaris 설정]
    POLARIS_ENABLED: bool = False

    # [글로벌 문서 설정]
    GLOBAL_INVOKE_ID: str = os.getenv("GLOBAL_INVOKE_ID", "__global__")

    # [타임아웃 설정 (초)]
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "180"))
    STT_TIMEOUT: float = float(os.getenv("STT_TIMEOUT", "300"))
    MODEL_SERVER_EMBED_TIMEOUT: float = float(os.getenv("MODEL_SERVER_EMBED_TIMEOUT", "600"))
    MODEL_SERVER_QUERY_TIMEOUT: float = float(os.getenv("MODEL_SERVER_QUERY_TIMEOUT", "30"))

settings = Settings()
