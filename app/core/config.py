from pydantic import BaseModel
import os

class Settings(BaseModel):
    # 통합 Model Server URL (LLM + STT + Embedding)
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://213.173.111.112:32651")
    VLLM_MODEL: str = os.getenv("VLLM_MODEL", "LGAI-EXAONE/EXAONE-4.0-32B-AWQ")

    # LLM 기본 설정
    DEFAULT_MAX_TOKENS: int = int(os.getenv("DEFAULT_MAX_TOKENS", "4096"))
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0"))

    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_URL: str = os.getenv("REDIS_URL", f"redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}/0")

    # 대화 내용을 며칠 뒤에 자동 삭제할지 (7일)
    CHAT_HISTORY_TTL: int = 60 * 60 * 24 * 7

    # [파일 업로드 경로] 프로젝트 루트에 'uploaded_files' 라는 폴더를 기본값으로 사용
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploaded_files")


    # [검색 설정]
    SEARCH_TOP_K: int = 6  # 검색 상위 개수

    # [청크 설정] - 2025 연구 기반 최적화 (400-512 토큰 ≈ 800-1500자)
    CHUNK_SIZE: int = 1000      # 1000 (약 300-400 토큰)
    CHUNK_OVERLAP: int = 200    # 250 → 200 (20%)
    CONTEXT_EXPAND_SIZE: int = 800  # 인접 청크에서 가져올 추가 컨텍스트 크기

    # [Rerank 설정]
    RERANK_SCORE_THRESHOLD: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.55"))

    # [Polaris 설정] Polaris 사용 여부 (False일 경우 pdf4llm/markitdown 등 대체재 사용)
    POLARIS_ENABLED: bool = False

    # [Qdrant 설정]
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")
    EMBEDDING_DIM: int = 1024

    # [글로벌 문서 설정] 관리자가 올린 공용 문서의 invoke_id
    GLOBAL_INVOKE_ID: str = os.getenv("GLOBAL_INVOKE_ID", "__global__")

    # [타임아웃 설정] 각 클라이언트 요청 타임아웃 (초)
    LLM_TIMEOUT: float = float(os.getenv("LLM_TIMEOUT", "180"))
    STT_TIMEOUT: float = float(os.getenv("STT_TIMEOUT", "300"))
    MODEL_SERVER_EMBED_TIMEOUT: float = float(os.getenv("MODEL_SERVER_EMBED_TIMEOUT", "600"))  # 문서 임베딩 (대량)
    MODEL_SERVER_QUERY_TIMEOUT: float = float(os.getenv("MODEL_SERVER_QUERY_TIMEOUT", "30"))  # 쿼리 임베딩 (소량)

settings = Settings()