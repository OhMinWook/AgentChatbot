from pydantic import BaseModel
import os

class Settings(BaseModel):
    VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://198.13.252.3:58524")
    VLLM_MODEL: str = os.getenv("VLLM_MODEL", "LGAI-EXAONE/EXAONE-4.0-32B-AWQ")

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
    MAX_HISTORY_COUNT: int = 5  # 기존 개수 제한
    MAX_HISTORY_CHARS: int = 2500

    # [파일 업로드 경로] 프로젝트 루트에 'uploaded_files' 라는 폴더를 기본값으로 사용
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploaded_files")

    # [STT 설정] 음성인식 서버 URL
    STT_BASE_URL: str = os.getenv("STT_BASE_URL", "http://64.247.196.119:13850")

    # [모델 서버 설정] 임베딩, Reranker 등을 위한 외부 모델 서버 URL
    MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://198.13.252.5:10973")

    # [RAG 설정] 임베딩 차원 및 인덱스 이름
    EMBEDDING_DIMS: int = 1024
    RAG_INDEX_NAME: str = "otinus_knowledge_index"

    # [ColBERT 설정] Jina ColBERT v2 + Voyager 로컬 인덱스
    COLBERT_TOP_K: int = 6  # ColBERT 검색 상위 개수

    # [Polaris 설정] Polaris 사용 여부 (False일 경우 pdf4llm/markitdown 등 대체재 사용)
    POLARIS_ENABLED: bool = False

    # [키워드 추출 서버 설정] 4B LLM 서버 (없으면 MODEL_SERVER_URL 사용)
    KEYWORD_SERVER_URL: str = os.getenv("KEYWORD_SERVER_URL", "")

    # [Neo4j 설정] LightRAG용 Graph DB
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME: str = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")  # 초기 비밀번호 (변경 필요시 수정)

settings = Settings()