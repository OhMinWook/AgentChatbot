# LLM Gateway

**LLM Gateway**는 FastAPI 기반의 AI 백엔드 서버로, LLM(Large Language Model), RAG(Retrieval-Augmented Generation), 그리고 Knowledge Graph 기술을 통합하여 고도화된 챗봇 및 문서 자동화 서비스를 제공합니다.

## 🚀 주요 기능 (Key Features)

*   **AI 챗봇 (Chatbot)**: 문맥을 이해하고 자연스러운 대화를 수행하는 LLM 기반 챗봇 인터페이스
*   **하이브리드 RAG (Hybrid RAG)**:
    *   **LightRAG**: Neo4j 그래프 데이터베이스를 활용하여 엔티티와 관계 기반의 심층적인 문맥 검색 제공
    *   **Vector Search**: ColBERT 및 Voyager 인덱스를 활용한 고성능 의미 기반 검색
*   **AI 에이전트 (LangGraph)**: 단순 답변 생성을 넘어 요약, 분석, 재작성 등의 복잡한 작업을 수행하는 워크플로우 엔진
*   **통화 요약 (Call Summary)**: STT(Speech-to-Text) 결과물을 분석하여 통화 내용 요약 및 인사이트 도출
*   **문서 자동화 (Document Automation)**: 데이터 기반 HWPX 등 문서 템플릿 자동 생성 및 처리
*   **대화 변환 (Dialogue Converter)**: 다양한 형식의 대화 로그 처리 및 변환

## 🛠 기술 스택 (Tech Stack)

*   **Language**: Python 3.10+
*   **Framework**: FastAPI, Uvicorn
*   **AI Framework**: LangChain, LangGraph, LlamaIndex (partial), LightRAG
*   **Database**:
    *   **Redis**: 캐싱, 세션 관리 및 Vector Store
    *   **Neo4j**: 지식 그래프(Knowledge Graph) 저장소
*   **Search**: Voyager, ColBERT
*   **Infrastructure**: Docker, Docker Compose

## ⚙️ 설치 및 실행 (Installation & Run)

### 사전 요구 사항 (Prerequisites)
*   Python 3.10 이상
*   Redis Server
*   Neo4j Server
*   외부 LLM/Embedding 모델 서버 (설정 필요)

### 로컬 개발 환경 설정 (Local Setup)

1.  **저장소 클론 (Clone Repository)**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **가상 환경 생성 및 의존성 설치**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate

    pip install -r requirement.txt
    ```

3.  **환경 변수 설정 (Environment Variables)**
    `.env` 파일을 생성하거나 시스템 환경 변수를 설정해야 합니다. 주요 설정값은 `app/core/config.py`를 참고하세요.
    *   `VLLM_BASE_URL`: LLM 모델 서버 주소
    *   `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`: Neo4j 연결 정보
    *   `REDIS_HOST`, `REDIS_PORT`: Redis 연결 정보
    *   `STT_BASE_URL`: STT 서버 주소

4.  **서버 실행**
    ```bash
    uvicorn app.main:app --reload
    ```

### Docker 실행
프로젝트 루트에 포함된 `docker-compose.yaml`을 사용하여 서비스를 실행할 수 있습니다.
```bash
docker-compose up -d --build
```

## 📂 프로젝트 구조 (Project Structure)

```
├── app/
│   ├── api/            # API 라우트 (Endpoint) 정의
│   ├── core/           # 설정(Config), 보안, 로깅 등 공통 모듈
│   ├── schemas/        # Pydantic 데이터 모델 (DTO)
│   ├── services/       # 비즈니스 로직 (RAG, Chat, Agent 등)
│   │   ├── clients/    # 외부 API 클라이언트 (LLM, STT 등)
│   │   ├── rag/        # 검색 증강 생성 로직 (LightRAG 등)
│   │   ├── rag_agent/  # LangGraph 기반 에이전트 워크플로우
│   │   └── documents/  # 문서 처리 로직
│   └── templates/      # 문서 템플릿 파일
├── colbert_indexes/    # RAG 검색 인덱스 데이터
├── uploaded_files/     # 사용자 업로드 파일 저장소
├── requirement.txt     # 파이썬 의존성 목록
└── docker-compose.yaml # Docker 배포 설정
```

## 📚 API 문서
서버 실행 후 브라우저에서 아래 주소로 접속하여 API 명세를 확인할 수 있습니다.
*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`
