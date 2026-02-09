# LLM Gateway

**LLM Gateway**는 FastAPI 기반의 AI 백엔드 서버로, LLM(Large Language Model)과 RAG(Retrieval-Augmented Generation)를 통합하여 챗봇 및 문서 자동화 서비스를 제공합니다.

## 주요 기능 (Key Features)

*   **AI 챗봇 (Agentic RAG Chatbot)**
    *   LangGraph 기반 에이전트가 질문 분석, 검색, 답변 생성을 자율적으로 수행
    *   Human-in-the-loop: 질문이 불명확할 경우 사용자에게 명확화 요청 후 재개
    *   SSE(Server-Sent Events) 토큰 스트리밍으로 실시간 답변 전송
*   **하이브리드 RAG (Hybrid RAG)**
    *   Dense + Sparse(BM25) 벡터 검색을 Weighted Sum 방식으로 결합
    *   Qdrant 벡터 데이터베이스 기반
    *   Reranker를 통한 최종 순위 재조정
*   **문서 처리 (Document Processing)**
    *   PDF, HWP, Office 문서 업로드 및 자동 인덱싱
    *   Polaris / MarkItDown / pdf4llm 기반 문서 파싱
    *   Open Search (전체 문서) / Private Search (특정 파일) 모드 지원
*   **통화 요약 (Call Summary)**: STT 결과물 분석 및 통화 내용 요약
*   **문서 자동화 (Document Automation)**: Jinja2 기반 HWPX 문서 템플릿 자동 생성
*   **대화 변환 (Dialogue Converter)**: CSV 등 대화 로그 처리 및 변환

## 기술 스택 (Tech Stack)

*   **Language**: Python 3.12+
*   **Framework**: FastAPI, Uvicorn
*   **AI/Agent**: LangGraph (에이전트 워크플로우), LangChain Core
*   **LLM**: vLLM 호환 서버 (OpenAI API 형식)
*   **Database**
    *   **Redis**: 대화 히스토리, 캐싱
    *   **Qdrant**: 벡터 DB (Dense + Sparse 하이브리드 검색)
*   **Document**: MarkItDown, pdf4llm, Polaris
*   **Infrastructure**: Docker

## 아키텍처 (Architecture)

### 전체 시스템 구성

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser)                         │
│                    SSE EventSource 연결                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST + SSE Stream
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Gateway (FastAPI)                        │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ routes_     │  │ routes_      │  │ routes_document_       │ │
│  │ chatbot.py  │  │ callsummary  │  │ automation.py          │ │
│  └──────┬──────┘  └──────────────┘  └────────────────────────┘ │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │         SSEGraphAdapter (sse_adapter.py)     │                │
│  │  - 그래프 실행 → SSE 이벤트 변환             │                │
│  │  - 토큰 스트리밍 (vLLM SSE → Client SSE)    │                │
│  │  - Human-in-the-loop 세션 관리               │                │
│  └──────┬──────────────────────────┬────────────┘                │
│         │ graph.astream()          │ stream_llm_tokens()        │
│         ▼                          ▼                            │
│  ┌─────────────────┐    ┌───────────────────┐                   │
│  │ LangGraph Agent │    │   LLM Client      │                   │
│  │  (graph.py)     │    │  (llm_client.py)  │                   │
│  └─────────────────┘    └─────────┬─────────┘                   │
│                                   │                             │
└───────────────────────────────────┼─────────────────────────────┘
                                    │ httpx (streaming)
          ┌─────────────────────────┼──────────────────┐
          ▼                         ▼                  ▼
  ┌──────────────┐   ┌───────────────────┐  ┌──────────────────┐
  │    Redis     │   │   vLLM Server     │  │     Qdrant       │
  │  - History   │   │   (LLM 추론)      │  │  (Vector DB)     │
  │  - Cache     │   │   OpenAI API 호환  │  │  - Dense 검색    │
  └──────────────┘   │   토큰 스트리밍    │  │  - Sparse 검색   │
                     └───────────────────┘  └──────────────────┘
```

### LangGraph 에이전트 워크플로우

```
START
  │
  ▼
┌──────────────────┐
│  summarize_node  │  Redis에서 최근 대화 히스토리를 가져와 요약
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ analyze_rewrite_node │  요약 + 현재 질문을 분석
│                      │  → 질문이 명확한가? (is_clear)
│                      │  → 서브질문으로 분해 (rewritten_questions)
└────────┬─────────────┘
         │
         ├── is_clear=True ──────────────────────────┐
         │                                           │
         ▼                                           │
┌──────────────────┐                                 │
│ human_input_node │  interrupt_before로 그래프 정지  │
│                  │  → SSE: clarification_needed 전송│
│                  │  → 사용자 응답 대기               │
└────────┬─────────┘                                 │
         │                                           │
         │ (사용자 응답 후 그래프 재개)                 │
         │                                           │
         ├───────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│ process_question_node│  서브질문별 하이브리드 검색 + 답변 생성
│                      │
│  ┌─ Qdrant 하이브리드 검색 ─┐
│  │  Dense + Sparse(BM25)    │
│  │  Weighted Sum 결합       │
│  │  Reranker 재순위         │
│  └──────────────────────────┘
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  aggregate_node  │  답변 통합 + streaming_payload 준비
│                  │  → SSE adapter가 vLLM 스트리밍으로 전송
└────────┬─────────┘
         │
         ▼
        END
```

### 문서 인제스트 파이프라인

```
파일 업로드 (PDF/HWP/Office)
  │
  ▼
┌────────────────────────┐
│  rag_ingestion_service │
│                        │
│  확장자 판별            │
│  ├─ .pdf → Polaris 또는 pdf4llm으로 마크다운 변환
│  ├─ .hwp → PDF 변환 후 처리
│  └─ 기타 → MarkItDown으로 마크다운 변환
│                        │
│  청크 분할 (CHUNK_SIZE=1000, OVERLAP=200)
│  Dense 임베딩 + Sparse(BM25) 벡터 생성
│  Qdrant에 저장
└────────────────────────┘
```

## 설치 및 실행 (Installation & Run)

### 사전 요구 사항 (Prerequisites)
*   Docker & Docker Compose
*   vLLM 호환 LLM 서버 (외부)
*   임베딩/Reranker 모델 서버 (외부)

### Docker Compose로 실행 (권장)

```bash
# 환경 변수 설정 (선택)
export MODEL_SERVER_URL=http://your-model-server:21440

# 전체 스택 실행 (앱 + Redis + Qdrant)
docker-compose up -d --build

# 로그 확인
docker-compose logs -f app
```

| 서비스 | 컨테이너 | 포트 | 설명 |
|--------|----------|------|------|
| `app` | `llm_gateway` | 8000 | FastAPI 애플리케이션 |
| `redis` | `llm_gateway_redis` | 6379 | Redis (대화 히스토리) |
| `qdrant` | `llm_gateway_qdrant` | 6333, 6334 | Qdrant (벡터 DB) |

### Docker 단독 실행

```bash
# 이미지 빌드
docker build -t llm-gateway .

# 컨테이너 실행 (Redis, Qdrant 별도 필요)
docker run -d \
  -p 8000:8000 \
  -e MODEL_SERVER_URL=http://your-model-server:21440 \
  -e REDIS_HOST=your-redis-host \
  -e QDRANT_HOST=your-qdrant-host \
  llm-gateway
```

### 환경 변수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `MODEL_SERVER_URL` | LLM/임베딩/Reranker 서버 | `http://localhost:21440` |
| `VLLM_MODEL` | 사용 모델 이름 | `LGAI-EXAONE/EXAONE-4.0-32B-AWQ` |
| `REDIS_HOST` / `REDIS_PORT` | Redis 연결 | `localhost:6379` |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant 연결 | `localhost:6333` |
| `QDRANT_COLLECTION` | Qdrant 컬렉션 이름 | `documents` |

### 로컬 개발 환경

```bash
# 가상 환경 생성 및 의존성 설치
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

## 프로젝트 구조 (Project Structure)

```
├── app/
│   ├── main.py                    # FastAPI 엔트리포인트
│   ├── api/                       # API 라우트
│   │   ├── routes_chatbot.py              # 챗봇 (Open/Private/Continue)
│   │   ├── routes_callsummary.py          # 통화 요약
│   │   ├── routes_dialogue_converter.py   # 대화 변환
│   │   ├── routes_document_automation.py  # 문서 자동화
│   │   ├── routes_admin.py                # 관리자 API
│   │   └── schemas/                       # Pydantic 스키마
│   ├── core/                      # 설정
│   │   └── config.py
│   └── services/
│       ├── api_clients/           # 외부 서버 클라이언트
│       │   ├── llm_client.py              # LLM 서버 (vLLM)
│       │   ├── stt_client.py              # STT 서버
│       │   ├── model_server_client.py     # 임베딩/Reranker 서버
│       │   └── polaris_client.py          # Polaris 문서 파싱
│       ├── chat_agent/            # LangGraph 에이전트
│       │   ├── graph.py               # 그래프 정의
│       │   ├── graph_state.py         # 상태 클래스
│       │   ├── nodes.py               # 노드 함수
│       │   ├── edges.py               # 조건부 엣지
│       │   ├── prompts.py             # 프롬프트 템플릿
│       │   ├── tools.py               # 검색 도구
│       │   └── sse_adapter.py         # SSE 스트리밍 어댑터
│       ├── rag/                   # RAG 서비스
│       │   ├── qdrant_service.py          # Qdrant 하이브리드 검색
│       │   ├── rag_ingestion_service.py   # 문서 인제스트
│       │   └── sparse_encoder.py          # BM25 Sparse 인코더
│       ├── admin/                 # 관리자 서비스
│       ├── documents/             # 문서 자동화
│       ├── prompt_builders/       # 프롬프트 빌더
│       └── utils/                 # 유틸리티
├── uploaded_files/                # 사용자 업로드 파일
├── requirements.txt
├── Dockerfile
└── README.md
```

## API 문서

서버 실행 후 브라우저에서 아래 주소로 API 명세를 확인할 수 있습니다.
*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`
