# LLM Gateway

**LLM Gateway**는 FastAPI 기반의 AI 백엔드 서버로, LLM(Large Language Model)과 RAG(Retrieval-Augmented Generation)를 통합하여 챗봇 및 문서 자동화 서비스를 제공합니다.

## 주요 기능 (Key Features)

*   **AI 챗봇 (Agentic RAG Chatbot)**
    *   LangGraph 기반 에이전트가 검색, 답변 생성, 할루시네이션 검증을 자율적으로 수행
    *   할루시네이션 검증 (Hallucination Verification): 숫자·날짜·조건 포함 고위험 답변 자동 검증 및 재시도
    *   SSE(Server-Sent Events) 토큰 스트리밍으로 실시간 답변 전송
    *   답변 캐시: 동일 질문에 대한 중복 LLM 호출 방지
*   **하이브리드 RAG (Hybrid RAG)**
    *   Dense + Sparse(BM25) 벡터 검색을 RRF Fusion 방식으로 결합
    *   Qdrant 벡터 데이터베이스 기반 (후보 64개 → Reranker 재순위)
    *   인접 청크 컨텍스트 확장으로 검색 품질 향상
    *   HyDE(Hypothetical Document Embedding): 가설적 답변으로 검색 쿼리 확장 (선택적 활성화)
    *   Contextual Retrieval: 각 청크에 LLM 생성 맥락 접두어 추가 (선택적 활성화)
*   **문서 처리 (Document Processing)**
    *   PDF, HWP/HWPX, Office 문서 업로드 및 자동 인덱싱
    *   Polaris / MarkItDown / pdf4llm 기반 문서 파싱
    *   문서 유형(구조적/비구조적) 자동 감지 및 청킹 전략 분기
    *   Open Search (전체 문서) / Private Search (특정 파일) 모드 지원
*   **Guardrails (입출력 보안)**
    *   `BlankInputGuard`: 빈 메시지 차단
    *   `ProfanityInputGuard`: 금칙어 입력 차단 (선택적 활성화)
    *   `ProfanityOutputGuard`: 출력 금칙어 치환
*   **통화 요약 (Call Summary)**: STT 결과물 분석 및 통화 내용 요약, ffmpeg 기반 오디오 전처리 (300~3400Hz 필터, 16kHz, 모노)
*   **문서 자동화 (Document Automation)**
    *   Jinja2 기반 HWPX 문서 템플릿 자동 생성 (기안문, 공문, 회의록)
    *   회의 원문 텍스트/파일 → LLM 구조화 → 회의록 HWPX 자동 생성 (SSE 진행률)
    *   토큰 기반 임시 다운로드 링크 (만료시간/1회용 설정 지원)
*   **장애 이력 DB 챗봇 (DB Agent)**: MariaDB 장애 이력 조회 기반 원인 분석 답변
*   **통합 챗봇 (Unified Chat)**: 문서 업로드 여부에 따라 RAG 챗봇 또는 DB 에이전트로 자동 라우팅
*   **대화 변환 (Dialogue Converter)**: CSV 등 대화 로그 처리 및 변환

## 기술 스택 (Tech Stack)

*   **Language**: Python 3.12+
*   **Framework**: FastAPI, Uvicorn
*   **AI/Agent**: LangGraph (에이전트 워크플로우), LangChain Core
*   **LLM**: vLLM 호환 서버 (OpenAI API 형식)
*   **Database**
    *   **Redis**: 대화 히스토리, 캐싱
    *   **Qdrant**: 벡터 DB (Dense + Sparse 하이브리드 검색)
    *   **MariaDB**: 장애 이력 DB (DB 에이전트용)
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
│  ┌──────────────┐  ┌──────────────────────────────────────────┐ │
│  │routes_db_    │  │          routes_unified.py               │ │
│  │chat.py       │  │  (RAG ↔ DB 자동 라우팅)                   │ │
│  └──────────────┘  └──────────────────────────────────────────┘ │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────┐                │
│  │         SSEGraphAdapter (sse_adapter.py)     │                │
│  │  - 그래프 실행 → SSE 이벤트 변환             │                │
│  │  - 토큰 스트리밍 (vLLM SSE → Client SSE)    │                │
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
┌────────────────────────┐
│  process_question_node │  서브질문별 하이브리드 검색 + 답변 생성
│                        │
│  ┌─ Qdrant 하이브리드 검색 ─┐
│  │  Dense + Sparse(BM25)    │
│  │  RRF Fusion 결합          │
│  │  Reranker 재순위          │
│  │  인접 청크 컨텍스트 확장   │
│  └──────────────────────────┘
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│  verify_answer_node    │  할루시네이션 검증 (고위험 질문만)
│                        │  숫자·날짜·조건 포함 답변 LLM 검증
└────────┬───────────────┘
         │
         ├── passed=true  ─────────────────── END
         │
         ├── passed=false + retry 가능 ──→ process_question_node
         │                                  (AGENT_STRICT 프롬프트)
         │
         └── passed=false + 재시도 초과 ─── END (원본 답변 사용)
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
│  ├─ .hwp/.hwpx → Polaris 처리
│  └─ 기타 → MarkItDown으로 마크다운 변환
│                        │
│  문서 유형 자동 감지    │
│  ├─ structured   → 규칙 기반 청킹
│  └─ unstructured → 시맨틱 청킹
│                        │
│  청크 분할 (CHUNK_SIZE=700, OVERLAP=150)
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
| `MODEL_SERVER_URL` | LLM/임베딩/Reranker 서버 URL | - |
| `VLLM_MODEL` | 사용 모델 이름 | - |
| `REDIS_HOST` / `REDIS_PORT` | Redis 연결 | `localhost:6379` |
| `QDRANT_HOST` / `QDRANT_PORT` | Qdrant 연결 | `localhost:6333` |
| `QDRANT_COLLECTION` | Qdrant 컬렉션 이름 | `documents` |
| `DB_HOST` / `DB_PORT` / `DB_NAME` | MariaDB 연결 (DB 에이전트용) | - |
| `DB_USER` / `DB_PASSWORD` | MariaDB 인증 | - |
| `RERANK_SCORE_THRESHOLD` | Reranker 최소 점수 | `0.45` |
| `RERANK_MIN_RESULTS` | Reranker 최소 결과 수 | `2` |
| `HYDE_ENABLED` | HyDE 활성화 여부 | `false` |
| `CONTEXTUAL_RETRIEVAL_ENABLED` | Contextual Retrieval 활성화 | `false` |
| `PROFANITY_INPUT_GUARD_ENABLED` | 금칙어 입력 가드 활성화 | `false` |
| `ANSWER_CACHE_ENABLED` | 답변 캐시 활성화 | `true` |
| `ANSWER_CACHE_TTL` | 답변 캐시 유효 시간 (초) | `3600` |
| `MAX_VERIFY_RETRIES` | 할루시네이션 검증 재시도 횟수 | `1` |
| `SSE_CHUNK_SIZE` | SSE 토큰 스트리밍 청크 크기 | `6` |
| `FFMPEG_PATH` | ffmpeg 실행 경로 (통화 요약용) | `ffmpeg` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | Langfuse 트레이싱 | - |
| `LANGFUSE_HOST` | Langfuse 서버 URL | `http://localhost:3000` |

### 로컬 개발 환경

```bash
# 가상 환경 생성 및 의존성 설치
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

## 주요 엔드포인트 (API Endpoints)

| 엔드포인트 | 메서드 | 설명 |
|---|---|---|
| `/message/open/{invokeId}` | POST | Open 챗봇 (전체 문서 검색, SSE) |
| `/message/private/{invokeId}` | POST | Private 챗봇 (특정 파일 검색, SSE) |
| `/message/{invokeId}/continue` | POST | Human-in-the-loop 재개 |
| `/message/document-summary/{invokeId}` | POST | 문서 요약 (SSE 진행률) |
| `/message/db/{invokeId}` | POST | 장애 이력 DB 챗봇 (SSE) |
| `/message/{invokeId}` | POST | 통합 챗봇 (RAG ↔ DB 자동 라우팅, SSE) |
| `/upload/{invokeId}` | POST | 문서 업로드 및 Qdrant 인덱싱 |
| `/files/{invokeId}` | GET | 업로드 파일 목록 |
| `/history/{invokeId}` | GET | RAG 챗봇 대화 이력 |
| `/history/db/{invokeId}` | GET | DB 챗봇 대화 이력 |
| `/documents/generate-hwpx` | POST | HWPX 문서 자동 생성 |
| `/documents/meeting-minutes/generate-from-text` | POST | 회의 원문 → 회의록 HWPX (SSE) |
| `/documents/download/{token}` | GET | 토큰 기반 파일 다운로드 |
| `/call-summary/{invokeId}` | POST | 통화 요약 (오디오 파일, SSE) |
| `/call-summary-sync/{invokeId}` | POST | 통화 요약 동기 (JSON 응답) |
| `/admin/...` | POST/GET | 관리자 API (문서 관리, 금칙어 등) |

## 프로젝트 구조 (Project Structure)

```
├── app/
│   ├── main.py                    # FastAPI 엔트리포인트
│   ├── api/                       # API 라우트
│   │   ├── routes_chatbot.py              # RAG 챗봇 (Open/Private/Continue/Summary)
│   │   ├── routes_db_chat.py              # 장애 이력 DB 챗봇
│   │   ├── routes_unified.py              # 통합 챗봇 (RAG ↔ DB 자동 라우팅)
│   │   ├── routes_callsummary.py          # 통화 요약
│   │   ├── routes_dialogue_converter.py   # 대화 변환
│   │   ├── routes_document_automation.py  # 문서 자동화 (HWPX, 회의록)
│   │   ├── routes_admin.py                # 관리자 API
│   │   ├── routes_health.py               # 헬스 체크
│   │   └── schemas/                       # Pydantic 스키마
│   ├── core/                      # 설정
│   │   └── config.py
│   └── services/
│       ├── api_clients/           # 외부 서버 클라이언트
│       │   ├── llm_client.py              # LLM 서버 (vLLM)
│       │   ├── stt_client.py              # STT 서버
│       │   ├── model_server_client.py     # 임베딩/Reranker 서버
│       │   └── polaris_client.py          # Polaris 문서 파싱
│       ├── chat_agent/            # LangGraph RAG 에이전트
│       │   ├── graph.py               # 그래프 정의
│       │   ├── graph_state.py         # 상태 클래스
│       │   ├── nodes.py               # 노드 함수
│       │   ├── edges.py               # 조건부 엣지
│       │   ├── prompts.py             # 프롬프트 템플릿
│       │   ├── tools.py               # 하이브리드 검색 도구
│       │   ├── sse_adapter.py         # SSE 스트리밍 어댑터
│       │   └── guardrails_impl.py     # Guardrails (입출력 보안)
│       ├── db_agent/              # DB 에이전트
│       ├── router_agent/          # 통합 라우팅 에이전트
│       ├── rag/                   # RAG 서비스
│       │   ├── qdrant_service.py          # Qdrant 하이브리드 검색
│       │   ├── rag_ingestion_service.py   # 문서 인제스트
│       │   ├── chunker.py                 # 청킹 전략 (구조적/비구조적)
│       │   └── sparse_encoder.py          # BM25 Sparse 인코더
│       ├── admin/                 # 관리자 서비스
│       ├── documents/             # 문서 자동화 (HWPX, 회의록)
│       ├── prompt_builders/       # 프롬프트 빌더
│       └── utils/                 # 유틸리티
│           ├── download_service.py        # 토큰 기반 다운로드 링크
│           └── memory_service.py          # Redis 대화 히스토리
├── uploaded_files/                # 사용자 업로드 파일
├── requirements.txt
├── Dockerfile
└── README.md
```

## API 문서

서버 실행 후 브라우저에서 아래 주소로 API 명세를 확인할 수 있습니다.
*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`
