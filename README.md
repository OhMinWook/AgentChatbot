# LLM Gateway

**LLM Gateway**는 FastAPI 기반의 AI 백엔드 서버로, LLM(Large Language Model), RAG(Retrieval-Augmented Generation), 그리고 Knowledge Graph 기술을 통합하여 고도화된 챗봇 및 문서 자동화 서비스를 제공합니다.

## 주요 기능 (Key Features)

*   **AI 챗봇 (Agentic RAG Chatbot)**
    *   LangGraph 기반 에이전트가 질문 분석, 검색, 답변 생성을 자율적으로 수행
    *   Human-in-the-loop: 질문이 불명확할 경우 사용자에게 명확화 요청 후 재개
    *   SSE(Server-Sent Events) 토큰 스트리밍으로 실시간 답변 전송
*   **하이브리드 RAG (Hybrid RAG)**
    *   **LightRAG**: Neo4j 그래프 데이터베이스를 활용한 엔티티/관계 기반 심층 문맥 검색
    *   **ColBERT + Voyager**: 로컬 인덱스 기반 고성능 의미 검색
    *   두 검색 결과를 교차 검증하여 답변 품질 향상
*   **문서 처리 (Document Processing)**
    *   PDF, HWP, Office 문서 업로드 및 자동 인덱싱
    *   Polaris / MarkItDown / pdf4llm 기반 문서 파싱
    *   Open Search (전체 문서) / Private Search (특정 파일) 모드 지원
*   **통화 요약 (Call Summary)**: STT 결과물 분석 및 통화 내용 요약
*   **문서 자동화 (Document Automation)**: Jinja2 기반 HWPX 문서 템플릿 자동 생성
*   **대화 변환 (Dialogue Converter)**: CSV 등 대화 로그 처리 및 변환

## 기술 스택 (Tech Stack)

*   **Language**: Python 3.10+
*   **Framework**: FastAPI, Uvicorn
*   **AI/Agent**: LangGraph (에이전트 워크플로우), LangChain Core
*   **LLM**: vLLM 호환 서버 (OpenAI API 형식)
*   **Database**
    *   **Redis**: 대화 히스토리, 청크 저장소, 캐싱
    *   **Neo4j**: Knowledge Graph (LightRAG)
*   **Search**: ColBERT + Voyager (pylate), LightRAG
*   **Document**: MarkItDown, pdf4llm, Polaris
*   **Infrastructure**: Docker, Docker Compose

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
  │    Redis     │   │   vLLM Server     │  │     Neo4j        │
  │  - History   │   │   (LLM 추론)      │  │  (Knowledge      │
  │  - Chunks    │   │   OpenAI API 호환  │  │   Graph)         │
  │  - Cache     │   │   토큰 스트리밍    │  │  - LightRAG      │
  └──────────────┘   └───────────────────┘  └──────────────────┘
```

### LangGraph 에이전트 워크플로우

사용자 질문이 들어오면 아래 그래프가 순차적으로 실행됩니다. 각 노드는 `MainState`를 공유하며, 조건부 엣지가 흐름을 분기합니다.

```
START
  │
  ▼
┌──────────────────┐
│  summarize_node  │  Redis에서 최근 대화 히스토리를 가져와 요약
│                  │  → conversation_summary 생성
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
│ human_input_node │  interrupt_before로 그래프 일시정지  │
│                  │  → SSE: clarification_needed 전송   │
│                  │  → 사용자 응답 대기                  │
└────────┬─────────┘                                 │
         │                                           │
         │ (사용자 응답 후 그래프 재개)                │
         │ awaiting=False → analyze_rewrite로 복귀    │
         │ awaiting=True  → END                      │
         │                                           │
         ├───────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│ process_question_node│  서브질문별 병렬 검색 + 답변 생성
│                      │
│  ┌─ ColBERT 배치검색 ─┐  (Voyager 로컬 인덱스)
│  │  top-K 문서 검색    │
│  └───────────────────┘
│  ┌─ LightRAG 검색 ───┐  (Neo4j 그래프, Open Chat만)
│  │  엔티티/관계 탐색   │
│  └───────────────────┘
│
│  단일 질문: 프롬프트만 저장 (LLM 호출 보류 → 스트리밍 전용)
│  복수 질문: 각각 LLM 호출하여 개별 답변 생성
└────────┬─────────────┘
         │
         ▼
┌──────────────────┐
│  aggregate_node  │  답변 통합 + streaming_payload 준비
│                  │
│  단일 답변: streaming_payload에 프롬프트 저장
│            → SSE adapter가 vLLM 스트리밍으로 실시간 전송
│  복수 답변: 통합 프롬프트 생성
│            → SSE adapter가 vLLM 스트리밍으로 실시간 전송
│  결과 없음: precomputed 텍스트 즉시 전송
└────────┬─────────┘
         │
         ▼
        END → SSEGraphAdapter가 streaming_payload를 읽어
              vLLM에 스트리밍 요청 → 토큰 단위 SSE 전송
```

### SSE 이벤트 흐름

클라이언트는 단일 HTTP 연결로 아래 이벤트를 순서대로 수신합니다.

```
── 그래프 실행 중 ──
data: {"type": "progress", "step": "시작"}
data: {"type": "progress", "step": "대화 맥락 분석 중"}
data: {"type": "progress", "step": "질문 분석 완료 (2개 질문)"}

── 명확화 필요 시 (Human-in-the-loop) ──
data: {"type": "clarification_needed", "message": "...", "thread_id": "xxx"}
   └→ 클라이언트가 /message/{id}/continue로 응답 전송 후 재개

── 답변 전송 ──
data: {"type": "references", "docs": [...]}       ← 참고 문서 목록
data: {"type": "answer", "content": "검색된"}      ← 토큰 스트리밍
data: {"type": "answer", "content": " 문서에"}
data: {"type": "answer", "content": " 따르면"}
...
data: {"type": "done"}                             ← 완료
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
│  ├─ .hwp → win32com으로 PDF 변환 후 pdf4llm 처리
│  └─ 기타 → MarkItDown으로 마크다운 변환
│                        │
│  페이지 단위 청크 분할   │
│  ├─ local_index_service → ColBERT 임베딩 + Voyager 인덱스 구축
│  └─ lightrag_service → Neo4j 그래프에 엔티티/관계 추출 및 저장
└────────────────────────┘
```

## 설치 및 실행 (Installation & Run)

### 사전 요구 사항 (Prerequisites)
*   Docker & Docker Compose
*   vLLM 호환 LLM 서버 (외부)
*   임베딩/Reranker 모델 서버 (외부)

### 빠른 시작 (Docker)

1.  **저장소 클론**
    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2.  **환경 변수 설정**

    `.env` 파일을 생성하고 외부 서버 주소를 설정합니다.

    ```bash
    # .env 예시
    VLLM_BASE_URL=http://10.0.0.5:8080
    VLLM_MODEL=LGAI-EXAONE/EXAONE-4.0-32B-AWQ
    MODEL_SERVER_URL=http://10.0.0.5:8081
    STT_BASE_URL=http://10.0.0.5:8082
    NEO4J_PASSWORD=mypassword
    ```

    | 변수 | 설명 | 기본값 |
    |------|------|--------|
    | `VLLM_BASE_URL` | LLM 서버 주소 | `http://localhost:8000` |
    | `VLLM_MODEL` | 사용 모델 이름 | `LGAI-EXAONE/EXAONE-4.0-32B-AWQ` |
    | `MODEL_SERVER_URL` | 임베딩/Reranker 서버 | - |
    | `REDIS_HOST` / `REDIS_PORT` | Redis 연결 | `localhost:6379` |
    | `NEO4J_URI` | Neo4j 연결 | `bolt://localhost:7687` |
    | `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j 인증 | `neo4j` / `password` |
    | `STT_BASE_URL` | STT 서버 주소 | - |

3.  **Docker Compose로 실행**

    ```bash
    docker-compose up -d --build
    ```

    | 서비스 | 컨테이너 | 포트 | 설명 |
    |--------|----------|------|------|
    | `app` | `llm_gateway` | 8000 | FastAPI 애플리케이션 |
    | `redis` | `otinus_redis` | 6379 | Redis Stack (검색 기능 포함) |
    | `neo4j` | `otinus_neo4j` | 7474, 7687 | Neo4j Community (Browser + Bolt) |

4.  **확인**
    *   API 서버: `http://localhost:8000`
    *   Swagger UI: `http://localhost:8000/docs`
    *   Neo4j Browser: `http://localhost:7474`

### 로컬 개발 환경 (Docker 없이)

Docker 없이 직접 실행하려면 Redis와 Neo4j를 별도로 설치해야 합니다.

```bash
# 가상 환경 생성 및 의존성 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 서버 실행
uvicorn app.main:app --reload
```

## 프로젝트 구조 (Project Structure)

```
├── app/
│   ├── main.py                # FastAPI 엔트리포인트
│   ├── api/                   # API 라우트
│   │   ├── routes_chatbot.py          # 챗봇 (Open/Private/Continue)
│   │   ├── routes_callsummary.py      # 통화 요약
│   │   ├── routes_dialogue_converter.py
│   │   └── routes_document_automation.py
│   ├── core/                  # 설정, 보안, 로깅
│   │   └── config.py
│   ├── schemas/               # Pydantic 데이터 모델
│   ├── services/
│   │   ├── clients/           # 외부 서버 클라이언트 (LLM, STT, Model, Polaris)
│   │   ├── rag/               # RAG 서비스
│   │   │   ├── lightrag_service.py        # Neo4j 기반 그래프 검색
│   │   │   ├── local_index_service.py     # ColBERT + Voyager 로컬 인덱스
│   │   │   └── rag_ingestion_service.py   # 문서 인제스트 파이프라인
│   │   ├── rag_agent/         # LangGraph 에이전트
│   │   │   ├── graph.py           # 그래프 정의
│   │   │   ├── graph_state.py     # 상태 클래스
│   │   │   ├── nodes.py           # 노드 함수 + 스트리밍 LLM
│   │   │   ├── edges.py           # 조건부 엣지
│   │   │   ├── prompts.py         # 프롬프트 템플릿
│   │   │   ├── sse_adapter.py     # SSE 토큰 스트리밍 어댑터
│   │   │   └── tools.py           # 검색 도구
│   │   ├── documents/         # 문서 자동화
│   │   ├── prompt_builders/   # 프롬프트 빌더
│   │   └── utils/             # 메모리, 검증, 다운로드 유틸리티
│   └── templates/             # 문서 템플릿
├── colbert_indexes/           # RAG 검색 인덱스 데이터
├── uploaded_files/            # 사용자 업로드 파일
├── test_sse.html              # SSE 스트리밍 테스트 콘솔
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

## API 문서
서버 실행 후 브라우저에서 아래 주소로 API 명세를 확인할 수 있습니다.
*   **Swagger UI**: `http://localhost:8000/docs`
*   **ReDoc**: `http://localhost:8000/redoc`
