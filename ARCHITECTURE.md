# LLM Gateway 아키텍처 문서

> 이 문서는 **설계 원리와 데이터 흐름**에 초점을 맞춘 인수인계 문서입니다.
> API 엔드포인트 목록, 설정값 레퍼런스 등은 [CLAUDE.md](./CLAUDE.md)를 참조하세요.

---

## 1. 시스템 개요

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        클라이언트 (웹 브라우저)                    │
│                     SSE (Server-Sent Events)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   LLM Gateway   │ ← FastAPI (uvicorn :8000)
                    │   (이 서버)      │
                    └──┬─────┬─────┬──┘
                       │     │     │
          ┌────────────┘     │     └────────────┐
          ▼                  ▼                  ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────────┐
│   Model Server  │ │    Qdrant    │ │      Redis       │
│ (외부, GPU 서버) │ │ (벡터 DB)    │ │  (캐시/세션)      │
│                 │ │ :6333 REST   │ │  :6379           │
│ /embed          │ │ :6334 gRPC   │ │                  │
│ /rerank         │ │              │ │ - 대화 히스토리    │
│ /v1/chat/compl. │ │ Dense+Sparse │ │ - 다운로드 토큰    │
│ /stt            │ │ Hybrid Search│ │ - TTL 7일         │
└─────────────────┘ └──────────────┘ └──────────────────┘
```

### 각 컴포넌트 역할

| 컴포넌트 | 역할 | 비고 |
|----------|------|------|
| **LLM Gateway** | API 라우팅, LangGraph 에이전트 오케스트레이션, SSE 스트리밍 | Docker 컨테이너 (Python 3.12) |
| **Model Server** | LLM 추론, 임베딩 생성, 리랭킹, STT | GPU 서버에서 독립 운영 |
| **Qdrant** | 벡터 저장 및 하이브리드 검색 (Dense + Sparse) | Docker 컨테이너 |
| **Redis** | 대화 히스토리 캐싱, 다운로드 토큰 관리 | Docker 컨테이너 (Alpine) |

**코드 위치**: `docker-compose.yaml`, `app/main.py`

---

## 2. 사용 모델 및 선정 이유

| 모델 | 용도 | 핵심 스펙 | 선정 이유 |
|------|------|----------|-----------|
| **EXAONE-4.0-32B-AWQ** | LLM 답변 생성 | temperature=0, max_tokens=4096 | LG AI 한국어 특화 LLM, AWQ 양자화로 GPU 메모리 효율화 |
| **KURE-v1** | Dense 임베딩 | 1024차원, query/passage prefix 구분 | 한국어 검색 최적화, query용/passage용 임베딩 구분 지원 |
| **ko-reranker** | Cross-encoder 리랭킹 | sigmoid 0~1 점수, 임계값 0.55 | 한국어 문서 재정렬에 특화된 Cross-encoder |
| **SparseEncoder (자체)** | BM25 Sparse 벡터 | 해시 기반, vocab 100K | 키워드 매칭 보완용, CPU 전용으로 GPU 부담 없음 |
| **Whisper large-v3-turbo** | STT (음성→텍스트) | batch=24, beam=5 | 한국어 음성 인식 품질 우수 |

### 임베딩 모델 입출력

```
[쿼리 임베딩]
  Input:  {"texts": ["검색어"], "is_query": true}
  Output: {"embeddings": [[1024차원 벡터]]}
  → is_query=true이면 모델 서버에서 "query: " prefix 추가

[문서 임베딩]
  Input:  {"texts": ["문서 청크1", ...], "is_query": false}
  Output: {"embeddings": [[1024차원], [1024차원], ...]}
  → is_query=false이면 "passage: " prefix 추가
```

**왜 query/passage를 구분하는가**: KURE-v1은 비대칭 검색 모델로, 짧은 쿼리와 긴 문서 패시지의 임베딩 공간을 다르게 학습했기 때문이다.

**코드 위치**: `app/services/api_clients/model_server_client.py:106-133`

---

## 3. 문서 처리 파이프라인 (Ingestion)

### 3.1 파일 → 텍스트 변환

```
[파일 업로드]
    │
    ├─ PDF → pdf4llm (멀티프로세싱, 페이지 단위 Markdown 변환)
    │         └─ ProcessPoolExecutor (8 workers)
    │
    ├─ HWP/HWPX → Polaris (JAR) → JSON → Markdown
    │              └─ 항상 Polaris 사용 (Polaris 설정과 무관)
    │
    └─ 기타 (DOCX, PPTX 등) → MarkItDown
```

**왜 PDF에 pdf4llm을 쓰는가**: pdf4llm은 페이지 단위 Markdown 변환을 지원하여 페이지 번호 추적이 가능하고, 멀티프로세싱으로 대용량 PDF를 병렬 처리할 수 있다.

**왜 HWP는 항상 Polaris인가**: HWP/HWPX는 한컴 고유 포맷으로, Polaris만이 안정적으로 변환할 수 있다. `settings.POLARIS_ENABLED`와 무관하게 강제 사용된다 (`rag_ingestion_service.py:361-363`).

**코드 위치**: `app/services/rag/rag_ingestion_service.py:49-68` (PDF), `70-84` (Polaris), `459-549` (Legacy 경로)

### 3.2 청킹 전략

#### IncrementalChunker 동작 원리

```
┌──────────────────────────────────────────────┐
│                원본 텍스트                      │
├──────────┬──────────┬──────────┬──────────────┤
│  1000자   │  1000자   │  1000자   │  나머지       │
│ Chunk 0  │ Chunk 1  │ Chunk 2  │ Chunk 3     │
│          │◄─200자─►│          │              │
│          │ overlap  │          │              │
└──────────┴──────────┴──────────┴──────────────┘

각 Chunk는 prev/next 포인터로 연결:
  Chunk 0 ──next──► Chunk 1 ──next──► Chunk 2 ──next──► Chunk 3
  Chunk 0 ◄──prev── Chunk 1 ◄──prev── Chunk 2 ◄──prev── Chunk 3
```

**핵심 파라미터**:
- `CHUNK_SIZE = 1000자` (약 300~400 토큰)
- `CHUNK_OVERLAP = 200자` (20%)
- 문장 경계 자르기: 청크 끝의 70% 이후 마지막 `.` 또는 `\n` 위치에서 자름

**왜 1000자인가**: 2025 연구 기반으로 400~512 토큰이 RAG 검색에 최적이며, 한국어는 1토큰 ≈ 2~3자이므로 1000자가 적절하다 (`config.py:27` 주석 참조).

**왜 prev/next 체인을 만드는가**: 검색 결과가 부족할 때 인접 청크를 따라가서 컨텍스트를 확장하기 위함이다 (3.2절 "컨텍스트 확장" 참조).

**코드 위치**: `app/services/rag/rag_ingestion_service.py:87-188` (IncrementalChunker 클래스)

### 3.3 임베딩 및 저장

```
[청크 리스트]
    │
    ├─ Sparse 임베딩 (CPU, 즉시 완료)
    │   └─ SparseEncoder.encode_batch() → {indices, values}
    │
    ├─ Dense 임베딩 (GPU, 배치 64개씩 순차 호출)
    │   └─ model_server_client.embed_texts(batch, is_query=False)
    │
    └─ Qdrant 저장
        └─ Named Vectors: {"dense": [1024], "sparse": SparseVector}
```

**왜 배치 64개씩인가**: GPU 메모리 보호를 위해 한 번에 64개씩 순차 호출한다. 동시 사용자의 대량 문서 업로드 시 GPU OOM을 방지한다.

**왜 Sparse는 CPU인가**: SparseEncoder는 단순 해시 기반 TF 계산으로, GPU가 필요 없다. 100K 어휘에 대해 해시 충돌 시 값을 합산하며, `re.findall(r'[\w가-힣]+')` 토크나이징으로 2글자 이상 토큰만 유지한다.

**코드 위치**: `app/services/rag/rag_ingestion_service.py:283-335` (배치 처리), `app/services/rag/sparse_encoder.py` (Sparse 인코더)

### 3.4 Admin vs 일반 문서 차이

| 구분 | Admin 문서 | 일반 문서 |
|------|-----------|----------|
| **invoke_id** | `__global__` | 사용자별 ID |
| **저장 경로** | `uploaded_files/admin/{MD5(key)[:16]}/` | `uploaded_files/{invoke_id}/` |
| **Qdrant key 필드** | 있음 (문서 고유키) | 없음 |
| **is_use 필터** | 적용 (False인 문서 검색 제외) | 미적용 |
| **Contextual Embedding** | `[파일: xxx.pdf]` 프리픽스 추가 | 프리픽스 추가 |

**Contextual Embedding**: 리랭커에 전달할 때 `add_source_prefix(content, source)` 함수로 `[파일: xxx.pdf]` 프리픽스를 추가한다. 이를 통해 리랭커가 동일 내용이라도 어떤 문서 출처인지를 참고하여 점수를 매긴다.

**코드 위치**: `app/services/rag/text_utils.py:6-10`, `app/services/chat_agent/tools.py:69-71`

---

## 4. 검색 파이프라인 (Retrieval)

### 4.1 Hybrid Search

```
[쿼리]
    │
    ├─ Dense 임베딩 생성 (is_query=true)
    │   └─ 1024차원 벡터
    │
    ├─ Sparse 임베딩 생성 (BM25 스타일)
    │   └─ {indices, values}
    │
    └─ Qdrant 검색 (각 64개 후보)
        ├─ Dense 검색 (Cosine, 0~1)
        ├─ Sparse 검색 (IDF 보정, 정규화 필요)
        │
        └─ 가중 합계
            Dense 점수 × 0.8 + Sparse 정규화 점수 × 0.2
```

**가중 합계 방식**: Sparse 점수는 범위가 다르므로 Min-Max 정규화 후 합산한다. Dense는 이미 Cosine 유사도(0~1)이므로 정규화 불필요하다.

**왜 8:2인가**: Dense 임베딩(의미적 유사도)이 주 검색 수단이고, Sparse(키워드 매칭)는 Dense가 놓치는 정확한 키워드 매칭을 보완하는 역할이다. 실험적으로 8:2가 한국어 문서 검색에서 가장 좋은 결과를 보였다.

**코드 위치**: `app/services/rag/qdrant_service.py:212-346` (search_hybrid)

### 4.2 Reranking

```
[Qdrant 64개 후보]
    │
    ▼
Cross-encoder Reranking (/rerank)
    │
    ├─ 입력: query + [파일: xxx.pdf] 프리픽스 + content
    ├─ 출력: sigmoid 점수 (0~1)
    │
    ▼
필터링
    ├─ 최소 2개 보장
    ├─ 3번째부터 score < 0.55 제외
    └─ 인접 청크(prev/next) 제외 → 중복 방지
```

**왜 인접 청크를 제외하는가**: 오버랩이 200자이므로, 연속된 청크가 모두 상위에 랭킹될 수 있다. 이 경우 같은 내용이 중복되므로, 선택된 청크의 prev/next를 제외 대상(adjacent_ids)에 등록하여 다양한 문서 영역을 확보한다.

**왜 최소 2개를 보장하는가**: 점수가 낮더라도 최소한의 컨텍스트는 제공해야 LLM이 "관련 정보 없음"을 판단할 수 있다.

**코드 위치**: `app/services/chat_agent/tools.py:29-149`

### 4.3 컨텍스트 확장

검색 결과가 확보되면, 두 단계로 컨텍스트를 확장한다:

#### 단계 1: 결과 ≤ 2개일 때 → 다음 청크 덧붙이기

```
결과가 2개 이하이면 각 청크의 next_chunk_id를 따라 2개 청크를 content에 덧붙인다.

[원본 청크 1000자]──next──>[다음 청크 (overlap 200자 제외)]──next──>[다다음 청크 (overlap 제외)]
```

**왜 2개 이하일 때만**: 결과가 적다는 것은 관련 문서가 한정적이라는 의미이므로, 그 문서의 전후 맥락을 최대한 제공하기 위함이다.

**코드 위치**: `app/services/chat_agent/tools.py:151-216` (_append_following_chunks)

#### 단계 2: 모든 결과 → 앞/뒤 인접 청크 확장

```
     ┌─────800자─────┐┌────1000자────┐┌─────800자─────┐
     │   prev 청크    ││  원본 청크    ││  next 청크     │
     │ (overlap 제외) ││              ││ (overlap 제외) │
     └───────────────┘└──────────────┘└───────────────┘

     최종 청크 크기: ~2600자 (800 + 1000 + 800)
```

- prev 청크: 끝에서 overlap(200자)을 제외한 뒤, 마지막 800자
- next 청크: 앞에서 overlap(200자)을 제외한 뒤, 처음 800자

**왜 오버랩을 제외하는가**: 오버랩 영역은 원본 청크와 겹치므로 제거하여 중복을 방지한다.

**코드 위치**: `app/services/chat_agent/tools.py:218-286` (_expand_with_adjacent_chunks)

### 4.4 Open vs Private 모드

```
                        ┌──── filter_filename ────┐
                        │                         │
                  None (Open)              "파일명" (Private)
                        │                         │
            invoke_id = __global__      invoke_id = 사용자ID
                        │                         │
            Admin 문서 전체 검색         해당 파일만 검색
                        │                         │
            download_url 포함          download_url 미포함
```

**분기 코드** (`nodes.py:184`):
```python
search_invoke_id = invoke_id if filter_filename else settings.GLOBAL_INVOKE_ID
```

Open 모드에서는 Admin이 등록한 글로벌 문서 전체를 검색하며, 답변에 포함된 참조 문서의 다운로드 URL을 생성한다. Private 모드에서는 사용자가 지정한 특정 파일만 검색한다.

---

## 5. LangGraph 에이전트 흐름

### 5.1 그래프 구조

```
                         ┌─────────────────┐
                         │      START      │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                    ┌───►│ analyze_rewrite  │◄──────────────┐
                    │    └────────┬────────┘               │
                    │             │                         │
                    │    ┌───────▼────────┐                │
                    │    │   명확한가?     │                │
                    │    └───┬────────┬───┘                │
                    │        │Yes     │No                   │
                    │        │        │                     │
                    │   ┌────▼───┐  ┌─▼──────────────┐    │
                    │   │process │  │  human_input    │    │
                    │   │question│  │ (interrupt_before)   │
                    │   └────┬───┘  └─┬──────────────┘    │
                    │        │        │                     │
                    │   ┌────▼───┐    │ (사용자 응답 후)     │
                    │   │aggregate│    └─────────────────────┘
                    │   └────┬───┘
                    │        │
                    │   ┌────▼───┐
                    │   │  END   │
                    │   └────────┘
                    │
              (명확화 2회 초과 시 강제 진행)
```

**코드 위치**: `app/services/chat_agent/graph.py:24-69`

### 5.2 각 노드 역할과 설계 의도

#### analyze_rewrite_node

**역할**: 사용자 질문을 분석하여 명확성 판단 + 서브 질문 분해

**입력**: 사용자 메시지 (HumanMessage)
**출력**: `{is_clear, rewritten_questions, clarification_message}`

**설계 의도**:
- vLLM `structured_outputs` (JSON 스키마)를 사용하여 LLM 응답을 구조화한다
- 복잡한 질문("A와 B 비교")을 독립적 검색 단위로 분리하여 각각 최적의 문서를 찾는다
- Private chat에서는 `[문서: 파일명]` 태그를 쿼리에 추가하여 LLM이 문서 컨텍스트를 인식하게 한다

**명확화 제한**: `clarification_count >= 2`이면 불명확하더라도 강제 진행 (`nodes.py:128-130`)

**코드 위치**: `app/services/chat_agent/nodes.py:49-153`

#### human_input_node

**역할**: Human-in-the-loop 중단점

**설계 의도**:
- LangGraph의 `interrupt_before` 메커니즘을 사용하여 그래프 실행을 일시 정지
- SSE를 통해 클라이언트에 `clarification_needed` 이벤트 전송
- 클라이언트 응답 시 `continue_with_sse()`로 그래프 재개
- 이 노드 자체는 빈 딕셔너리를 반환 (상태는 외부에서 `update_state`로 주입)

**코드 위치**: `app/services/chat_agent/nodes.py:156-166`

#### process_question_node

**역할**: 문서 검색 + LLM 답변 생성

**핵심 분기**:

```
[질문 수]
    │
    ├─ 1개 → 메시지만 저장 (LLM 호출 안 함)
    │         → SSE adapter에서 vLLM 실시간 스트리밍
    │         → 첫 토큰 응답 시간(TTFT) 최소화
    │
    └─ 2개+ → 각 질문별 병렬 LLM 호출 (asyncio.gather)
              → 완성된 텍스트를 aggregate에서 통합
```

**왜 단일 질문은 LLM을 호출하지 않는가**: 단일 질문은 aggregate 노드를 거쳐 SSE adapter에서 바로 vLLM 스트리밍을 하면 첫 토큰이 훨씬 빨리 도착한다. 복수 질문은 통합이 필요하므로 먼저 각각 완성된 답변을 만든 뒤 통합한다.

**코드 위치**: `app/services/chat_agent/nodes.py:169-264`

#### aggregate_node

**역할**: streaming_payload 생성 (실제 LLM 호출은 SSE adapter에서 수행)

**출력 분기**:

| 케이스 | streaming_payload | SSE 동작 |
|--------|-------------------|----------|
| 단일 질문 + 메시지 있음 | `{precomputed: false, messages, max_tokens}` | vLLM 실시간 스트리밍 |
| 단일 질문 + 메시지 없음 | `{precomputed: true, content}` | 완성 텍스트 6자씩 전송 |
| 복수 질문 | `{precomputed: false, messages, max_tokens}` | 통합 프롬프트로 vLLM 스트리밍 |

**코드 위치**: `app/services/chat_agent/nodes.py:267-317`

### 5.3 프롬프트 설계

모든 프롬프트는 `PromptPair(system, user)` 데이터클래스로 정의된다.

#### ANALYZE_REWRITE

- **명확성 판단**: "그거", "아까 그것" 등 지시대명사만 있고 대상 특정 불가 → 불명확
- **질문 분리 기준**: 각 서브 질문이 **별도 문서에서 답을 찾아야 할 때** 분리
  - "A와 B 비교" → `["A란?", "B란?"]`
  - "A의 장단점" → `["A의 장단점"]` (같은 문서에서 답변 가능)

#### AGENT (RAG 답변 생성)

```
## 질문
{question}

## 검색된 문서
{context}

## 질문
{question}    ← 질문을 2번 반복
```

**왜 질문을 2번 넣는가**: LLM이 긴 컨텍스트를 처리할 때 "Lost in the Middle" 현상이 발생할 수 있다. 질문을 처음과 끝에 배치하면 LLM의 주의력(attention)이 분산되지 않고 질문에 집중하게 된다.

#### AGGREGATE (복수 답변 통합)

- 중복 정보 제거, 논리적 순서 재구성
- 원본 질문도 2번 반복하여 동일한 주의력 유지 효과

**코드 위치**: `app/services/chat_agent/prompts.py`

---

## 6. SSE 스트리밍 구조

### 이벤트 타입

| 타입 | 설명 | 발생 시점 |
|------|------|----------|
| `progress` | 진행 단계 알림 | 그래프 실행 중 각 단계 |
| `clarification_needed` | 명확화 요청 (Human-in-the-loop) | 질문이 불명확할 때 |
| `references` | 참조 문서 목록 | 답변 전송 전 |
| `rag_documents` | RAG 검색 결과 원본 (디버깅용) | 답변 전송 전 |
| `answer` | LLM 답변 토큰 | 실시간 스트리밍 |
| `done` | 스트리밍 완료 | 마지막 |
| `error` | 에러 발생 | 에러 시 |

### 스트리밍 방식

```
[단일 질문]
  vLLM SSE → 토큰 단위 yield → 클라이언트
  (첫 토큰 빠름, 실시간 타이핑 효과)

[복수 질문 or 검색 실패]
  완성 텍스트 → 6자씩 쪼개서 전송
  (precomputed 모드)
```

**코드 위치**: `app/services/chat_agent/sse_adapter.py:71-86` (_stream_answer_tokens)

### Human-in-the-loop 세션 관리

```
[새 질문] → cancel_pending() → 기존 대기 세션 폐기
    │
    ▼
[분석] → 불명확 → clarification_needed 이벤트 전송
                    │
                    ├─ _pending_threads[invoke_id] = (thread_id, timestamp) 등록
                    │
                    └─ 클라이언트 응답 → continue_with_sse()
                         │
                         ├─ TTL 초과 (600초) → 세션 만료 에러
                         ├─ 다른 요청으로 폐기됨 → 세션 만료 에러
                         └─ 정상 → 그래프 재개 (analyze_rewrite부터)
```

**왜 TTL 600초인가**: 사용자가 명확화 요청을 받고 응답하기까지 합리적인 대기 시간이다. 이 시간이 지나면 LangGraph 체크포인트가 메모리에서 제거된다.

**코드 위치**: `app/services/chat_agent/sse_adapter.py:32-67` (세션 관리), `184-256` (invoke_with_sse), `261-344` (continue_with_sse)

### Open 모드 다운로드 URL 생성

Open 모드에서만 참조 문서에 다운로드 URL을 추가한다:

```
[references 전송 전]
    │
    ├─ source 목록 추출
    ├─ Qdrant에서 메타데이터 조회 (key 유무로 Admin/일반 구분)
    ├─ Admin: uploaded_files/admin/{MD5(key)[:16]}/{source}
    │  일반: uploaded_files/{invoke_id}/{source}
    └─ Redis에 다운로드 토큰 저장 (만료 1시간)
        └─ download_url: /documents/download/{token}
```

**코드 위치**: `app/services/chat_agent/sse_adapter.py:91-156` (파일 경로 해석 + 다운로드 URL)

---

## 7. 모델 서버 동시성 설계

### 아키텍처 결정

Model Server는 GPU를 사용하는 연산(임베딩, 리랭킹)이 CPU 이벤트 루프를 블로킹하지 않도록 설계되어 있다.

```
[FastAPI 이벤트 루프 (메인 스레드)]
    │
    ├─ httpx.AsyncClient ─── HTTP 요청 ──→ [Model Server (GPU 서버)]
    │   (non-blocking I/O)                   ├─ /embed (Lock 보호)
    │                                        ├─ /rerank (Lock 보호)
    │                                        └─ /v1/chat/completions
    │
    ├─ ProcessPoolExecutor (8 workers)
    │   └─ PDF 파싱, Polaris 변환 (CPU 집약적 작업)
    │
    └─ SparseEncoder.encode()
        └─ CPU 전용, 이벤트 루프에서 직접 실행 (가벼움)
```

### HTTP 클라이언트 설계

| 클라이언트 | 용도 | 동시 연결 | 타임아웃 |
|-----------|------|----------|---------|
| `model_server_client` | 임베딩/리랭킹 | 50 (keepalive 20) | 임베딩 600s, 쿼리 30s |
| `llm_client.client` | 일반 LLM 요청 | 500 (keepalive 100) | 180s |
| `llm_client.stream_client` | 스트리밍 LLM | 무제한 | 없음 (timeout=None) |

**왜 스트리밍에 타임아웃이 없는가**: vLLM 스트리밍은 토큰 단위로 계속 데이터가 오므로, 타임아웃이 있으면 긴 답변 생성 중 연결이 끊길 수 있다.

**재시도 전략**: `_request_with_retry`로 일시적 장애(429, 502, 503, 연결 에러)에 대해 최대 3회 시도, 대기 시간은 지수 백오프(1초, 2초, 4초).

**코드 위치**: `app/services/api_clients/model_server_client.py` (임베딩/리랭킹), `app/services/api_clients/llm_client.py` (LLM)

### 배치 처리로 GPU 보호

```
[문서 100개 업로드]
    │
    └─ 64개씩 순차 임베딩 요청
        ├─ Batch 1: 청크 0~63  → GPU 처리
        ├─ Batch 2: 청크 64~99 → GPU 처리
        └─ (동시에 다른 사용자의 쿼리 임베딩도 처리 가능)
```

**왜 64개 배치인가**: Model Server의 GPU 메모리와 처리량의 균형. 너무 크면 OOM, 너무 작으면 오버헤드 증가.

**코드 위치**: `app/services/rag/rag_ingestion_service.py:192, 315-318`

---

## 8. 설정값 레퍼런스

모든 설정값의 목록과 기본값은 [CLAUDE.md의 설정값 참조 섹션](./CLAUDE.md#설정값-참조-configpy)을 참조하세요.

**코드 위치**: `app/core/config.py`
