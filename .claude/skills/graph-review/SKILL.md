---
name: graph-review
description: LangGraph 그래프 흐름을 리뷰한다. 노드/엣지/상태 분석 및 잠재적 이슈를 리포트.
disable-model-invocation: true
allowed-tools: Read
---

Read the following files in parallel:
- `app/services/chat_agent/graph.py`
- `app/services/chat_agent/nodes.py`
- `app/services/chat_agent/graph_state.py`
- `app/services/chat_agent/edges.py`

Then produce a structured review with these sections:

## 1. 흐름 다이어그램
Mermaid flowchart로 노드/엣지/조건 분기를 시각화하라. 조건 분기는 라벨을 붙여라.

## 2. 노드별 요약
각 노드에 대해 한 줄로: 입력 state 필드 → 핵심 동작 → 출력 state 필드

## 3. State 필드 사용 현황
MainState의 각 필드가 어느 노드에서 읽히고 어느 노드에서 쓰이는지 표로 정리하라. 사용되지 않는 필드는 별도 표시.

## 4. 잠재적 이슈
다음 관점에서 문제를 찾아라:
- 엣지 케이스 (빈 state, 빈 응답, 타입 불일치 등)
- 명확화 루프 탈출 조건
- 스트리밍/비스트리밍 분기 일관성
- 미사용 state 필드

각 이슈는 `[심각도: HIGH/MED/LOW]` 태그와 해당 파일:라인 번호를 포함하라.

## 5. 개선 제안
이슈에 대한 구체적인 수정 방향을 간결하게 제시하라.
