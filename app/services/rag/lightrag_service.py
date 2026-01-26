import logging
import asyncio
from typing import List, Dict, Any, Optional
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from app.core.config import settings
from app.services.clients.llm_client import llm_client

logger = logging.getLogger(__name__)

# 엔티티 및 관계 추출 프롬프트 (LightRAG 스타일)
ENTITY_EXTRACTION_PROMPT = """
당신은 텍스트에서 지식 그래프를 구축하기 위한 데이터 추출 전문가입니다.
주어진 텍스트에서 중요한 **엔티티(Entity)**와 그들 간의 **관계(Relationship)**를 추출하세요.

## 추출 가이드라인
1. **엔티티**: 사람, 조직, 장소, 개념, 사건, 날짜 등 중요한 명사구.
2. **관계**: 엔티티 사이의 상호작용이나 속성을 나타내는 동사구.
3. **형식**: (주체_엔티티) -[관계]-> (목적어_엔티티)

## 텍스트
{text}

## 출력 형식 (JSON)
{{
  "entities": [
    {{"name": "엔티티이름", "type": "유형", "description": "한줄설명"}}
  ],
  "relationships": [
    {{"source": "주체", "target": "목적어", "type": "관계유형", "description": "관계설명"}}
  ]
}}
"""

GRAPH_SEARCH_PROMPT = """
당신은 지식 그래프 검색 전문가입니다.
다음은 질문과 관련된 지식 그래프의 엔티티 및 관계 정보입니다.
이를 바탕으로 질문에 대한 포괄적인 맥락과 답을 서술형으로 작성하세요.

## 질문
{question}

## 그래프 데이터
{graph_context}

## 답변 작성 가이드
- 그래프의 연결 관계를 활용하여 숨겨진 맥락을 설명하세요.
- 구체적인 사실보다는 전체적인 구조와 흐름에 집중하세요.
"""

class LightRAGService:
    def __init__(self):
        self.neo4j_graph = None
        self._connect_neo4j()

    def _connect_neo4j(self):
        """Neo4j 데이터베이스 연결"""
        try:
            self.neo4j_graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD
            )
            logger.info("✅ Neo4j Connected for LightRAG")
        except Exception as e:
            logger.error(f"⚠️ Neo4j Connection Failed: {e}")
            self.neo4j_graph = None

    async def index_chunks(self, chunks: List[Dict], invoke_id: str, on_progress=None):
        """
        청크 목록을 받아 엔티티/관계를 추출하고 Neo4j에 저장합니다.
        (비동기 처리를 위해 배치로 실행)
        
        Args:
            chunks: 청크 리스트
            invoke_id: 세션 ID
            on_progress: 진행률 콜백 함수 (async def func(percent, message)) - 50% ~ 100% 구간 담당
        """
        if not self.neo4j_graph:
            logger.warning("Neo4j not connected, skipping indexing.")
            return

        total_chunks = len(chunks)
        logger.info(f"🌿 [LightRAG] {total_chunks}개 청크 인덱싱 시작...")

        # 비동기 세마포어로 동시 실행 제한 (LLM 부하 조절)
        sem = asyncio.Semaphore(3)
        completed_count = 0

        async def process_chunk(chunk):
            nonlocal completed_count
            async with sem:
                text = chunk.get("content", "")
                if not text:
                    return
                
                # 메타데이터에서 소스(파일명) 추출
                metadata = chunk.get("metadata", {})
                source_file = metadata.get("source", "unknown")
                
                # 1. LLM으로 엔티티/관계 추출
                try:
                    extraction_payload = {
                        "model": settings.VLLM_MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a knowledge graph extractor."},
                            {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=text[:3000])} # 길이 제한
                        ],
                        "max_tokens": 2048,
                        "temperature": 0,
                        "guided_json": {
                            "type": "object",
                            "properties": {
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "type": {"type": "string"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["name", "type"]
                                    }
                                },
                                "relationships": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "source": {"type": "string"},
                                            "target": {"type": "string"},
                                            "type": {"type": "string"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["source", "target", "type"]
                                    }
                                }
                            },
                            "required": ["entities", "relationships"]
                        }
                    }
                    
                    response = await llm_client.chat_completions(extraction_payload)
                    result_json = response['choices'][0]['message']['content']
                    
                    import json
                    data = json.loads(result_json)
                    
                    # 2. Neo4j에 Cypher 쿼리로 저장
                    queries = []
                    
                    # 엔티티 생성 (MERGE)
                    for entity in data.get("entities", []):
                        # invoke_id와 name으로 유니크 엔티티 식별
                        # source는 리스트 형태로 관리하거나, 가장 최근 파일명으로 덮어씀 (여기선 단순화하여 덮어쓰기)
                        # 실제로는 한 엔티티가 여러 문서에 나올 수 있으므로, source를 속성으로 관리 시 주의 필요
                        # 여기서는 간단히 '주요 출처' 개념으로 저장
                        cypher = """
                        MERGE (e:Entity {name: $name, invoke_id: $invoke_id})
                        ON CREATE SET e.type = $type, e.description = $desc, e.source = $source
                        ON MATCH SET e.source = $source
                        """
                        queries.append((cypher, {
                            "name": entity["name"], 
                            "type": entity.get("type", "Thing"), 
                            "desc": entity.get("description", ""),
                            "invoke_id": invoke_id,
                            "source": source_file
                        }))
                    
                    # 관계 생성 (MERGE)
                    for rel in data.get("relationships", []):
                        cypher = f"""
                        MATCH (a:Entity {{name: $source_node, invoke_id: $invoke_id}})
                        MATCH (b:Entity {{name: $target_node, invoke_id: $invoke_id}})
                        MERGE (a)-[r:{rel['type'].upper().replace(' ', '_')}]->(b)
                        SET r.description = $desc, r.source = $source
                        """
                        queries.append((cypher, {
                            "source_node": rel["source"],
                            "target_node": rel["target"],
                            "desc": rel.get("description", ""),
                            "invoke_id": invoke_id,
                            "source": source_file
                        }))
                        
                    # Neo4j 실행
                    if queries:
                        await asyncio.to_thread(self._execute_batch, queries)
                
                except Exception as e:
                    logger.error(f"Extraction failed for chunk: {e}")
                
                # 진행률 업데이트 (50% ~ 100% 구간)
                completed_count += 1
                if on_progress:
                    # LightRAG는 전체 공정의 50% ~ 95% 정도 차지한다고 가정
                    progress = 50 + int((completed_count / total_chunks) * 45)
                    await on_progress(progress, f"지식 그래프 구축 중 ({completed_count}/{total_chunks})")

        # 태스크 생성 및 실행
        tasks = [process_chunk(chunk) for chunk in chunks]
        await asyncio.gather(*tasks)
        
        logger.info(f"🌿 [LightRAG] 인덱싱 완료")

    def _execute_batch(self, queries):
        """배치 쿼리 실행"""
        if not self.neo4j_graph:
            return
        try:
            # LangChain Neo4jGraph의 query 메서드 사용
            for cypher, params in queries:
                self.neo4j_graph.query(cypher, params)
        except Exception as e:
            logger.error(f"Neo4j batch execution failed: {e}")

    async def search(self, query: str, invoke_id: str, filename: Optional[str] = None) -> str:
        """
        그래프 기반 검색 수행
        
        Args:
            query: 질문 텍스트
            invoke_id: 세션 ID
            filename: (Optional) 특정 파일 내에서만 검색하려면 파일명 지정
        """
        if not self.neo4j_graph:
            return ""

        try:
            # 기본 조건: invoke_id 일치 + 이름 매칭
            where_clause = "WHERE e.name CONTAINS $query OR $query CONTAINS e.name"
            
            # 파일명 필터링 추가 (filename이 있을 때만)
            if filename:
                where_clause += " AND (e.source = $filename OR e.source CONTAINS $filename)"
            
            search_cypher = f"""
            MATCH (e:Entity {{invoke_id: $invoke_id}})
            {where_clause}
            OPTIONAL MATCH (e)-[r]-(neighbor)
            RETURN e.name, type(r) as rel, neighbor.name, r.description, e.source
            LIMIT 20
            """
            
            params = {"invoke_id": invoke_id, "query": query}
            if filename:
                params["filename"] = filename
            
            results = await asyncio.to_thread(
                self.neo4j_graph.query, 
                search_cypher, 
                params
            )
            
            if not results:
                return ""
            
            # 그래프 컨텍스트 텍스트화
            graph_context = "\n".join([
                f"({r['e.name']}) -[{r['rel']}]-> ({r['neighbor.name']}): {r['r.description']} (Source: {r.get('e.source', 'unknown')})"
                for r in results if r['neighbor.name']
            ])
            
            return graph_context

        except Exception as e:
            logger.error(f"LightRAG search failed: {e}")
            return ""

lightrag_service = LightRAGService()
