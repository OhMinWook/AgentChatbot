import logging
import asyncio
import re
from typing import List, Dict, Any, Optional
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from app.core.config import settings
from app.services.api_clients.llm_client import llm_client
from app.services.api_clients.model_server_client import model_server_client

logger = logging.getLogger(__name__)

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
                password=settings.NEO4J_PASSWORD,
                refresh_schema=False  # APOC 의존성 제거 및 초기화 속도 향상
            )
            logger.info("✅ Neo4j Connected for LightRAG")
        except Exception as e:
            logger.error(f"⚠️ Neo4j Connection Failed: {e}")
            self.neo4j_graph = None

    async def index_chunks(self, chunks: List[Dict], invoke_id: str, on_progress=None):
        """
        청크 목록을 받아 엔티티/관계를 추출하고 Neo4j에 저장합니다.
        (모델 서버의 경량화 LLM 활용)
        
        Args:
            chunks: 청크 리스트
            invoke_id: 세션 ID
            on_progress: 진행률 콜백 함수
        """
        if not self.neo4j_graph:
            logger.warning("Neo4j not connected, skipping indexing.")
            return

        total_chunks = len(chunks)
        logger.info(f"🌿 [LightRAG] {total_chunks}개 청크 인덱싱 시작 (모델 서버 활용)...")

        # 배치 단위로 모델 서버에 요청 (한 번에 8개씩)
        batch_size = 8
        completed_count = 0

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [c.get("content", "") for c in batch]
            
            # 1. 모델 서버에서 엔티티/관계 배치 추출
            try:
                graph_results = await model_server_client.extract_graph_batch(batch_texts)
                
                # 2. Neo4j 저장 쿼리 생성
                all_queries = []
                
                for idx, data in enumerate(graph_results):
                    chunk_meta = batch[idx].get("metadata", {})
                    source_file = chunk_meta.get("source", "unknown")
                    
                    # 엔티티 쿼리
                    for entity in data.get("entities", []):
                        cypher = """
                        MERGE (e:Entity {name: $name, invoke_id: $invoke_id})
                        ON CREATE SET e.type = $type, e.description = $desc, e.source = $source
                        ON MATCH SET e.source = $source
                        """
                        all_queries.append((cypher, {
                            "name": entity["name"], 
                            "type": entity.get("type", "Thing"), 
                            "desc": entity.get("description", ""),
                            "invoke_id": invoke_id,
                            "source": source_file
                        }))
                    
                    # 관계 쿼리
                    for rel in data.get("relationships", []):
                        # 관계 타입 정제: 알파벳, 숫자, 언더바 외에는 모두 언더바로 치환
                        safe_type = re.sub(r'[^A-Z0-9_]', '_', rel['type'].upper())
                        
                        cypher = f"""
                        MATCH (a:Entity {{name: $source_node, invoke_id: $invoke_id}})
                        MATCH (b:Entity {{name: $target_node, invoke_id: $invoke_id}})
                        MERGE (a)-[r:{safe_type}]->(b)
                        SET r.description = $desc, r.source = $source
                        """
                        all_queries.append((cypher, {
                            "source_node": rel["source"],
                            "target_node": rel["target"],
                            "desc": rel.get("description", ""),
                            "invoke_id": invoke_id,
                            "source": source_file
                        }))
                
                # 3. Neo4j 실행
                if all_queries:
                    await asyncio.to_thread(self._execute_batch, all_queries)
                    
            except Exception as e:
                logger.error(f"Batch graph extraction/storage failed: {e}")

            # 진행률 업데이트
            completed_count += len(batch)
            if on_progress:
                progress = 50 + int((completed_count / total_chunks) * 45)
                await on_progress(progress, f"지식 그래프 구축 중 ({completed_count}/{total_chunks})")

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
