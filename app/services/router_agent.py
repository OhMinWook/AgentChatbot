"""
RouterAgent - 통합 라우팅 에이전트

문서 업로드 여부에 따라 chat_agent 또는 db_agent로 라우팅
- 문서 있음 → chat_agent (문서 검색)
- 문서 없음 → DB 조회 → 결과 있으면 db_agent, 없으면 chat_agent (벡터 DB 검색)
"""

import logging
import re
from typing import AsyncGenerator, Optional

from app.services.api_clients.llm_utils import call_llm, strip_markdown_codeblock, strip_think_blocks
from app.services.api_clients.llm_client import llm_client
from app.services.utils.llm_payload import build_chat_payload
from app.services.db_agent.prompts import SQL_GENERATOR
from app.services.db_agent.tools import db_tool
from app.services.db_agent.sse_adapter import db_sse_adapter
from app.services.chat_agent.sse_adapter import sse_graph_adapter

logger = logging.getLogger(__name__)


class RouterAgent:

    async def route(
        self,
        invoke_id: str,
        user_query: str,
        target_filename: Optional[str] = None,
    ) -> AsyncGenerator[bytes, None]:
        """
        라우팅 메인 메서드

        - target_filename 있음 → chat_agent (문서 검색)
        - target_filename 없음 → DB 조회 → db_agent or 기본 답변
        """
        if target_filename:
            logger.info(f"[Router] 문서 검색 라우팅: {target_filename}")
            async for chunk in sse_graph_adapter.invoke_with_sse(
                invoke_id=invoke_id,
                user_query=user_query,
                filter_filename=target_filename,
            ):
                yield chunk
        else:
            logger.info(f"[Router] DB 조회 라우팅: {user_query[:50]}")
            async for chunk in self._route_to_db(invoke_id, user_query):
                yield chunk

    async def _route_to_db(
        self, invoke_id: str, user_query: str
    ) -> AsyncGenerator[bytes, None]:
        """DB 조회 후 결과에 따라 db_agent 또는 chat_agent(벡터 DB)로 라우팅"""

        # 1. LLM으로 SQL 생성
        sql = await self._generate_sql(user_query)

        if sql:
            # 2. 품질 필터 서브쿼리 주입 후 DB 조회
            sql = self._inject_quality_filter(sql)
            logger.debug(f"[Router] 품질 필터 적용 SQL:\n{sql}")
            result = await db_tool.query(sql)
            db_results = result.get("results", [])
            logger.info(f"[Router] DB 조회 결과: {len(db_results)}건")

            if db_results:
                # 3a. DB 결과 있음 → db_agent로 답변
                async for chunk in db_sse_adapter.invoke_with_sse(
                    invoke_id=invoke_id,
                    user_query=user_query,
                    db_results=db_results,
                ):
                    yield chunk
                return

        # 3b. SQL 생성 실패 or DB 결과 없음 → 벡터 DB (open 모드) 폴백
        logger.info(f"[Router] DB 결과 없음 - 벡터 DB 폴백: {user_query[:50]}")
        async for chunk in sse_graph_adapter.invoke_with_sse(
            invoke_id=invoke_id,
            user_query=user_query,
            filter_filename=None,
        ):
            yield chunk

    async def _generate_sql(self, question: str) -> str:
        """LLM으로 SQL 생성"""
        payload = build_chat_payload(
            messages=[
                {"role": "system", "content": SQL_GENERATOR.system},
                {"role": "user", "content": SQL_GENERATOR.user.format(question=question)},
            ],
            max_tokens=1024,
        )
        response = await llm_client.chat_completions(payload)
        raw = llm_client.extract_content(response)

        # 1순위: think 블록 제거 후 SELECT 탐색 (모델이 think 외부에 SQL 출력한 경우)
        no_think = strip_think_blocks(raw)
        no_think_clean = strip_markdown_codeblock(no_think).strip()
        if "SELECT" in no_think_clean.upper():
            idx = no_think_clean.upper().find("SELECT")
            sql = self._fix_truncated_sql(no_think_clean[idx:].strip())
            logger.info(f"[Router] 생성된 SQL (think 외부): {sql}")
            return sql

        # 2순위: think 블록 내부에서 SELECT ... FROM 패턴 역순 탐색
        cleaned = strip_markdown_codeblock(raw)
        for match in reversed(list(re.finditer(r"SELECT\b", cleaned, re.IGNORECASE))):
            candidate = cleaned[match.start():]
            candidate = re.sub(r"</think>.*", "", candidate, flags=re.DOTALL).strip()
            if re.search(r"\bFROM\b", candidate, re.IGNORECASE):
                candidate = self._fix_truncated_sql(candidate)
                logger.info(f"[Router] 생성된 SQL (think 내부): {candidate}")
                return candidate

        logger.warning(f"[Router] SQL 미포함 응답: {raw[:200]}")
        return ""

    @staticmethod
    def _inject_quality_filter(sql: str) -> str:
        """LLM 생성 SQL의 wb_v1_task를 품질 필터 서브쿼리로 교체.
        품질 필터 → LLM 검색 조건 순서로 실행됨."""
        quality_subquery = (
            "(\n"
            "    SELECT * FROM wb_v1_task\n"
            "    WHERE (\n"
            "        CHAR_LENGTH(REGEXP_REPLACE(resolution, '<[^>]*>', '')) >= 10\n"
            "        OR CHAR_LENGTH(REGEXP_REPLACE(prevention_measure, '<[^>]*>', '')) >= 10\n"
            "        OR CHAR_LENGTH(REGEXP_REPLACE(failure_cause, '<[^>]*>', '')) >= 10\n"
            "    )\n"
            "    AND (\n"
            "        CHAR_LENGTH(REGEXP_REPLACE(resolution, '<[^>]*>', '')) > 12\n"
            "        OR CHAR_LENGTH(REGEXP_REPLACE(prevention_measure, '<[^>]*>', '')) > 20\n"
            "    )\n"
            "    AND REGEXP_REPLACE(resolution, '<[^>]*>', '') NOT REGEXP '(.{2,})\\\\1{3,}'\n"
            "    AND REGEXP_REPLACE(prevention_measure, '<[^>]*>', '') NOT REGEXP '(.{2,})\\\\1{3,}'\n"
            "    AND REGEXP_REPLACE(failure_cause, '<[^>]*>', '') NOT REGEXP '(.{2,})\\\\1{3,}'\n"
            ") AS wb_v1_task"
        )
        return re.sub(r'\bFROM\s+wb_v1_task\b', f'FROM {quality_subquery}', sql, flags=re.IGNORECASE)

    @staticmethod
    def _fix_truncated_sql(sql: str) -> str:
        """토큰 제한으로 잘린 SQL 보정"""
        # LIMIT 뒤에 숫자가 없으면 10 추가
        if re.search(r"\bLIMIT\s*$", sql, re.IGNORECASE):
            sql = sql + " 10"
        # LIMIT 자체가 없으면 추가
        elif not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
            sql = sql + " LIMIT 10"
        return sql


router_agent = RouterAgent()
