"""
DB Agent Tool - MariaDB 연결 및 조회
"""

import logging
from typing import Any, Dict, List, Optional

import re
import aiomysql

from app.services.agent_base.tool import BaseTool
from app.core.config import settings


def strip_html(text: str) -> str:
    """HTML 태그 제거 및 공백 정리"""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

logger = logging.getLogger(__name__)


class DBTool(BaseTool):
    """MariaDB 조회 도구"""

    @property
    def name(self) -> str:
        return "db_tool"

    async def execute(self, input_text: str, **kwargs) -> Dict[str, Any]:
        """SQL 쿼리를 실행하고 결과를 반환"""
        sql = kwargs.get("sql", input_text)
        return await self.query(sql)

    async def query(self, sql: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """
        SQL 쿼리 실행

        Returns:
            {"results": [...], "success": True} or {"results": [], "success": False, "error": "..."}
        """
        try:
            conn = await aiomysql.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                db=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD,
                charset="utf8mb4",
                autocommit=True,
            )

            async with conn.cursor(aiomysql.DictCursor) as cursor:
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()
                results = [dict(row) for row in rows]

            conn.close()

            # HTML 태그 제거
            cleaned = []
            for row in results:
                cleaned.append({k: strip_html(v) if isinstance(v, str) else v for k, v in row.items()})

            logger.info(f"[DBTool] 조회 완료: {len(cleaned)}건")
            return {"results": cleaned, "success": True}

        except Exception as e:
            logger.error(f"[DBTool] 쿼리 실패: {e}")
            return {"results": [], "success": False, "error": str(e)}

    async def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """테이블 스키마 조회"""
        sql = f"DESCRIBE {table_name}"
        return await self.query(sql)

    async def get_all_tables(self) -> List[str]:
        """DB 내 전체 테이블 목록 조회"""
        result = await self.query("SHOW TABLES")
        if not result["success"]:
            return []
        return [list(row.values())[0] for row in result["results"]]


db_tool = DBTool()
