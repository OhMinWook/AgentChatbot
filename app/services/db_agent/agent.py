"""
DB Agent 메인 오케스트레이터

현재: DB 결과 → LLM 답변 생성
추후: 서브 에이전트 등록/실행으로 확장 가능
"""

import logging
from typing import Any, Dict, List

from app.core.langfuse_client import observe  # langfuse 비활성화 스텁

from app.services.agent_base.base_agent import BaseAgent
from app.services.db_agent.prompts import AGENT
from app.core.config import settings

logger = logging.getLogger(__name__)


class DBMainAgent(BaseAgent):

    def __init__(self):
        self._sub_agents: List[BaseAgent] = []  # 추후 서브 에이전트 등록

    @property
    def name(self) -> str:
        return "db_main_agent"

    def register_agent(self, agent: BaseAgent) -> None:
        """서브 에이전트 등록"""
        self._sub_agents.append(agent)
        logger.info(f"[DBMainAgent] 서브 에이전트 등록: {agent.name}")

    @observe()
    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        DB 챗봇 메인 실행

        Args:
            query: 사용자 질문
            db_results: DB 조회 결과 (list of dict)

        Returns:
            {
                "precomputed": bool,
                "content": str,        # precomputed=True 일 때
                "messages": list,      # precomputed=False 일 때 (스트리밍용)
                "max_tokens": int,
            }
        """
        db_results = kwargs.get("db_results", [])

        if not db_results:
            logger.info(f"[DBMainAgent] DB 결과 없음: {query[:50]}")
            return {"precomputed": True, "content": "관련 장애 이력을 찾지 못했습니다."}

        context = self._format_db_results(db_results)
        messages = [
            {"role": "system", "content": AGENT.system},
            {"role": "user", "content": AGENT.user.format(context=context, question=query)}
        ]

        logger.info(f"[DBMainAgent] 답변 생성 준비: {query[:50]}")
        return {
            "precomputed": False,
            "messages": messages,
            "max_tokens": settings.DEFAULT_MAX_TOKENS,
        }

    def _format_db_results(self, db_results: list) -> str:
        """DB 조회 결과를 LLM 컨텍스트 문자열로 변환"""
        lines = []
        for i, row in enumerate(db_results, 1):
            lines.append(f"[{i}]")
            for key, value in row.items():
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


db_main_agent = DBMainAgent()
