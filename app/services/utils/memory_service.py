# 여기서는 AI의 기억 유지를 담당하는 파트입니다. 기본적인 대화는 REDIS를 통해 기억을 하고 기억 기간은 7일입니다.(CONFIG 확인)
# 유저에 맞춘 장기 기억의 경우 MEM0를 통해 관리할 예정입니다. 이 부분은 아직 구현 되어 있지 않습니다.

import json
from typing import List, Dict

from app.core.config import settings
from app.core.redis_client import async_redis


class RedisMemoryService:
    def __init__(self):
        self.redis = async_redis

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Redis에서 전체 대화 기록을 가져옴 (제한 없음)
        """
        key = f"chat:{session_id}"
        data = await self.redis.get(key)

        if not data:
            return []

        return json.loads(data)

    async def add_history(self, session_id: str, user_msg: str, ai_msg: str):
        """
        대화 내용을 추가하고 유효기간(TTL)을 갱신함
        """
        key = f"chat:{session_id}"

        # 기존 기록 가져오기
        data = await self.redis.get(key)
        current_history = json.loads(data) if data else []

        # 새 대화 추가
        current_history.append({"role": "user", "content": user_msg})
        current_history.append({"role": "assistant", "content": ai_msg})

        # Redis에 저장 (TTL 갱신)
        await self.redis.setex(
            name=key,
            time=settings.CHAT_HISTORY_TTL,  # 7일 뒤 자동 삭제
            value=json.dumps(current_history, ensure_ascii=False)
        )


# 인스턴스 생성
memory_service = RedisMemoryService()