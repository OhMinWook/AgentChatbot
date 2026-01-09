# 여기서는 AI의 기억 유지를 담당하는 파트입니다. 기본적인 대화는 REDIS를 통해 기억을 하고 기억 기간은 7일입니다.(CONFIG 확인)
# 유저에 맞춘 장기 기억의 경우 MEM0를 통해 관리할 예정입니다. 이 부분은 아직 구현 되어 있지 않습니다.

import json
from redis import asyncio as aioredis
from typing import List, Dict
from app.core.config import settings


class RedisMemoryService:
    def __init__(self):
        # Redis 연결 풀 생성 (매번 연결하지 않고 재사용)
        self.redis = aioredis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",
            decode_responses=True  # 바이트가 아니라 문자열로 받기
        )

    async def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Redis에서 해당 세션의 대화 기록을 가져옴
        """
        # Redis 키 포맷: "chat:세션ID"
        key = f"chat:{session_id}"

        # 저장된 JSON 문자열을 가져옴
        data = await self.redis.get(key)

        if data:
            full_history = json.loads(data)
            # settings에 정의된 개수(예: 10)를 가져옴
            limit = settings.MAX_HISTORY_COUNT

            return full_history[-limit:]

        return []

    async def add_history(self, session_id: str, user_msg: str, ai_msg: str):
        """
        대화 내용을 추가하고, 오래된 건 자르고, 유효기간(TTL)을 갱신함
        """
        key = f"chat:{session_id}"

        # 1. 기존 기록 가져오기
        current_history = await self.get_history(session_id)

        # 2. 새 대화 추가
        current_history.append({"role": "user", "content": user_msg})
        current_history.append({"role": "assistant", "content": ai_msg})

        # 3. [Sliding Window] 너무 길면 앞부분 자르기
        # 질문+답변이 2개씩 쌓이니까, MAX * 2 만큼만 남김
        max_len = settings.MAX_HISTORY_COUNT * 2
        if len(current_history) > max_len:
            current_history = current_history[-max_len:]

        # 4. Redis에 다시 저장 (덮어쓰기)
        await self.redis.setex(
            name=key,
            time=settings.CHAT_HISTORY_TTL,  # 7일 뒤 자동 폭파
            value=json.dumps(current_history, ensure_ascii=False)
        )


# 인스턴스 생성
memory_service = RedisMemoryService()