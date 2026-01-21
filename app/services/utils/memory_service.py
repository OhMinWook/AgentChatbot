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
        Redis에서 대화 기록을 가져오되, 글자 수와 개수 제한을 모두 적용함
        """
        key = f"chat:{session_id}"
        data = await self.redis.get(key)

        if not data:
            return []

        full_history = json.loads(data)

        # settings에서 설정값 로드
        char_limit = settings.MAX_HISTORY_CHARS  # 1500
        count_limit = settings.MAX_HISTORY_COUNT  # 5

        limited_history = []
        current_chars = 0

        # 1. 가장 최근 메시지부터 역순으로 순회
        for message in reversed(full_history):
            # 2. 개수 제한 확인 (이미 설정된 개수를 넘으면 중단)
            if len(limited_history) >= count_limit:
                break

            content = message.get("content", "")
            msg_len = len(content)

            # 3. 글자 수 제한 확인
            if current_chars + msg_len > char_limit:
                # 첫 메시지가 너무 길 경우 최소 하나는 보장하거나, 바로 중단
                if not limited_history:
                    limited_history.append(message)
                break

            limited_history.append(message)
            current_chars += msg_len
        # 4. 역순으로 담았으므로 다시 시간 순서(정방향)로 뒤집어서 반환
        return limited_history[::-1]

    async def add_history(self, session_id: str, user_msg: str, ai_msg: str):
        """
        대화 내용을 추가하고, 오래된 건 자르고, 유효기간(TTL)을 갱신함
        """
        key = f"chat:{session_id}"

        # 1. 기존 기록 가져오기 (원본 전체 데이터를 가져와야 함)
        data = await self.redis.get(key)
        current_history = json.loads(data) if data else []

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