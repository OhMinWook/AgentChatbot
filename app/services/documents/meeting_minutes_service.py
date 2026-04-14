"""회의 원문 -> Template3 JSON 추출 서비스"""

import json
import logging
from typing import Any, Dict, List

from app.services.api_clients.llm_utils import call_llm, strip_markdown_codeblock
from app.services.documents.meeting_minutes_prompts import MEETING_MINUTES_EXTRACTOR

logger = logging.getLogger(__name__)

MAX_ATTENDEES = 8


class MeetingMinutesService:
    """회의 원문에서 Template3 스키마에 맞는 JSON을 추출"""

    async def extract_meeting_minutes_json(self, raw_text: str) -> Dict[str, Any]:
        if not raw_text or not raw_text.strip():
            raise ValueError("raw_text is empty")

        messages = [
            {"role": "system", "content": MEETING_MINUTES_EXTRACTOR.system},
            {"role": "user", "content": MEETING_MINUTES_EXTRACTOR.user.format(raw_text=raw_text)},
        ]

        response = await call_llm(messages, max_tokens=4096)
        if not response:
            raise RuntimeError("LLM returned empty response")

        data = self._parse_json(response)
        if data is None:
            # 한 번 재시도 (LLM 출력 불안정 대응)
            logger.warning("[MeetingMinutes] First JSON parse failed, retrying once")
            response = await call_llm(messages, max_tokens=4096)
            data = self._parse_json(response)

        if data is None:
            raise ValueError("Failed to parse LLM response as JSON")

        return self._normalize(data)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any] | None:
        cleaned = strip_markdown_codeblock(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # JSON 블록만 뽑아서 재시도
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(cleaned[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _normalize(data: Dict[str, Any]) -> Dict[str, Any]:
        """Template3Processor가 기대하는 스키마로 정규화"""
        attendees_raw = data.get("attendees") or []
        attendees: List[Dict[str, str]] = []
        for item in attendees_raw[:MAX_ATTENDEES]:
            if not isinstance(item, dict):
                continue
            attendees.append({
                "affiliation": str(item.get("affiliation", "")),
                "name": str(item.get("name", "")),
            })

        content_lines = data.get("meeting_content_lines") or []
        result_lines = data.get("meeting_result_lines") or []

        if not isinstance(content_lines, list):
            content_lines = [str(content_lines)]
        if not isinstance(result_lines, list):
            result_lines = [str(result_lines)]

        content_lines = [str(x) for x in content_lines if str(x).strip()]
        result_lines = [str(x) for x in result_lines if str(x).strip()]

        if not content_lines:
            content_lines = [""]
        if not result_lines:
            result_lines = [""]

        return {
            "meeting_title": str(data.get("meeting_title", "")),
            "datetime": str(data.get("datetime", "")),
            "person_in_charge": str(data.get("person_in_charge", "")),
            "location": str(data.get("location", "")),
            "attendee_count": str(data.get("attendee_count", str(len(attendees)))),
            "agenda": str(data.get("agenda", "")),
            "attendees": attendees,
            "meeting_content_lines": content_lines,
            "meeting_result_lines": result_lines,
        }


meeting_minutes_service = MeetingMinutesService()
