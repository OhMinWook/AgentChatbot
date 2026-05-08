"""회의 원문 -> Template3 JSON 추출 서비스"""

import json
import logging
import os
import re
import tempfile
import zipfile
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

from app.services.api_clients.llm_utils import call_llm, strip_markdown_codeblock
from app.services.documents.meeting_minutes_prompts import MEETING_MINUTES_EXTRACTOR
from app.core.config import settings

logger = logging.getLogger(__name__)


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
        for item in attendees_raw[:settings.MAX_MEETING_ATTENDEES]:
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


    async def extract_file_text(self, uploaded_bytes: bytes, uploaded_suffix: str, uploaded_filename: str) -> str:
        """파일 바이트에서 텍스트 추출 (HWPX 직접 파싱 → Polaris/PDF/MarkItDown 폴백)"""
        from app.services.rag.extractors import file_text_extractor
        from app.services.rag.rag_ingestion_service import detect_file_type

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_suffix) as tmp:
                tmp.write(uploaded_bytes)
                tmp_path = tmp.name

            if uploaded_suffix.lower() == ".hwpx":
                text = self._extract_hwpx_text(tmp_path)
                if text:
                    return text

            file_type = detect_file_type(uploaded_filename)
            try:
                page_results, md_content = await file_text_extractor.extract_text(tmp_path, file_type)
            except Exception as ex:
                logger.warning(f"[MeetingMinutes] extractor failed, fallback empty: {ex}")
                return ""

            if md_content:
                return md_content
            elif page_results:
                return "\n".join(text for _, text in page_results)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _extract_hwpx_text(file_path: str) -> Optional[str]:
        """HWPX(ZIP+XML) 파일에서 텍스트를 직접 추출 (한컴 오피스/win32com 불필요).

        1순위: Preview/PrvText.txt (문서 전체 평문, 표/헤더 포함)
        2순위: Contents/section*.xml의 <hp:t> 텍스트 노드
        """
        try:
            with zipfile.ZipFile(file_path, "r") as zf:
                names = zf.namelist()
                if "Preview/PrvText.txt" in names:
                    raw = zf.read("Preview/PrvText.txt")
                    for enc in ("utf-8", "utf-16", "utf-16-le", "cp949"):
                        try:
                            text = raw.decode(enc).strip()
                        except UnicodeDecodeError:
                            continue
                        if text:
                            return text

                texts: list[str] = []
                section_names = sorted(
                    n for n in names
                    if n.startswith("Contents/section") and n.endswith(".xml")
                )
                for name in section_names:
                    try:
                        raw = zf.read(name)
                        root = ET.fromstring(raw)
                    except (KeyError, ET.ParseError):
                        continue
                    for elem in root.iter():
                        tag = elem.tag.split("}", 1)[-1]
                        if tag in ("t", "char") and elem.text:
                            texts.append(elem.text)
                    texts.append("\n")

            joined = "".join(texts)
            joined = re.sub(r"[ \t]+", " ", joined)
            joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
            return joined or None
        except Exception as e:
            logger.error(f"[MeetingMinutes] HWPX direct parse failed: {e}")
            return None


meeting_minutes_service = MeetingMinutesService()
