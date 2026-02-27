import logging

from .base_processor import BaseTemplateProcessor

logger = logging.getLogger(__name__)


class Template2Processor(BaseTemplateProcessor):
    """'template2.hwpx' 일반 공문 전용 처리기"""

    def generate(self, context_data: dict) -> bytes:
        template_path, section0 = self._load_section0("template2.hwpx")

        # --- 단순 필드 교체 ---
        section0 = self._replace_fields(section0, {
            "{{RECIPIENT}}": context_data.get("recipient", ""),
            "{{VIA}}": context_data.get("via", ""),
            "{{TITLE}}": context_data.get("title", ""),
            "{{ATTACHMENT}}": context_data.get("attachment", ""),
        })

        # --- 본문 교체 ---
        content_lines = context_data.get("content_lines", [])
        new_paragraphs = ''.join(self._make_paragraph(line) for line in content_lines)
        section0 = self._replace_content_block(section0, "{{CONTENT}}", new_paragraphs)

        return self._repackage(template_path, section0)

    @staticmethod
    def _make_paragraph(text: str) -> str:
        return (
            f'<hp:p id="2147483648" paraPrIDRef="0" styleIDRef="0">'
            f'<hp:run charPrIDRef="9"><hp:t>{text}</hp:t></hp:run>'
            f'<hp:linesegarray><hp:lineseg textpos="0" vertpos="0" vertsize="1200" '
            f'textheight="1200" baseline="1020" spacing="720" horzpos="0" '
            f'horzsize="49608" flags="393216"/></hp:linesegarray></hp:p>'
        )
