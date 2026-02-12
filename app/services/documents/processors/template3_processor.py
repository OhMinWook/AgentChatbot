import zipfile
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "documents"


class Template3Processor:
    """'template3.hwpx' 회의록 전용 처리기"""

    def generate(self, context_data: dict) -> bytes:
        template_name = "template3.hwpx"
        input_template_path = TEMPLATE_DIR / template_name
        if not input_template_path.exists():
            raise FileNotFoundError(f"템플릿 파일 '{template_name}'을 찾을 수 없습니다.")

        with tempfile.TemporaryDirectory() as temp_dir:
            final_output_path = Path(temp_dir) / f"generated_{template_name}"

            with zipfile.ZipFile(input_template_path, 'r') as z_in:
                section0_content = z_in.read("Contents/section0.xml").decode('utf-8')

            # --- 단순 필드 플레이스홀더 교체 ---
            field_replacements = {
                "{{AGENDA_TITLE}}": context_data.get("meeting_title", ""),
                "{{DATETIME}}": context_data.get("datetime", ""),
                "{{PERSON_IN_CHARGE}}": context_data.get("person_in_charge", ""),
                "{{LOCATION}}": context_data.get("location", ""),
                "{{ATTENDEE_COUNT}}": context_data.get("attendee_count", ""),
                "{{MAIN_AGENDA}}": context_data.get("agenda", ""),
            }
            for placeholder, value in field_replacements.items():
                section0_content = section0_content.replace(placeholder, str(value))

            # --- 참석자 필드 교체 ---
            attendees = context_data.get("attendees", [])
            for i in range(8):
                att = attendees[i] if i < len(attendees) else {}
                section0_content = section0_content.replace(
                    f"{{{{ATT_AFF_{i+1}}}}}", str(att.get("affiliation", ""))
                )
                section0_content = section0_content.replace(
                    f"{{{{ATT_NAME_{i+1}}}}}", str(att.get("name", ""))
                )

            # --- 회의 내용 (복수 문단) 교체 ---
            section0_content = self._replace_content_block(
                section0_content, "{{MEETING_CONTENT}}",
                context_data.get("meeting_content_lines", [])
            )

            # --- 회의 결과 (복수 문단) 교체 ---
            section0_content = self._replace_content_block(
                section0_content, "{{MEETING_RESULT}}",
                context_data.get("meeting_result_lines", [])
            )

            # --- 재패키징 ---
            with zipfile.ZipFile(input_template_path, 'r') as z_read:
                with zipfile.ZipFile(final_output_path, 'w') as z_write:
                    for item in z_read.infolist():
                        if item.filename == "Contents/section0.xml":
                            z_write.writestr(item, section0_content.encode('utf-8'))
                        else:
                            z_write.writestr(item, z_read.read(item.filename))

            with open(final_output_path, "rb") as f:
                return f.read()

    @staticmethod
    def _make_paragraph(text: str) -> str:
        return (
            '<hp:p id="0" paraPrIDRef="3" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">'
            f'<hp:run charPrIDRef="6"><hp:t>{text}</hp:t></hp:run></hp:p>'
        )

    def _replace_content_block(self, xml_content: str, placeholder: str, lines: list) -> str:
        content_idx = xml_content.find(placeholder)
        if content_idx == -1:
            return xml_content

        p_start = xml_content.rfind('<hp:p', 0, content_idx)
        p_end = xml_content.find('</hp:p>', content_idx) + len('</hp:p>')
        if p_start == -1 or p_end < p_start:
            return xml_content

        new_paragraphs = ''.join(self._make_paragraph(line) for line in lines)
        if not new_paragraphs:
            new_paragraphs = self._make_paragraph("")

        return xml_content[:p_start] + new_paragraphs + xml_content[p_end:]
