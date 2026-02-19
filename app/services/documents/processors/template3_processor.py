import re
import logging
from html import escape

from .base_processor import BaseTemplateProcessor

logger = logging.getLogger(__name__)


class Template3Processor(BaseTemplateProcessor):
    """'template3.hwpx' 회의록 전용 처리기"""

    def generate(self, context_data: dict) -> bytes:
        template_path, section0 = self._load_section0("template3.hwpx")

        # --- 단순 필드 교체 ---
        section0 = self._replace_fields(section0, {
            "{{AGENDA_TITLE}}": escape(str(context_data.get("meeting_title", ""))),
            "{{DATETIME}}": escape(str(context_data.get("datetime", ""))),
            "{{PERSON_IN_CHARGE}}": escape(str(context_data.get("person_in_charge", ""))),
            "{{LOCATION}}": escape(str(context_data.get("location", ""))),
            "{{ATTENDEE_COUNT}}": escape(str(context_data.get("attendee_count", ""))),
            "{{MAIN_AGENDA}}": escape(str(context_data.get("agenda", ""))),
        })

        # --- 참석자 필드 교체 ---
        attendees = context_data.get("attendees", [])
        for i in range(8):
            att = attendees[i] if i < len(attendees) else {}
            section0 = section0.replace(
                f"{{{{ATT_AFF_{i+1}}}}}", escape(str(att.get("affiliation", "")))
            )
            section0 = section0.replace(
                f"{{{{ATT_NAME_{i+1}}}}}", escape(str(att.get("name", "")))
            )

        # --- 회의 내용 (원본 구조 보존 + 문단 수 유지) ---
        content_lines = context_data.get("meeting_content_lines", [])
        section0 = self._replace_cell_paragraphs(section0, "{{MEETING_CONTENT}}", content_lines)

        # --- 회의 결과 (원본 구조 보존 + 문단 수 유지) ---
        result_lines = context_data.get("meeting_result_lines", [])
        section0 = self._replace_cell_paragraphs(section0, "{{MEETING_RESULT}}", result_lines)

        # --- 레이아웃 메타데이터 제거 (한글이 열 때 재계산하도록) ---
        section0 = re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>', '', section0, flags=re.DOTALL)

        return self._repackage(template_path, section0)

    @staticmethod
    def _replace_cell_paragraphs(xml: str, placeholder: str, lines: list) -> str:
        """
        플레이스홀더가 포함된 셀의 모든 문단을 교체합니다.
        - 원본 첫 문단의 XML 구조를 템플릿으로 사용하여 서식 보존
        - 원본 빈 스페이서 구조를 그대로 재사용
        - 원본 문단 수를 유지하여 레이아웃 안정성 확보
        """
        idx = xml.find(placeholder)
        if idx == -1:
            return xml

        # 셀 내용 영역(<hp:subList>) 찾기
        sublist_start = xml.rfind('<hp:subList', 0, idx)
        sublist_end = xml.find('</hp:subList>', idx) + len('</hp:subList>')
        if sublist_start == -1 or sublist_end < sublist_start:
            return xml

        old_sublist = xml[sublist_start:sublist_end]

        # 기존 <hp:subList> 열기 태그 보존
        tag_end = old_sublist.find('>') + 1
        sublist_open_tag = old_sublist[:tag_end]

        # 원본 첫 문단(텍스트가 있는 문단) 구조를 템플릿으로 추출
        first_para_match = re.search(r'<hp:p .*?</hp:p>', old_sublist, re.DOTALL)
        if not first_para_match:
            return xml
        first_para_template = first_para_match.group()

        # 원본 빈 스페이서 문단 구조 추출
        empty_para_match = re.search(r'<hp:p [^>]*><hp:run [^/]*/>(.*?)</hp:p>', old_sublist, re.DOTALL)
        empty_para_template = empty_para_match.group() if empty_para_match else None

        # 원본 전체 문단 수 계산
        original_para_count = len(re.findall(r'<hp:p ', old_sublist))

        # 새 내용 문단 생성 (원본 구조의 텍스트만 교체, count=1로 첫 번째만)
        if not lines:
            lines = [""]
        new_paras = ''
        for text in lines:
            new_para = re.sub(
                r'<hp:t>.*?</hp:t>',
                f'<hp:t>{escape(text)}</hp:t>',
                first_para_template,
                count=1,
            )
            new_paras += new_para

        # 빈 스페이서로 원본 문단 수 맞추기
        spacer_count = max(0, original_para_count - len(lines))
        if empty_para_template and spacer_count > 0:
            new_paras += empty_para_template * spacer_count

        return xml[:sublist_start] + sublist_open_tag + new_paras + '</hp:subList>' + xml[sublist_end:]
