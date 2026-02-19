import re
import logging
from jinja2 import Template

from .base_processor import BaseTemplateProcessor

logger = logging.getLogger(__name__)


class Template1Processor(BaseTemplateProcessor):
    """'template.hwpx' 기안문/시행문 전용 처리기"""

    def generate(self, context_data: dict) -> bytes:
        template_path, section0 = self._load_section0("template.hwpx")

        # --- 단순 필드 교체 ---
        section0 = self._replace_fields(section0, {
            "{{DOC_NUMBER}}": context_data.get("doc_number", ""),
            "{{DRAFT_DATE}}": context_data.get("draft_date", ""),
            "{{EXEC_DATE}}": context_data.get("exec_date", ""),
            "{{VIA}}": context_data.get("via", ""),
            "{{RECIPIENT}}": context_data.get("recipient", ""),
            "{{REFERENCE}}": context_data.get("reference", ""),
            "{{TITLE}}": context_data.get("title", ""),
            "{{RETENTION}}": context_data.get("retention", ""),
            "{{SIGN_MANAGER}}": context_data.get("sign_manager", ""),
            "{{SIGN_DRAFTER}}": context_data.get("sign_drafter", ""),
            "{{SIGN_COOP}}": context_data.get("sign_coop", ""),
            "{{DOC_NUMBER_2}}": context_data.get("doc_number_2", ""),
            "{{EXEC_DATE_2}}": context_data.get("exec_date_2", ""),
            "{{VIA_2}}": context_data.get("via_2", ""),
            "{{RECIPIENT_2}}": context_data.get("recipient_2", ""),
            "{{REFERENCE_2}}": context_data.get("reference_2", ""),
            "{{TITLE_2}}": context_data.get("title_2", ""),
        })

        # --- 본문 항목 처리 ---
        items_data = context_data.get("items", [])
        processed_items = []
        for idx, item in enumerate(items_data):
            text = item.get('text', '')
            if idx == len(items_data) - 1 and not context_data.get('has_attachment', False):
                text += "  끝."
            processed_items.append({
                "style_id": "2" if item.get('level', 1) == 1 else "1",
                "para_pr_id": "28" if item.get('level', 1) == 1 else "29",
                "text": text,
            })
        if not items_data and not context_data.get('has_attachment', False):
            processed_items.append({"style_id": "2", "para_pr_id": "28", "text": "끝."})

        xml_template = Template(
            """{%- for item in items -%}"""
            """<hp:p id="0" paraPrIDRef="{{ item.para_pr_id }}" styleIDRef="{{ item.style_id }}">"""
            """<hp:run charPrIDRef="19"><hp:t>{{ item.text }}</hp:t></hp:run></hp:p>"""
            """{%- endfor -%}"""
        )
        new_xml_block = xml_template.render(items=processed_items)

        section0 = self._replace_content_block(section0, "{{CONTENT}}", new_xml_block)
        section0 = self._replace_content_block(section0, "{{CONTENT_2}}", new_xml_block)
        section0 = self._adjust_empty_paragraphs(section0, context_data)

        return self._repackage(template_path, section0)

    def _adjust_empty_paragraphs(self, section0_content: str, context_data: dict) -> str:
        content_line_count = len(context_data.get('items', []))
        if not content_line_count:
            return section0_content

        empty_para_pattern = r'<hp:p[^>]*><hp:run[^>]*/><hp:linesegarray>.*?</hp:linesegarray></hp:p>'

        # --- 기안문 빈 문단 조절 ---
        section2_marker = '붙임.2'
        section2_start_idx = section0_content.find(section2_marker)
        if section2_start_idx == -1:
            section2_start_idx = len(section0_content)

        sign_marker = '신 사 종 합 사 회 복 지 관 장'
        coop_idx = section0_content.find('협조')

        first_item_text = context_data['items'][0].get('text') if context_data.get('items') else None
        if first_item_text:
            first_item_idx = section0_content.find(first_item_text)
            end_marker_idx = section0_content.find('끝.')
            sign_idx = section0_content.find(sign_marker)

            if all(i > -1 and i < section2_start_idx for i in [first_item_idx, end_marker_idx, sign_idx, coop_idx]):
                section0_content = self._remove_lines_in_section(
                    section0_content, empty_para_pattern, content_line_count,
                    start_boundary=coop_idx, content_start=first_item_idx,
                    content_end=end_marker_idx, end_boundary=sign_idx,
                    section_name="기안문"
                )

        # --- 시행문 빈 문단 조절 ---
        if section2_start_idx < len(section0_content):
            staff_in_section2 = section0_content.find('담 당 자', section2_start_idx)
            if first_item_text:
                first_in_s1 = section0_content.find(first_item_text)
                first_in_s2 = section0_content.find(first_item_text, first_in_s1 + 1)

                first_end_s1 = section0_content.find('끝.')
                second_end_s2 = section0_content.find('끝.', first_end_s1 + 1)

                first_sign_s1 = section0_content.find(sign_marker)
                second_sign_s2 = section0_content.find(sign_marker, first_sign_s1 + 1)

                if all(i > -1 for i in [first_in_s2, second_end_s2, second_sign_s2, staff_in_section2]):
                    section0_content = self._remove_lines_in_section(
                        section0_content, empty_para_pattern, content_line_count,
                        start_boundary=staff_in_section2, content_start=first_in_s2,
                        content_end=second_end_s2, end_boundary=second_sign_s2,
                        section_name="시행문"
                    )
        return section0_content

    def _remove_lines_in_section(self, xml_content, pattern, line_count, start_boundary, content_start, content_end, end_boundary, section_name):
        before_section = xml_content[start_boundary:content_start]
        before_empties = list(re.finditer(pattern, before_section, re.DOTALL))

        after_start = xml_content.find('</hp:p>', content_end) + len('</hp:p>')
        after_section = xml_content[after_start:end_boundary]
        after_empties = list(re.finditer(pattern, after_section, re.DOTALL))

        lines_to_remove = min(line_count - 1,
                              max(0, len(before_empties) - 3) + max(0, len(after_empties) - 3))

        if lines_to_remove <= 0:
            return xml_content

        to_delete = []
        removed_before, removed_after = 0, 0

        before_candidates = [(start_boundary + m.start(), start_boundary + m.end()) for m in reversed(before_empties)]
        after_candidates = [(after_start + m.start(), after_start + m.end()) for m in after_empties]

        max_before = max(0, len(before_candidates) - 3)
        max_after = max(0, len(after_candidates) - 3)

        for i in range(lines_to_remove):
            if i % 2 == 0:
                if removed_after < max_after:
                    to_delete.append(after_candidates[removed_after])
                    removed_after += 1
                elif removed_before < max_before:
                    to_delete.append(before_candidates[removed_before])
                    removed_before += 1
            else:
                if removed_before < max_before:
                    to_delete.append(before_candidates[removed_before])
                    removed_before += 1
                elif removed_after < max_after:
                    to_delete.append(after_candidates[removed_after])
                    removed_after += 1

        to_delete.sort(key=lambda x: x[0], reverse=True)
        for start, end in to_delete:
            xml_content = xml_content[:start] + xml_content[end:]

        logger.info(f"[{section_name} 조정] 빈 문단 삭제: 위쪽 {removed_before}개, 아래쪽 {removed_after}개")
        return xml_content
