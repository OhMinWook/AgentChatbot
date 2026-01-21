import zipfile
import re
import tempfile
import logging
from pathlib import Path
from jinja2 import Template

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "documents"

class Template1Processor:
    """'template.hwpx' 전용 처리기"""

    def generate(self, context_data: dict) -> bytes:
        template_name = "template.hwpx"
        input_template_path = TEMPLATE_DIR / template_name
        if not input_template_path.exists():
            raise FileNotFoundError(f"템플릿 파일 '{template_name}'을 찾을 수 없습니다.")

        with tempfile.TemporaryDirectory() as temp_dir:
            final_output_path = Path(temp_dir) / f"generated_{template_name}"
            with zipfile.ZipFile(input_template_path, 'r') as z_in:
                section0_content = z_in.read("Contents/section0.xml").decode('utf-8')

            field_replacements = {
                "{{DOC_NUMBER}}": context_data.get("doc_number", ""), "{{DRAFT_DATE}}": context_data.get("draft_date", ""),
                "{{EXEC_DATE}}": context_data.get("exec_date", ""), "{{VIA}}": context_data.get("via", ""),
                "{{RECIPIENT}}": context_data.get("recipient", ""), "{{REFERENCE}}": context_data.get("reference", ""),
                "{{TITLE}}": context_data.get("title", ""), "{{RETENTION}}": context_data.get("retention", ""),
                "{{SIGN_MANAGER}}": context_data.get("sign_manager", ""), "{{SIGN_DRAFTER}}": context_data.get("sign_drafter", ""),
                "{{SIGN_COOP}}": context_data.get("sign_coop", ""), "{{DOC_NUMBER_2}}": context_data.get("doc_number_2", ""),
                "{{EXEC_DATE_2}}": context_data.get("exec_date_2", ""), "{{VIA_2}}": context_data.get("via_2", ""),
                "{{RECIPIENT_2}}": context_data.get("recipient_2", ""), "{{REFERENCE_2}}": context_data.get("reference_2", ""),
                "{{TITLE_2}}": context_data.get("title_2", ""),
            }
            for placeholder, value in field_replacements.items():
                section0_content = section0_content.replace(placeholder, str(value))

            items_data = context_data.get("items", [])
            processed_items = []
            for idx, item in enumerate(items_data):
                text = item.get('text', '')
                if idx == len(items_data) - 1 and not context_data.get('has_attachment', False): text += "  끝."
                processed_items.append({
                    "style_id": "2" if item.get('level', 1) == 1 else "1",
                    "para_pr_id": "28" if item.get('level', 1) == 1 else "29", "text": text
                })
            if not items_data and not context_data.get('has_attachment', False):
                 processed_items.append({"style_id": "2", "para_pr_id": "28", "text": "끝."})
            
            xml_template = Template("""{%- for item in items -%}<hp:p id="0" paraPrIDRef="{{ item.para_pr_id }}" styleIDRef="{{ item.style_id }}"><hp:run charPrIDRef="19"><hp:t>{{ item.text }}</hp:t></hp:run></hp:p>{%- endfor -%}""")
            new_xml_block = xml_template.render(items=processed_items)
            
            section0_content = self._replace_content_block(section0_content, "{{CONTENT}}", new_xml_block)
            section0_content = self._replace_content_block(section0_content, "{{CONTENT_2}}", new_xml_block)
            section0_content = self._adjust_empty_paragraphs(section0_content, context_data)
            
            with zipfile.ZipFile(input_template_path, 'r') as z_read:
                with zipfile.ZipFile(final_output_path, 'w') as z_write:
                    for item in z_read.infolist():
                        if item.filename == "Contents/section0.xml":
                            z_write.writestr(item, section0_content.encode('utf-8'))
                        else:
                            z_write.writestr(item, z_read.read(item.filename))
            with open(final_output_path, "rb") as f:
                return f.read()

    def _replace_content_block(self, xml_content: str, placeholder: str, new_xml_block: str) -> str:
        content_idx = xml_content.find(placeholder)
        if content_idx == -1: return xml_content
        start_idx = xml_content.rfind('<hp:p', 0, content_idx)
        end_idx = xml_content.find('</hp:p>', content_idx) + len('</hp:p>')
        if start_idx == -1 or end_idx < start_idx: return xml_content
        return xml_content[:start_idx] + new_xml_block + xml_content[end_idx:]

    def _adjust_empty_paragraphs(self, section0_content: str, context_data: dict) -> str:
        content_line_count = len(context_data.get('items', []))
        if not content_line_count:
            return section0_content

        empty_para_pattern = r'<hp:p[^>]*><hp:run[^>]*/><hp:linesegarray>.*?</hp:linesegarray></hp:p>'
        
        # --- 기안문 빈 문단 조절 ---
        section2_marker = '붙임.2'
        section2_start_idx = section0_content.find(section2_marker)
        if section2_start_idx == -1: section2_start_idx = len(section0_content)

        sign_marker = '신 사 종 합 사 회 복 지 관 장'
        coop_idx = section0_content.find('협조')
        
        # 기안문 영역 특정
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
        # 본문 위쪽
        before_section = xml_content[start_boundary:content_start]
        before_empties = list(re.finditer(pattern, before_section, re.DOTALL))
        
        # 본문 아래쪽
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
            # 아래쪽, 위쪽 번갈아가며 삭제 대상 선정
            if i % 2 == 0: # 아래쪽 먼저
                if removed_after < max_after:
                    to_delete.append(after_candidates[removed_after])
                    removed_after += 1
                elif removed_before < max_before:
                    to_delete.append(before_candidates[removed_before])
                    removed_before += 1
            else: # 위쪽
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
