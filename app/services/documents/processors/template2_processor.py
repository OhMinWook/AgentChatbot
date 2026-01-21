import zipfile
import os
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "documents"

class Template2Processor:
    """'template2.hwpx' 전용 처리기"""

    def generate(self, context_data: dict) -> bytes:
        template_name = "template2.hwpx"
        input_template_path = TEMPLATE_DIR / template_name
        if not input_template_path.exists():
            raise FileNotFoundError(f"템플릿 파일 '{template_name}'을 찾을 수 없습니다.")

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 압축 해제
            with zipfile.ZipFile(input_template_path, 'r') as z:
                z.extractall(tmpdir)

            # 2. section0.xml 수정
            section_path = os.path.join(tmpdir, "Contents", "section0.xml")
            with open(section_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # --- 단순 플레이스홀더 교체 ---
            simple_replacements = {
                "{{RECIPIENT}}": context_data.get("recipient", ""),
                "{{VIA}}": context_data.get("via", ""),
                "{{TITLE}}": context_data.get("title", ""),
                "{{ATTACHMENT}}": context_data.get("attachment", ""),
            }
            for placeholder, value in simple_replacements.items():
                content = content.replace(placeholder, str(value))
            
            # --- 본문 교체 ---
            content_marker = "{{CONTENT}}"
            if content_marker in content:
                p_start = content.rfind('<hp:p ', 0, content.find(content_marker))
                p_end = content.find('</hp:p>', content.find(content_marker)) + len('</hp:p>')
                
                def make_paragraph(text):
                    # 원본 스크립트의 하드코딩된 XML 구조를 그대로 사용
                    return f'<hp:p id="2147483648" paraPrIDRef="0" styleIDRef="0"><hp:run charPrIDRef="9"><hp:t>{text}</hp:t></hp:run><hp:linesegarray><hp:lineseg textpos="0" vertpos="0" vertsize="1200" textheight="1200" baseline="1020" spacing="720" horzpos="0" horzsize="49608" flags="393216"/></hp:linesegarray></hp:p>'
                
                new_paragraphs = ''.join(make_paragraph(line) for line in context_data.get("content_lines", []))
                
                if p_start > -1 and p_end > -1:
                    content = content[:p_start] + new_paragraphs + content[p_end:]
            
            with open(section_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # 3. 새 HWPX 파일로 다시 압축
            output_path = Path(tmpdir) / "result_new.hwpx"
            with zipfile.ZipFile(output_path, 'w') as z_new:
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, tmpdir)
                        z_new.write(file_path, arcname)

            # 4. 생성된 파일의 바이트를 읽어 반환
            with open(output_path, "rb") as f:
                return f.read()
