import zipfile
import tempfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "documents"


class BaseTemplateProcessor(ABC):
    """HWPX 템플릿 처리기의 공통 로직을 제공하는 베이스 클래스"""

    @abstractmethod
    def generate(self, context_data: dict) -> bytes:
        pass

    def _load_section0(self, template_name: str) -> tuple:
        """템플릿 파일 검증 후 section0.xml 내용을 반환합니다."""
        template_path = TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"템플릿 파일 '{template_name}'을 찾을 수 없습니다.")

        with zipfile.ZipFile(template_path, 'r') as z:
            section0 = z.read("Contents/section0.xml").decode('utf-8')

        return template_path, section0

    def _replace_fields(self, xml: str, replacements: dict) -> str:
        """단순 플레이스홀더를 값으로 교체합니다."""
        for placeholder, value in replacements.items():
            xml = xml.replace(placeholder, str(value))
        return xml

    def _replace_content_block(self, xml: str, placeholder: str, new_xml_block: str) -> str:
        """플레이스홀더가 포함된 <hp:p> 블록을 새 XML로 교체합니다."""
        idx = xml.find(placeholder)
        if idx == -1:
            return xml

        p_start = xml.rfind('<hp:p', 0, idx)
        p_end = xml.find('</hp:p>', idx) + len('</hp:p>')

        if p_start == -1 or p_end < p_start:
            return xml

        return xml[:p_start] + new_xml_block + xml[p_end:]

    def _repackage(self, template_path: Path, section0: str) -> bytes:
        """수정된 section0.xml로 HWPX를 재패키징하여 바이트로 반환합니다."""
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "output.hwpx"
            with zipfile.ZipFile(template_path, 'r') as z_read:
                with zipfile.ZipFile(output_path, 'w') as z_write:
                    for item in z_read.infolist():
                        if item.filename == "Contents/section0.xml":
                            z_write.writestr(item, section0.encode('utf-8'))
                        else:
                            z_write.writestr(item, z_read.read(item.filename))

            with open(output_path, "rb") as f:
                return f.read()
