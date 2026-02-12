import logging
from typing import Dict, Any

from .processors.template1_processor import Template1Processor
from .processors.template2_processor import Template2Processor
from .processors.template3_processor import Template3Processor

logger = logging.getLogger(__name__)

class DocumentAutomaterService:
    def __init__(self):
        """
        지원하는 템플릿 처리기를 등록합니다.
        """
        self._processors = {
            "template.hwpx": Template1Processor(),
            "template2.hwpx": Template2Processor(),
            "template3.hwpx": Template3Processor(),
        }
        logger.info(f"문서 자동화 처리기가 등록되었습니다: {list(self._processors.keys())}")

    def generate_hwpx_document(self, template_name: str, context_data: dict) -> bytes:
        """
        템플릿 이름에 맞는 처리기를 찾아 문서 생성을 위임합니다.
        
        :param template_name: 사용할 HWPX 템플릿 파일명
        :param context_data: 문서에 채워 넣을 데이터
        :return: 생성된 HWPX 문서의 바이트 내용
        """
        if template_name not in self._processors:
            logger.error(f"지원하지 않는 템플릿입니다: {template_name}")
            raise FileNotFoundError(f"지원하지 않는 템플릿이거나, 해당 템플릿의 처리기가 등록되지 않았습니다: {template_name}")

        processor = self._processors[template_name]
        logger.info(f"'{template_name}'에 대한 생성을 '{processor.__class__.__name__}'로 시작합니다.")
        
        # 해당 프로세서의 generate 메서드 호출
        return processor.generate(context_data)

# 싱글톤 인스턴스 생성
document_automater_service = DocumentAutomaterService()
