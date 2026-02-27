"""
파일 텍스트 추출 서비스

포맷별(PDF / Polaris / MarkItDown) 텍스트 추출 로직을 담당합니다.
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Generator, List, Optional, Tuple

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import pdf4llm
except ImportError:
    pdf4llm = None

try:
    import win32com.client
    import pythoncom
except ImportError:
    win32com = None
    pythoncom = None

logger = logging.getLogger(__name__)

# PDF 처리용 프로세스 풀 (8 workers)
_pdf_process_pool: Optional[ProcessPoolExecutor] = None


def get_pdf_process_pool() -> ProcessPoolExecutor:
    """PDF 처리용 프로세스 풀 (lazy init)"""
    global _pdf_process_pool
    if _pdf_process_pool is None:
        _pdf_process_pool = ProcessPoolExecutor(max_workers=8)
    return _pdf_process_pool


# ----------------------------------------
# 멀티프로세싱용 모듈 레벨 함수 (ProcessPoolExecutor pickle 필요)
# ----------------------------------------

def _parse_pdf_in_process(pdf_path: str) -> List[Tuple[int, str]]:
    """별도 프로세스에서 PDF 파싱 (멀티프로세싱용)

    Returns: [(page_num, content), ...]
    """
    try:
        import pdf4llm as _pdf4llm
        pages_data = _pdf4llm.to_markdown(pdf_path, page_chunks=True)
        results = []
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
            if content_text.strip():
                results.append((page_num, content_text.strip()))
        return results
    except Exception:
        return []


def _convert_polaris_in_process(file_path: str, temp_output_dir: str) -> Optional[str]:
    """별도 프로세스에서 Polaris 변환 (멀티프로세싱용)

    Returns: markdown 문자열 또는 None
    """
    try:
        from app.services.api_clients.polaris_client import polaris_converter, PolarisJsonParser
        polaris_data = polaris_converter.convert(file_path, temp_output_dir, True)
        if polaris_data:
            parser = PolarisJsonParser(polaris_data)
            return parser.parse_to_markdown()
        return None
    except Exception:
        return None


class FileTextExtractor:
    """파일에서 텍스트를 추출하는 서비스"""

    def __init__(self):
        self._markitdown = MarkItDown() if MarkItDown else None

    # ----------------------------------------
    # 동기 추출 (asyncio.to_thread 용)
    # ----------------------------------------

    def iter_pdf_pages(self, pdf_path: str) -> Generator[Tuple[int, str], None, None]:
        """PDF를 페이지 단위로 yield하는 제너레이터"""
        if pdf4llm is None:
            logger.error("[Extractor] pdf4llm 라이브러리가 설치되지 않았습니다.")
            return
        try:
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            for page_idx, page_data in enumerate(pages_data):
                page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                if content_text.strip():
                    yield page_num, content_text.strip()
        except Exception as e:
            logger.exception(f"[Extractor] pdf4llm 변환 중 예외 발생: {e}")

    def _try_hwp_save(self, hwp, output_pdf_path: str) -> bool:
        """HWP → PDF 저장을 3가지 방법으로 순차 시도. 성공 시 True 반환."""
        try:
            hwp.HAction.GetDefault("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
            hwp.HParameterSet.HFileOpenSave.filename = output_pdf_path
            hwp.HParameterSet.HFileOpenSave.Format = "PDF"
            return hwp.HAction.Execute("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
        except Exception as e1:
            logger.warning(f"[Extractor] HWP 저장 방법 1 (FileSaveAsPdf) 실패: {e1}")

        try:
            hwp.SaveAs(output_pdf_path, "PDF")
            return os.path.exists(output_pdf_path)
        except Exception as e2:
            logger.warning(f"[Extractor] HWP 저장 방법 2 (SaveAs) 실패: {e2}")

        try:
            hwp.XHwpDocuments.Active.SaveAs(output_pdf_path, "PDF", "")
            return os.path.exists(output_pdf_path)
        except Exception as e3:
            logger.warning(f"[Extractor] HWP 저장 방법 3 (XHwpDocuments) 실패: {e3}")

        return False

    def convert_hwp_to_pdf(self, input_path: str, output_dir: str) -> Optional[str]:
        """win32com을 사용하여 HWP 파일을 PDF로 변환"""
        if not win32com or not pythoncom:
            logger.error("[Extractor] win32com 또는 pythoncom 라이브러리가 없습니다.")
            return None

        input_abs_path = os.path.abspath(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_pdf_path = os.path.join(os.path.abspath(output_dir), f"{base_name}.pdf")

        hwp = None
        com_initialized = False
        try:
            pythoncom.CoInitialize()
            com_initialized = True

            try:
                hwp = win32com.client.gencache.EnsureDispatch("HWPFrame.HwpObject")
            except Exception:
                hwp = win32com.client.Dispatch("HWPFrame.HwpObject")

            hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
            hwp.Open(input_abs_path, "HWP", "forceopen:true")

            success = self._try_hwp_save(hwp, output_pdf_path)

            if success and os.path.exists(output_pdf_path):
                logger.info(f"[Extractor] HWP -> PDF 변환 성공: {output_pdf_path}")
                return output_pdf_path

            logger.error("[Extractor] HWP -> PDF 변환: 모든 방법 실패")
            return None

        except Exception as e:
            logger.exception(f"[Extractor] win32com HWP 변환 중 오류 발생: {e}")
            return None
        finally:
            if hwp:
                try:
                    hwp.Quit()
                except Exception:
                    pass
            if com_initialized:
                pythoncom.CoUninitialize()

    def convert_sync(self, file_path: str) -> Optional[str]:
        """MarkItDown 동기 변환 (asyncio.to_thread 사용 시)"""
        if self._markitdown is None:
            return None
        try:
            result = self._markitdown.convert(file_path)
            return result.text_content if result and result.text_content else None
        except Exception as e:
            logger.error(f"[Extractor] MarkItDown 변환 오류: {e}")
            return None

    # ----------------------------------------
    # 비동기 추출
    # ----------------------------------------

    async def extract_with_polaris(self, file_path: str, on_progress=None) -> Optional[str]:
        """Polaris 변환으로 텍스트 추출 (임시 디렉토리 자체 관리)"""
        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")
        temp_output_dir = tempfile.mkdtemp()
        try:
            loop = asyncio.get_event_loop()
            parse_start = time.time()
            markdown_content = await loop.run_in_executor(
                get_pdf_process_pool(),
                _convert_polaris_in_process,
                file_path,
                temp_output_dir
            )
            parse_elapsed = time.time() - parse_start
            if not markdown_content:
                logger.error("[Extractor] Polaris 변환 실패")
                return None
            logger.info(f"[Extractor] Polaris 텍스트 추출 완료: {len(markdown_content)}자, {parse_elapsed:.2f}초")
            return markdown_content
        finally:
            shutil.rmtree(temp_output_dir, ignore_errors=True)

    async def extract_pdf_pages(self, file_path: str, on_progress=None) -> List[Tuple[int, str]]:
        """PDF 멀티프로세싱 페이지별 텍스트 추출"""
        if on_progress:
            await on_progress(10, "페이지 파싱 중...")
        loop = asyncio.get_event_loop()
        parse_start = time.time()
        page_results = await loop.run_in_executor(
            get_pdf_process_pool(),
            _parse_pdf_in_process,
            file_path
        )
        parse_elapsed = time.time() - parse_start
        logger.info(f"[Extractor] PDF 텍스트 추출 완료: {len(page_results)}페이지, {parse_elapsed:.2f}초")
        return page_results

    async def extract_with_markitdown(self, file_path: str, on_progress=None) -> Optional[str]:
        """MarkItDown으로 텍스트 추출"""
        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")
        if self._markitdown is None:
            logger.error("[Extractor] MarkItDown 미설치")
            return None
        try:
            parse_start = time.time()
            result = await asyncio.to_thread(self._markitdown.convert, file_path)
            parse_elapsed = time.time() - parse_start
            if result and result.text_content:
                logger.info(f"[Extractor] MarkItDown 텍스트 추출 완료: {len(result.text_content)}자, {parse_elapsed:.2f}초")
                return result.text_content
            return None
        except Exception as e:
            logger.error(f"[Extractor] MarkItDown 오류: {e}")
            return None

    async def extract_with_win32hwp(self, file_path: str, on_progress=None) -> Optional[List[Tuple[int, str]]]:
        """HWP → PDF (win32com) → 텍스트 추출"""
        if on_progress:
            await on_progress(5, "HWP 문서 변환 중...")
        temp_dir = tempfile.mkdtemp()
        try:
            loop = asyncio.get_event_loop()
            pdf_path = await loop.run_in_executor(
                None,  # thread pool (COM 객체는 프로세스 풀 불가)
                self.convert_hwp_to_pdf,
                file_path,
                temp_dir
            )
            if not pdf_path:
                logger.error("[Extractor] HWP → PDF 변환 실패")
                return None
            return await self.extract_pdf_pages(pdf_path, on_progress)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def extract_text(
        self, file_path: str, file_type: str, on_progress=None
    ) -> Tuple[Optional[List[Tuple[int, str]]], Optional[str]]:
        """파일 타입별 추출 함수로 분기

        Returns:
            (page_results, markdown_content)
            - pdf:    ([(page_num, text), ...], None)
            - 나머지: (None, "markdown string")
        """
        if file_type == "pdf":
            return await self.extract_pdf_pages(file_path, on_progress), None
        if file_type == "polaris":
            return None, await self.extract_with_polaris(file_path, on_progress)
        if file_type == "hwp_win32":
            return await self.extract_with_win32hwp(file_path, on_progress), None
        return None, await self.extract_with_markitdown(file_path, on_progress)


# 싱글톤 인스턴스
file_text_extractor = FileTextExtractor()
