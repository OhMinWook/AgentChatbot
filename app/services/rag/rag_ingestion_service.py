import asyncio
import tempfile
import logging
import os
import re
import hashlib
from typing import List, Dict, Any, Optional

# Polaris 비활성화 시 사용할 대체 라이브러리들
try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

try:
    import pdf4llm
except ImportError:
    pdf4llm = None

# Windows 환경에서 HWP 제어를 위한 win32com
try:
    import win32com.client
    import pythoncom
except ImportError:
    win32com = None
    pythoncom = None

from app.core.config import settings
from app.services.clients.model_server_client import model_server_client
from app.services.clients.polaris_client import polaris_converter, PolarisJsonParser
from app.services.rag.local_index_service import local_index_service
from app.services.rag.lightrag_service import lightrag_service

logger = logging.getLogger(__name__)

# 청크 설정 (512 토큰 ≈ 1500자 한국어 기준)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


class RagIngestionService:
    def __init__(self):
        self._markitdown = MarkItDown() if MarkItDown else None

    async def _create_chunks_with_keywords(
        self,
        markdown_content: str,
        file_name: str
    ) -> List[Dict]:
        """
        마크다운을 512토큰 청크로 분할하고 키워드를 추출합니다.

        Returns:
            [{"id": "...", "content": "[문서:...][키워드:...]\n본문", "metadata": {...}}, ...]
        """
        if not markdown_content.strip():
            return []

        # 1. 페이지별 텍스트 파싱
        page_pattern = re.compile(r'\n--- Page (\d+) ---\n')
        parts = page_pattern.split(markdown_content)

        page_texts = []  # [(page_num, text), ...]
        if parts[0].strip():
            page_texts.append((1, parts[0].strip()))

        for i in range(1, len(parts), 2):
            try:
                page_num = int(parts[i])
                content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                if content:
                    page_texts.append((page_num, content))
            except (IndexError, ValueError):
                continue

        if not page_texts:
            return []

        # 2. 청크 분할 (페이지 정보 유지)
        raw_chunks = []  # [{"text": "...", "page": N}, ...]

        for page_num, page_text in page_texts:
            start = 0
            text_len = len(page_text)

            while start < text_len:
                end = start + CHUNK_SIZE
                chunk_text = page_text[start:end]

                # 단어 중간에서 자르지 않기
                if end < text_len:
                    last_space = chunk_text.rfind(' ')
                    last_newline = chunk_text.rfind('\n')
                    cut_point = max(last_space, last_newline)
                    if cut_point > CHUNK_SIZE * 0.5:
                        chunk_text = chunk_text[:cut_point]
                        end = start + cut_point

                if chunk_text.strip():
                    raw_chunks.append({
                        "text": chunk_text.strip(),
                        "page": page_num
                    })

                start = end - CHUNK_OVERLAP if end < text_len else text_len

        if not raw_chunks:
            return []

        logger.info(f"[Chunking] {file_name}: {len(raw_chunks)}개 청크 생성")

        # 3. 키워드 추출 (배치)
        chunk_texts = [c["text"] for c in raw_chunks]

        try:
            keywords_list = await model_server_client.extract_keywords_batch(chunk_texts)
            logger.info(f"[Keywords] {len(keywords_list)}개 키워드 추출 완료")
        except Exception as e:
            logger.warning(f"[Keywords] 키워드 추출 실패, 빈 키워드 사용: {e}")
            keywords_list = [""] * len(raw_chunks)

        # 4. 최종 청크 포맷팅
        final_chunks = []
        for i, chunk in enumerate(raw_chunks):
            keywords = keywords_list[i] if i < len(keywords_list) else ""
            page = chunk["page"]
            text = chunk["text"]

            # 포맷: [문서: 파일명 | 페이지: N][키워드: ...]\n본문
            if keywords:
                content = f"[문서: {file_name} | 페이지: {page}]\n[키워드: {keywords}]\n\n{text}"
            else:
                content = f"[문서: {file_name} | 페이지: {page}]\n\n{text}"

            # 콘텐츠 기반 ID 생성 (중복 방지)
            # 파일명 + 페이지 + 텍스트 해시 조합
            content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
            chunk_id = f"{file_name}_{page}_{content_hash}"

            final_chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source": file_name,
                    "page": page,
                    "keywords": keywords
                }
            })

        return final_chunks

    def _process_pdf_with_pdf4llm(self, pdf_path: str, display_name: str, invoke_id: str) -> str:
        """pdf4llm을 사용하여 PDF를 마크다운으로 변환합니다. (페이지 구분자 포함)"""
        if pdf4llm is None:
            logger.error("pdf4llm 라이브러리가 설치되지 않았습니다.")
            return ""

        try:
            pages_data = pdf4llm.to_markdown(pdf_path, page_chunks=True)
            markdown_parts = []
            for page_idx, page_data in enumerate(pages_data):
                real_page_num = page_idx + 1
                content_text = page_data if isinstance(page_data, str) else page_data.get('text', '')
                if content_text.strip():
                    markdown_parts.append(f"\n--- Page {real_page_num} ---\n{content_text.strip()}")

            full_markdown = "\n".join(markdown_parts)
            logger.info(f"[PDF] {display_name}: {len(full_markdown)}자, {len(markdown_parts)}페이지")
            return full_markdown
        except Exception as e:
            logger.exception(f"pdf4llm 변환 중 예외 발생: {e}")
            return ""

    def _convert_hwp_to_pdf_with_win32com(self, input_path: str, output_dir: str) -> Optional[str]:
        """win32com을 사용하여 한글(HWP) 파일을 PDF로 변환합니다."""
        if not win32com or not pythoncom:
            logger.error("win32com 또는 pythoncom 라이브러리가 없습니다.")
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

            success = False
            try:
                hwp.HAction.GetDefault("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
                hwp.HParameterSet.HFileOpenSave.filename = output_pdf_path
                hwp.HParameterSet.HFileOpenSave.Format = "PDF"
                success = hwp.HAction.Execute("FileSaveAsPdf", hwp.HParameterSet.HFileOpenSave.HSet)
            except Exception as e1:
                logger.warning(f"방법 1 (FileSaveAsPdf) 실패: {e1}")
                try:
                    hwp.SaveAs(output_pdf_path, "PDF")
                    success = os.path.exists(output_pdf_path)
                except Exception as e2:
                    logger.warning(f"방법 2 (SaveAs) 실패: {e2}")
                    try:
                        hwp.XHwpDocuments.Active.SaveAs(output_pdf_path, "PDF", "")
                        success = os.path.exists(output_pdf_path)
                    except Exception as e3:
                        logger.warning(f"방법 3 (XHwpDocuments) 실패: {e3}")

            if success and os.path.exists(output_pdf_path):
                logger.info(f"HWP -> PDF 변환 성공: {output_pdf_path}")
                return output_pdf_path
            else:
                logger.error("HWP -> PDF 변환: 모든 방법 실패")
                return None

        except Exception as e:
            logger.exception(f"win32com HWP 변환 중 오류 발생: {e}")
            return None
        finally:
            if hwp:
                try:
                    hwp.Quit()
                except Exception:
                    pass
            if com_initialized:
                pythoncom.CoUninitialize()

    async def _store_chunks_to_colbert(self, chunks: List[Dict], invoke_id: str):
        """
        청크 목록을 로컬 ColBERT 인덱스에 저장합니다.
        - 모델 서버: 인코딩만 담당 (Stateless)
        - 게이트웨이: Voyager + Redis에 저장
        """
        if not chunks:
            return

        total_chunks = len(chunks)
        logger.info(f"[Ingestion] ColBERT에 저장할 총 청크 수: {total_chunks}")

        # local_index_service가 내부적으로 배치 처리
        indexed_count = await local_index_service.index_documents(invoke_id, chunks)
        logger.info(f"[Ingestion] ColBERT 인덱싱 완료: {indexed_count} / {total_chunks} 청크")

    async def _store_chunks_to_lightrag(self, chunks: List[Dict], invoke_id: str):
        """
        청크 목록을 LightRAG 지식 그래프에 저장합니다.
        (백그라운드에서 실행됨)
        """
        if not chunks:
            return

        try:
            logger.info(f"🌿 [Ingestion] LightRAG 인덱싱 시작 (백그라운드)...")
            await lightrag_service.index_chunks(chunks, invoke_id)
        except Exception as e:
            logger.error(f"🌿 [Ingestion] LightRAG 인덱싱 실패: {e}")

    async def ingest_file(self, file_name: str, invoke_id: str, file_path: str, on_progress=None):
        """
        파일을 처리하고, 결과를 ColBERT 및 LightRAG 인덱스에 저장합니다.
        
        Args:
            file_name: 파일명 (확장자 포함, 예: 'report.pdf')
            invoke_id: 세션 ID
            file_path: 파일 경로
            on_progress: 진행률 콜백 (async def func(percent, message))
        """
        logger.info(f"[Ingestion] 파일 처리 시작: {file_name} (Room: {invoke_id})")

        if on_progress:
            await on_progress(0, "파일 처리 시작")

        # 파일명 자체가 display_name (확장자 포함됨)
        display_name = file_name
        
        # 확장자 추출
        _, ext = os.path.splitext(display_name)
        ext_to_use = ext.lower().strip()

        markdown_content = ""

        # === 1. 문서 파싱 (0% -> 10%) ===
        if on_progress:
            await on_progress(5, "문서 내용 추출 중...")

        if settings.POLARIS_ENABLED:
            logger.info("[Ingestion] Polaris 엔진 사용")
            with tempfile.TemporaryDirectory() as temp_output_dir:
                try:
                    polaris_data = await asyncio.to_thread(
                        polaris_converter.convert, file_path, temp_output_dir, True
                    )
                    if polaris_data:
                        parser = PolarisJsonParser(polaris_data)
                        markdown_content = parser.parse_to_markdown()
                        logger.info(f"[Polaris] {display_name}: {len(markdown_content)}자")
                    else:
                        logger.error("[Ingestion] Polaris 변환 실패")
                        return
                except Exception as e:
                    logger.error(f"[Ingestion] Polaris 오류: {e}")
                    return
        else:
            logger.info(f"[Ingestion] 확장자: '{ext_to_use}' (파일명: {display_name})")

            # 1. HWP/HWPX 파일 처리
            if ext_to_use in ['.hwp', '.hwpx']:
                logger.info("[Ingestion] 한글 문서 -> PDF 변환 -> 마크다운 추출")
                with tempfile.TemporaryDirectory() as temp_pdf_dir:
                    converted_pdf = await asyncio.to_thread(
                        self._convert_hwp_to_pdf_with_win32com, file_path, temp_pdf_dir
                    )
                    if converted_pdf and os.path.exists(converted_pdf):
                        markdown_content = await asyncio.to_thread(
                            self._process_pdf_with_pdf4llm, converted_pdf, display_name, invoke_id
                        )
                    else:
                        logger.error("[Ingestion] HWP -> PDF 변환 실패")
                        return

            # 2. PDF 파일
            elif ext_to_use == '.pdf':
                logger.info("[Ingestion] PDF -> 마크다운 추출")
                markdown_content = await asyncio.to_thread(
                    self._process_pdf_with_pdf4llm, file_path, display_name, invoke_id
                )

            # 3. 기타 문서 (MarkItDown)
            else:
                logger.info("[Ingestion] 일반 문서 -> MarkItDown 변환")
                if self._markitdown is None:
                    logger.error("[Ingestion] MarkItDown 미설치")
                    return

                try:
                    result = await asyncio.to_thread(self._markitdown.convert, file_path)
                    if result and result.text_content:
                        markdown_content = result.text_content
                except Exception as e:
                    logger.error(f"[Ingestion] MarkItDown 오류: {e}")
                    return
        
        if on_progress:
            await on_progress(10, "문서 파싱 완료")

        # 최종 체크
        if not markdown_content:
            logger.error("[Ingestion] 마크다운 없음")
            return

        # === 2. 청킹 및 키워드 추출 (10% -> 20%) ===
        if on_progress:
            await on_progress(15, "텍스트 청킹 및 키워드 추출 중...")

        # 키워드 enrichment 청킹
        chunks = await self._create_chunks_with_keywords(markdown_content, display_name)

        if not chunks:
            logger.error("[Ingestion] 청크 생성 실패")
            return
            
        if on_progress:
            await on_progress(20, f"청크 생성 완료 ({len(chunks)}개). 인덱싱 시작...")

        # === 3. 병렬 인덱싱 (ColBERT Only - LightRAG 잠시 비활성화) ===
        
        async def run_colbert():
            try:
                if on_progress:
                    await on_progress(30, "ColBERT 인덱싱 저장 중...")

                # 실제 인덱싱을 별도 태스크로 실행하면서 4초마다 5%씩 진행률 표시
                indexing_task = asyncio.create_task(
                    self._store_chunks_to_colbert(chunks, invoke_id)
                )
                current_percent = 35
                while not indexing_task.done() and current_percent <= 85:
                    await asyncio.sleep(4)
                    if indexing_task.done():
                        break
                    if on_progress:
                        await on_progress(current_percent, "ColBERT 인덱싱 저장 중...")
                    current_percent += 5

                await indexing_task  # 예외 전파
                logger.info(f"✅ [Ingestion] ColBERT 인덱싱 완료")
                if on_progress:
                    await on_progress(90, "ColBERT 인덱싱 완료")
            except Exception as e:
                logger.error(f"❌ [Ingestion] ColBERT 인덱싱 실패: {e}")

        # LightRAG는 현재 비활성화 (필요 시 주석 해제)
        """
        async def run_lightrag():
            # LightRAG 내부 진행률(0~100)을 전체 공정(20~99)으로 매핑
            async def lightrag_progress_adapter(p, msg):
                if on_progress:
                    # 20 + (p * 0.79) -> 약 20%에서 99%까지
                    mapped_percent = 20 + int(p * 0.79)
                    await on_progress(mapped_percent, msg)

            try:
                logger.info(f"🌿 [Ingestion] LightRAG 인덱싱 시작...")
                await lightrag_service.index_chunks(chunks, invoke_id, on_progress=lightrag_progress_adapter)
                logger.info(f"✅ [Ingestion] LightRAG 인덱싱 완료")
            except Exception as e:
                logger.error(f"❌ [Ingestion] LightRAG 인덱싱 실패: {e}")
        """

        # ColBERT만 실행
        await run_colbert()

        # === 4. 완료 (100%) ===
        if on_progress:
            await on_progress(100, "모든 인덱싱 작업 완료!")
            
        logger.info(f"[Ingestion] {file_name} 완료 ({len(chunks)}개 청크)")


# 싱글톤 인스턴스
rag_ingestion_service = RagIngestionService()
