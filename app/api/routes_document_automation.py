import json
import logging
import os
import re
import tempfile
import zipfile
from typing import Optional
from urllib.parse import quote
from xml.etree import ElementTree as ET

from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse

logger = logging.getLogger(__name__)

from app.services.documents.document_automater_service import document_automater_service
from app.services.documents.meeting_minutes_service import meeting_minutes_service
from app.core.config import settings
from app.services.rag.extractors import file_text_extractor
from app.services.rag.rag_ingestion_service import detect_file_type
from app.services.utils.download_service import download_service


def _extract_hwpx_text(file_path: str) -> Optional[str]:
    """HWPX(ZIP+XML) 파일에서 텍스트를 직접 추출 (한컴 오피스/win32com 불필요).

    1순위: Preview/PrvText.txt (문서 전체 평문, 표/헤더 포함)
    2순위: Contents/section*.xml의 <hp:t> 텍스트 노드
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()

            if "Preview/PrvText.txt" in names:
                raw = zf.read("Preview/PrvText.txt")
                for enc in ("utf-8", "utf-16", "utf-16-le", "cp949"):
                    try:
                        text = raw.decode(enc)
                    except UnicodeDecodeError:
                        continue
                    text = text.strip()
                    if text:
                        return text

            texts: list[str] = []
            section_names = sorted(
                n for n in names
                if n.startswith("Contents/section") and n.endswith(".xml")
            )
            for name in section_names:
                try:
                    raw = zf.read(name)
                    root = ET.fromstring(raw)
                except (KeyError, ET.ParseError):
                    continue
                for elem in root.iter():
                    tag = elem.tag.split("}", 1)[-1]
                    if tag in ("t", "char") and elem.text:
                        texts.append(elem.text)
                texts.append("\n")

        joined = "".join(texts)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r"\n{3,}", "\n\n", joined).strip()
        return joined or None
    except Exception as e:
        logger.error(f"[MeetingMinutes] HWPX direct parse failed: {e}")
        return None


router = APIRouter()

@router.post("/documents/generate-hwpx", summary="HWPX 문서 자동 생성")
async def generate_hwpx_document_api(
        template_name: str = Form(..., description="사용할 HWPX 템플릿 파일명: 'template.hwpx', 'template2.hwpx', 'template3.hwpx'"),
        context_data_str: str = Form(..., description="문서에 채워 넣을 JSON 데이터 (문자열 형태)", alias="context_data"),
        expires_in: int = Form(3600, description="다운로드 링크 만료 시간 (초 단위, 기본 1시간)"),
        one_time: bool = Form(True, description="1회용 링크 여부 (기본 True)")
):
    """
    제공된 HWPX 템플릿과 JSON 데이터를 사용하여 새로운 HWPX 문서를 생성하고,
    1회용/기간제 다운로드 링크를 반환합니다.

    - **template_name**: 사용할 HWPX 템플릿 파일명
    - **context_data**: 문서의 플레이스홀더를 채울 JSON 형식의 데이터 (템플릿별 구조는 아래 참조)
    - **expires_in**: 다운로드 링크 만료 시간 (초 단위, 기본 3600초 = 1시간)
    - **one_time**: True면 1회 다운로드 후 링크 무효화

    ---

    ### 지원 템플릿 및 context_data 구조

    #### 1. `template.hwpx` (기안문/시행문)
    ```json
    {
      "doc_number": "문서번호",
      "draft_date": "기안일",
      "exec_date": "시행일",
      "via": "경유",
      "recipient": "수신",
      "reference": "참조",
      "title": "제목",
      "retention": "보존기한",
      "sign_manager": "관리자 서명",
      "sign_drafter": "기안자 서명",
      "sign_coop": "협조자 서명",
      "doc_number_2": "시행문 문서번호",
      "exec_date_2": "시행문 시행일",
      "via_2": "시행문 경유",
      "recipient_2": "시행문 수신",
      "reference_2": "시행문 참조",
      "title_2": "시행문 제목",
      "has_attachment": false,
      "items": [
        {"text": "본문 내용", "level": 1}
      ]
    }
    ```

    #### 2. `template2.hwpx` (일반 공문)
    ```json
    {
      "recipient": "수신",
      "via": "경유",
      "title": "제목",
      "attachment": "붙임",
      "content_lines": ["본문 1줄", "본문 2줄"]
    }
    ```

    #### 3. `template3.hwpx` (회의록)
    ```json
    {
      "meeting_title": "회의 안건",
      "datetime": "일시",
      "person_in_charge": "담당자",
      "location": "장소",
      "attendee_count": "참석 인원수",
      "agenda": "주요 안건",
      "attendees": [
        {"affiliation": "소속", "name": "이름"}
      ],
      "meeting_content_lines": ["회의 내용 1줄", "회의 내용 2줄"],
      "meeting_result_lines": ["회의 결과 1줄", "회의 결과 2줄"]
    }
    ```
    *attendees는 최대 8명까지 지원

    ---

    **반환값**: {"success": bool, "download_url": str, "expires_at": str, "one_time": bool, "filename": str}
    """
    try:
        # 문자열로 받은 context_data를 JSON(딕셔너리)으로 파싱
        try:
            context_data = json.loads(context_data_str)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="'context_data' 필드가 유효한 JSON 문자열이 아닙니다.")

        # 서비스 함수 호출
        generated_hwpx_bytes = document_automater_service.generate_hwpx_document(
            template_name=template_name,
            context_data=context_data
        )

        # HWPX 파일의 MIME 타입
        media_type = "application/haansofthwpml"
        filename = f"generated_{template_name}"

        # 다운로드 링크 생성
        link_info = download_service.create_download_link(
            file_bytes=generated_hwpx_bytes,
            filename=filename,
            expires_in_seconds=expires_in,
            one_time=one_time,
            media_type=media_type
        )

        return JSONResponse(content={
            "success": True,
            "download_url": link_info["download_url"],
            "expires_at": link_info["expires_at"],
            "one_time": link_info["one_time"],
            "filename": filename
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[HWPX Generation Error] {e}")
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {e}")


@router.post("/documents/meeting-minutes/generate-from-text", summary="회의 원문 텍스트 -> AI 회의록 HWPX 생성 (SSE 진행률)")
async def generate_meeting_minutes_from_text(
        raw_text: Optional[str] = Form(None, description="회의 원문 텍스트 (붙여넣기) - 선택"),
        file: Optional[UploadFile] = File(None, description="회의 원문 파일 (pdf/hwp/hwpx/docx/txt 등). raw_text와 함께 전송 가능"),
        expires_in: int = Form(3600, description="다운로드 링크 만료 시간 (초)"),
        one_time: bool = Form(True, description="1회용 링크 여부"),
):
    """
    회의 원문 텍스트 또는 파일을 받아 LLM으로 구조화된 회의록 JSON을 추출한 뒤,
    `template3.hwpx` 템플릿으로 HWPX 파일을 생성하여 다운로드 링크를 반환합니다.

    응답은 **SSE(text/event-stream)** 형식이며, 각 이벤트는 다음 스키마의 JSON 한 줄입니다.

    ```json
    {"stage": "extract|llm|render|link|done|error", "percent": 0-100, "message": "...", "data": {...}}
    ```

    - 최종 성공 시 `stage="done"`, `percent=100`, `data`에 `download_url`, `expires_at`, `filename`, `extracted` 포함.
    - 실패 시 `stage="error"`와 `message` 전송 후 스트림 종료.

    참고: 파일 업로드 자체의 바이트 진행률은 서버가 핸들러 진입 시점에 이미 업로드 완료 상태이므로
    내보낼 수 없습니다. 업로드 진행률은 클라이언트 측(XHR/axios `onUploadProgress`)에서 표시하세요.
    """
    # 파일은 핸들러 진입 시 이미 업로드 완료 상태 — 바이트를 먼저 읽어두고
    # 이후 단계별 진행률을 SSE로 스트리밍한다.
    uploaded_suffix = ""
    uploaded_bytes: Optional[bytes] = None
    uploaded_filename: Optional[str] = None
    if file is not None and file.filename:
        uploaded_filename = file.filename
        uploaded_suffix = os.path.splitext(file.filename)[1] or ""
        uploaded_bytes = await file.read()
        if not uploaded_bytes:
            raise HTTPException(status_code=400, detail="업로드된 파일이 비어 있습니다.")

    if not (raw_text and raw_text.strip()) and not uploaded_bytes:
        raise HTTPException(status_code=400, detail="raw_text 또는 file 중 하나는 반드시 제공되어야 합니다.")

    async def event_stream():
        def _event(stage: str, percent: int, message: str = "", data: Optional[dict] = None) -> str:
            payload = {"stage": stage, "percent": percent, "message": message}
            if data is not None:
                payload["data"] = data
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        tmp_path = None
        try:
            yield _event("start", 0, "요청 수신")

            # 1단계: 파일 텍스트 추출 (0 -> 30%)
            file_text = ""
            if uploaded_bytes:
                yield _event("extract", 5, "파일 텍스트 추출 시작")
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_suffix) as tmp:
                        tmp.write(uploaded_bytes)
                        tmp_path = tmp.name

                    ext_lower = uploaded_suffix.lower()
                    if ext_lower == ".hwpx":
                        file_text = _extract_hwpx_text(tmp_path) or ""
                    if not file_text:
                        file_type = detect_file_type(uploaded_filename or "")
                        try:
                            page_results, md_content = await file_text_extractor.extract_text(tmp_path, file_type)
                        except Exception as ex_inner:
                            logger.warning(f"[MeetingMinutes] extractor failed, fallback empty: {ex_inner}")
                            page_results, md_content = None, None
                        if md_content:
                            file_text = md_content
                        elif page_results:
                            file_text = "\n".join(text for _, text in page_results)
                except Exception as e:
                    logger.exception(f"[MeetingMinutes] file extraction error: {e}")
                    yield _event("error", 0, f"파일에서 텍스트를 추출하지 못했습니다: {e}")
                    return
                yield _event("extract", 30, "파일 텍스트 추출 완료")
            else:
                yield _event("extract", 30, "파일 없음 - raw_text 사용")

            combined_parts = []
            if raw_text and raw_text.strip():
                combined_parts.append(raw_text.strip())
            if file_text and file_text.strip():
                combined_parts.append(file_text.strip())

            if not combined_parts:
                yield _event("error", 0, "추출된 텍스트가 없습니다.")
                return

            combined_text = "\n\n".join(combined_parts)

            # 2단계: LLM 회의록 JSON 추출 (30 -> 70%)
            yield _event("llm", 35, "LLM 회의록 구조화 시작")
            try:
                context_data = await meeting_minutes_service.extract_meeting_minutes_json(combined_text)
            except ValueError as e:
                yield _event("error", 35, f"회의록 추출 실패: {e}")
                return
            except Exception as e:
                logger.exception(f"[MeetingMinutes] extraction error: {e}")
                yield _event("error", 35, f"회의록 추출 중 서버 오류: {e}")
                return
            yield _event("llm", 70, "LLM 회의록 구조화 완료")

            # 3단계: HWPX 렌더링 (70 -> 90%)
            yield _event("render", 75, "HWPX 템플릿 렌더링 시작")
            try:
                generated_hwpx_bytes = document_automater_service.generate_hwpx_document(
                    template_name="template3.hwpx",
                    context_data=context_data,
                )
            except FileNotFoundError as e:
                yield _event("error", 75, f"템플릿을 찾을 수 없습니다: {e}")
                return
            except ValueError as e:
                yield _event("error", 75, f"렌더링 실패: {e}")
                return
            except Exception as e:
                logger.exception(f"[MeetingMinutes] HWPX generation error: {e}")
                yield _event("error", 75, f"서버 내부 오류: {e}")
                return
            yield _event("render", 90, "HWPX 렌더링 완료")

            # 4단계: 다운로드 링크 생성 (90 -> 100%)
            yield _event("link", 95, "다운로드 링크 생성")
            media_type = "application/haansofthwpml"
            filename = "generated_meeting_minutes.hwpx"
            link_info = download_service.create_download_link(
                file_bytes=generated_hwpx_bytes,
                filename=filename,
                expires_in_seconds=expires_in,
                one_time=one_time,
                media_type=media_type,
            )

            yield _event(
                "done",
                100,
                "완료",
                data={
                    "success": True,
                    "download_url": link_info["download_url"],
                    "expires_at": link_info["expires_at"],
                    "one_time": link_info["one_time"],
                    "filename": filename,
                    "extracted": context_data,
                },
            )
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/documents/download/{token}", summary="토큰 기반 파일 다운로드")
async def download_document(token: str):
    """
    생성된 토큰을 사용하여 문서를 다운로드합니다.

    - **token**: 문서 생성 시 반환된 다운로드 토큰
    - 1회용 링크인 경우 다운로드 후 링크가 무효화됩니다.
    - 만료된 링크는 사용할 수 없습니다.
    """
    # 토큰으로 파일 정보 조회
    file_info = download_service.get_file_info(token)

    if not file_info:
        raise HTTPException(
            status_code=404,
            detail="다운로드 링크가 만료되었거나 유효하지 않습니다."
        )

    file_path = file_info["file_path"]
    filename = file_info["filename"]
    media_type = file_info["media_type"]
    is_one_time = file_info.get("one_time", True)

    # 파일 존재 확인
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    # 파일 읽기
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # 1회용 링크면 파일 삭제 (참조 파일은 삭제하지 않음)
    is_reference = file_info.get("is_reference", False)
    if is_one_time and not is_reference:
        download_service.delete_file(file_path)

    return StreamingResponse(
        content=iter([file_bytes]),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"}
    )
