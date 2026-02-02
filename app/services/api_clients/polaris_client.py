import subprocess
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List

# 로깅 설정
logger = logging.getLogger(__name__)


class PolarisConverter:
    """
    Java 기반의 'PolarisDataInsight.jar'를 사용하여 다양한 문서(hwp, docx 등)를
    구조화된 JSON 데이터로 변환하는 Python Wrapper 클래스.
    """

    # 문서 타입별 출력 파일명 패턴
    OUTPUT_FILE_PATTERNS = [
        "word_page_all.json",   # Word, HWP
        "pdf_page_all.json",    # PDF
        "slide_page_all.json",  # PPT
        "sheet_page_all.json",  # Excel
    ]

    def __init__(self):
        """
        PolarisConverter를 초기화하고, 필요한 파일의 존재 여부를 확인합니다.
        """
        project_root = Path(__file__).parent.parent.parent.parent
        self.jar_dir = project_root / "libs" / "polaris"
        self.jar_path = self.jar_dir / "PolarisDataInsight.jar"
        self.license_path = self.jar_dir / "poclicense.dat"

        logger.info(f"Polaris JAR 디렉토리: {self.jar_dir}")

        self._validate_dependencies()

    def _validate_dependencies(self):
        """
        라이브러리 실행에 필요한 파일들이 모두 존재하는지 검증합니다.
        """
        if not self.jar_path.is_file():
            logger.error(f"Polaris JAR 파일을 찾을 수 없습니다: {self.jar_path}")
            raise FileNotFoundError(f"Polaris JAR file not found at: {self.jar_path}")

        if not self.license_path.is_file():
            logger.error(f"라이선스 파일을 찾을 수 없습니다: {self.license_path}")
            raise FileNotFoundError(f"License file not found at: {self.license_path}")

        logger.info("PolarisDataInsight 라이브러리 종속성 확인 완료.")

    def _find_output_json(self, output_path: Path) -> Optional[Path]:
        """
        출력 디렉토리에서 생성된 JSON 파일을 찾습니다.
        문서 타입에 따라 다른 파일명이 생성될 수 있으므로 여러 패턴을 시도합니다.
        """
        for pattern in self.OUTPUT_FILE_PATTERNS:
            json_path = output_path / pattern
            if json_path.is_file():
                return json_path

        # 패턴에 없는 경우 *_page_all.json 형태로 검색
        for json_file in output_path.glob("*_page_all.json"):
            return json_file

        return None

    def convert(self, file_path: str, output_dir: str, keep_json: bool = False) -> Optional[Dict[str, Any]]:
        """
        주어진 파일을 구조화된 JSON으로 변환합니다.

        :param file_path: 변환할 원본 파일의 절대 경로.
        :param output_dir: 결과 JSON 파일이 저장될 폴더 경로.
        :param keep_json: True면 생성된 JSON 파일을 삭제하지 않고 유지 (디버그용)
        :return: 변환 성공 시 JSON 데이터를 담은 딕셔너리, 실패 시 None.
        """
        source_file = Path(file_path)
        output_path = Path(output_dir)

        if not source_file.is_file():
            logger.error(f"변환할 원본 파일을 찾을 수 없습니다: {file_path}")
            return None

        # 출력 폴더가 없으면 생성
        output_path.mkdir(parents=True, exist_ok=True)

        # 동시성 처리를 위한 고유한 임시 디렉토리 생성
        with tempfile.TemporaryDirectory(prefix="polaris_temp_") as temp_dir:
            command = [
                "java",
                "-Xms512m",   # 초기 힙 메모리 512MB
                "-Xmx4g",     # 최대 힙 메모리 4GB (큰 문서 처리용)
                "-jar",
                str(self.jar_path),
                "DATA",
                str(source_file.resolve()),
                str(output_path.resolve()),
                temp_dir
            ]

            logger.info(f"실행 명령어: {' '.join(command)}")
            logger.info(f"Working Directory: {self.jar_dir}")

            try:
                # jar 파일이 있는 디렉토리를 CWD로 설정하여 실행
                # Windows 한국어 환경에서는 CP949로 출력될 수 있으므로 바이트로 받아서 처리
                result = subprocess.run(
                    command,
                    cwd=self.jar_dir,
                    capture_output=True,
                    check=False,
                    timeout=300  # 5분 타임아웃
                )

                # stdout 디코딩 (UTF-8 시도 후 실패하면 CP949)
                try:
                    stdout_text = result.stdout.decode('utf-8') if result.stdout else ""
                except UnicodeDecodeError:
                    stdout_text = result.stdout.decode('cp949', errors='replace') if result.stdout else ""

                try:
                    stderr_text = result.stderr.decode('utf-8') if result.stderr else ""
                except UnicodeDecodeError:
                    stderr_text = result.stderr.decode('cp949', errors='replace') if result.stderr else ""

                # 성공 여부 확인 (JAR 버전에 따라 출력 형식이 다를 수 있음)
                is_success = "DATA Success:true" in stdout_text or "bSuccess : true" in stdout_text
                if is_success:
                    logger.info(f"파일 변환 성공: {source_file.name}")

                    # 동적으로 출력 파일 찾기
                    json_output_path = self._find_output_json(output_path)

                    if json_output_path and json_output_path.is_file():
                        with open(json_output_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        # 디버그 모드가 아니면 JSON 파일 삭제
                        if keep_json:
                            logger.info(f"[DEBUG] JSON 파일 유지됨: {json_output_path}")
                        else:
                            json_output_path.unlink()
                        return data
                    else:
                        logger.error(f"변환은 성공했으나 결과 JSON 파일을 찾을 수 없습니다. 출력 디렉토리: {output_path}")
                        return None
                else:
                    logger.error(f"파일 변환 실패: {source_file.name}")
                    logger.error(f"Exit Code: {result.returncode}")
                    logger.error(f"STDOUT: {stdout_text.strip()}")
                    logger.error(f"STDERR: {stderr_text.strip()}")

                    # 추가 디버깅 정보
                    file_size_mb = source_file.stat().st_size / (1024 * 1024)
                    logger.error(f"파일 크기: {file_size_mb:.2f} MB")

                    # Java OutOfMemoryError 감지
                    if "OutOfMemoryError" in stdout_text or "OutOfMemoryError" in stderr_text:
                        logger.error("💀 Java 메모리 부족 오류 감지! -Xmx 값을 늘리거나 POLARIS_ENABLED=False로 대체 변환기를 사용하세요.")

                    # 기타 Java 오류 감지
                    if "Exception" in stderr_text or "Error" in stderr_text:
                        logger.error("Java 예외 발생 - Polaris JAR 내부 오류일 수 있습니다.")

                    return None

            except subprocess.TimeoutExpired:
                logger.error(f"프로세스 시간 초과: {command}")
                return None
            except Exception as e:
                logger.exception(f"문서 변환 중 예상치 못한 오류 발생: {e}")
                return None


# 싱글톤 인스턴스 생성
polaris_converter = PolarisConverter()


class PolarisJsonParser:
    """
    PolarisDataInsight 결과 JSON을 포맷(Word, PPT, PDF, Sheet)에 따라
    자동으로 감지하여 RAG용 Markdown으로 변환하는 통합 파서
    """
    def __init__(self, json_data: Dict[str, Any]):
        self.data = json_data
        # HWP/Word용 프레임 맵 (필요한 경우에만 빌드)
        self.frames_map = self._build_frames_map()

    def parse_to_markdown(self) -> str:
        """
        JSON 루트 키를 기반으로 문서 타입을 자동 감지하여 변환
        """
        # 1. Excel (Sheet)
        if "sheets" in self.data:
            return self._parse_sheet_mode()

        # 2. PPT (Slide)
        elif "slide" in self.data:
            return self._parse_slide_mode()

        # 3. Word / HWP (Body)
        elif "body" in self.data:
            return self._parse_word_mode()

        # 4. PDF (Pages) - Word 구조(body)와 겹치지 않게 체크
        elif "pages" in self.data:
            return self._parse_pdf_mode()

        else:
            return "지원되지 않는 JSON 형식이거나 빈 문서입니다."

    # =========================================================
    # [Mode 1] Excel (Sheet) 파싱
    # =========================================================
    def _parse_sheet_mode(self) -> str:
        full_text = []
        doc_name = self.data.get("docName", "Spreadsheet")
        full_text.append(f"# {doc_name}\n")

        sheets = self.data.get("sheets", [])
        for sheet in sheets:
            sheet_name = sheet.get("sheetName", "Sheet")
            full_text.append(f"\n## Sheet: {sheet_name}\n")

            # 행(Row) 데이터 처리
            if "rows" in sheet:
                table_md = self._convert_sheet_rows_to_markdown(sheet["rows"])
                full_text.append(table_md)

            # objects 처리 (charts, tables 등)
            if "objects" in sheet:
                objects = sheet["objects"]

                # 차트 처리
                if "charts" in objects:
                    for chart in objects["charts"]:
                        chart_md = self._process_chart(chart)
                        if chart_md:
                            full_text.append(chart_md)

                # 테이블 처리 (Excel 내장 테이블)
                if "tables" in objects:
                    for table in objects["tables"]:
                        if "csv" in table:
                            table_name = table.get("displayName", table.get("name", "Table"))
                            full_text.append(f"\n### {table_name}\n")
                            full_text.append(self._csv_to_markdown(table["csv"]))

                # 수식(Equation) 처리
                if "equations" in objects:
                    for eq in objects["equations"]:
                        eq_md = self._process_equation(eq)
                        if eq_md:
                            full_text.append(eq_md)

        return "\n".join(full_text)

    def _convert_sheet_rows_to_markdown(self, rows_data: List[Dict[str, Any]]) -> str:
        """Excel 행 데이터를 마크다운 표로 변환"""
        if not rows_data:
            return ""

        # 데이터 매핑: row_idx -> {col_idx: value}
        grid = {}
        max_col = 0

        for r_item in rows_data:
            r_idx = int(r_item.get("row", 0))
            cols = r_item.get("cols", [])

            if r_idx not in grid:
                grid[r_idx] = {}

            for c_item in cols:
                c_idx = int(c_item.get("col", 0))
                val = c_item.get("value", "")
                grid[r_idx][c_idx] = str(val).replace("\n", "<br>").replace("|", "\\|")
                max_col = max(max_col, c_idx)

        # 마크다운 생성
        lines = []
        sorted_rows = sorted(grid.keys())

        if not sorted_rows:
            return ""

        for i, r_idx in enumerate(sorted_rows):
            row_vals = []
            for c in range(max_col + 1):
                row_vals.append(grid[r_idx].get(c, ""))

            lines.append("| " + " | ".join(row_vals) + " |")

            # 첫 번째 줄 밑에 구분선 추가 (표 형식을 위해 필수)
            if i == 0:
                lines.append("| " + " | ".join(["---"] * (max_col + 1)) + " |")

        return "\n" + "\n".join(lines) + "\n"

    # =========================================================
    # [Mode 2] PDF 파싱
    # =========================================================
    def _parse_pdf_mode(self) -> str:
        full_text = []
        doc_name = self.data.get("docName", "PDF Document")
        full_text.append(f"# {doc_name}\n")

        pages = self.data.get("pages", [])
        for page in pages:
            p_num = page.get("pageNum", 0)
            full_text.append(f"\n--- Page {p_num} ---\n")

            if "contents" in page:
                for content in page["contents"]:
                    c_type = content.get("type")
                    c_data = content.get("content", {})

                    if c_type == "text":
                        text = c_data.get("text", "")
                        if text:
                            full_text.append(text)

                    elif c_type == "table":
                        # PDF 스키마상 표는 HTML로 제공됨. LLM은 HTML 표 이해 가능.
                        html = c_data.get("html", "")
                        if html:
                            full_text.append(f"\n{html}\n")

                    elif c_type == "chart":
                        # 차트 처리
                        chart_md = self._process_chart(c_data)
                        if chart_md:
                            full_text.append(chart_md)

                    elif c_type == "equation":
                        # 수식 처리
                        eq_md = self._process_equation(c_data)
                        if eq_md:
                            full_text.append(eq_md)

        return "\n\n".join(full_text)

    # =========================================================
    # [Mode 3] PPT (Slide) 파싱
    # =========================================================
    def _parse_slide_mode(self) -> str:
        full_text = []
        doc_name = self.data.get("summary_info", {}).get("docName", "Presentation")
        full_text.append(f"# {doc_name}\n")

        for slide in self.data.get("slide", []):
            s_num = slide.get("slide_number", 0)
            full_text.append(f"\n--- Slide {s_num} ---\n")

            if slide.get("note"):
                full_text.append(f"> **Note:** {slide['note']}\n")

            if "contents" in slide:
                full_text.append(self._process_slide_contents(slide["contents"]))

        return "\n".join(full_text)

    def _process_slide_contents(self, contents: List[Dict[str, Any]]) -> str:
        results = []
        for item in contents:
            i_type = item.get("type")

            if i_type == "shape" and "shape" in item:
                results.append(self._process_ppt_shape(item["shape"]))

            elif i_type == "table" and "table" in item:
                results.append(self._convert_table_to_markdown(item["table"]))

            elif i_type == "group" and "group" in item:
                # 그룹 내 shapes 재귀 처리
                group_shapes = item["group"].get("shapes", [])
                results.append(self._process_slide_contents(group_shapes))

            elif i_type == "chart" and "chart" in item:
                # 차트 처리
                chart_md = self._process_chart(item["chart"])
                if chart_md:
                    results.append(chart_md)

            elif i_type == "equation" and "equation" in item:
                # 수식 처리
                eq_md = self._process_equation(item["equation"])
                if eq_md:
                    results.append(eq_md)

        return "\n\n".join([r for r in results if r and r.strip()])

    def _process_ppt_shape(self, shape: Dict[str, Any]) -> str:
        """PPT 도형 내 텍스트 추출"""
        if "para" in shape:
            return self._process_paragraph_list(shape["para"])
        elif "col" in shape:
            # 다단 텍스트 처리
            results = []
            for col in shape["col"]:
                if "para" in col:
                    results.append(self._process_paragraph_list(col["para"]))
            return "\n\n".join(results)
        return ""

    # =========================================================
    # [Mode 4] Word / HWP 파싱
    # =========================================================
    def _parse_word_mode(self) -> str:
        full_text = []
        title = self.data.get("metadata", {}).get("coreProperties", {}).get("title")
        doc_name = self.data.get("docName", "")

        if title:
            full_text.append(f"# {title}\n")
        elif doc_name:
            full_text.append(f"# {doc_name}\n")

        for page in self.data.get("body", []):
            p_num = page.get("pageInfo", {}).get("pageNum", 0)
            full_text.append(f"\n--- Page {p_num} ---\n")

            if "para" in page:
                full_text.append(self._process_paragraph_list(page["para"]))

            # 다단 처리
            if "col" in page:
                for col in page["col"]:
                    col_num = col.get("colNum", 0)
                    if "para" in col:
                        full_text.append(f"\n[Column {col_num}]\n")
                        full_text.append(self._process_paragraph_list(col["para"]))

        return "\n".join(full_text)

    # =========================================================
    # [Helper] 프레임 맵 빌드 (Word/HWP용)
    # =========================================================
    def _build_frames_map(self) -> Dict[int, Any]:
        """
        HWP/Word용 프레임 Lookup 맵
        스키마에 따르면 frames 배열의 각 항목에 ID는 정수 타입
        """
        frames_map = {}
        if "frames" in self.data:
            for frame in self.data["frames"]:
                frame_id = frame.get("ID")
                if isinstance(frame_id, int):
                    frames_map[frame_id] = frame
        return frames_map

    # =========================================================
    # [Helper] 문단 처리
    # =========================================================
    def _process_paragraph_list(self, para_list: List[Dict[str, Any]]) -> str:
        results = []
        for para in para_list:
            text = self._process_single_paragraph(para)
            if text.strip():
                results.append(text)
        return "\n\n".join(results)

    def _process_single_paragraph(self, para: Dict[str, Any]) -> str:
        if "content" not in para:
            return ""

        text_parts = []
        for item in para["content"]:
            if "text" in item:
                text_parts.append(item["text"])

            elif "objectID" in item:
                # HWP/Word 객체 참조 (tableID, imageID, shapeID 등)
                obj_id_data = item["objectID"]
                ref_id = None

                # objectID 객체에서 *ID 키 찾기
                if isinstance(obj_id_data, dict):
                    for k, v in obj_id_data.items():
                        if k.endswith("ID") and isinstance(v, int):
                            ref_id = v
                            break
                elif isinstance(obj_id_data, int):
                    ref_id = obj_id_data

                if ref_id and ref_id in self.frames_map:
                    text_parts.append(self._process_frame_content(self.frames_map[ref_id]))

            elif "breakType" in item:
                if item["breakType"] == "softLineBreak":
                    text_parts.append("\n")

        return "".join(text_parts)

    def _process_frame_content(self, frame: Dict[str, Any]) -> str:
        """프레임 콘텐츠 처리 (테이블, 도형, 차트, 수식 등)"""
        f_type = frame.get("type")

        if f_type == "table":
            return self._convert_table_to_markdown(frame)

        elif f_type == "shape" and "para" in frame:
            return self._process_paragraph_list(frame["para"])

        elif f_type == "chart":
            return self._process_chart(frame)

        elif f_type == "equation":
            return self._process_equation(frame)

        elif f_type == "group" and "shapes" in frame:
            # 그룹 내 프레임 재귀 처리
            results = []
            for sub_frame in frame["shapes"]:
                sub_content = self._process_frame_content(sub_frame)
                if sub_content:
                    results.append(sub_content)
            return "\n\n".join(results)

        return ""

    # =========================================================
    # [Helper] 테이블 변환 (병합 셀 지원)
    # =========================================================
    def _convert_table_to_markdown(self, table_data: Dict[str, Any]) -> str:
        """Word/PPT 공용 테이블 변환기 (병합 셀 처리 포함)"""
        if "cells" not in table_data:
            return ""

        # 그리드 구성: {(row, col): {"text": str, "rowspan": int, "colspan": int}}
        grid = {}
        max_row = 0
        max_col = 0

        for cell in table_data["cells"]:
            metrics = cell.get("metrics", {})
            r = metrics.get("rowaddr", 0)
            c = metrics.get("coladdr", 0)
            rowspan = metrics.get("rowspan", 1)
            colspan = metrics.get("colspan", 1)

            max_row = max(max_row, r)
            max_col = max(max_col, c)

            txt = ""
            if "para" in cell:
                txt = " ".join([self._process_single_paragraph(p).strip() for p in cell["para"]])

            # 파이프 문자 이스케이프 및 줄바꿈 처리
            txt = txt.replace("|", "\\|").replace("\n", "<br>")

            grid[(r, c)] = {
                "text": txt,
                "rowspan": rowspan,
                "colspan": colspan
            }

            # 병합 셀의 나머지 영역 표시 (빈 셀로 처리)
            for dr in range(rowspan):
                for dc in range(colspan):
                    if dr == 0 and dc == 0:
                        continue
                    merge_r, merge_c = r + dr, c + dc
                    max_row = max(max_row, merge_r)
                    max_col = max(max_col, merge_c)
                    if (merge_r, merge_c) not in grid:
                        grid[(merge_r, merge_c)] = {"text": "", "rowspan": 0, "colspan": 0, "merged": True}

        # 마크다운 생성
        lines = []
        for row_idx in range(max_row + 1):
            row_vals = []
            for col_idx in range(max_col + 1):
                cell_data = grid.get((row_idx, col_idx), {"text": ""})
                # 병합된 하위 셀은 빈 문자열로
                if cell_data.get("merged"):
                    row_vals.append("")
                else:
                    row_vals.append(cell_data.get("text", ""))

            lines.append("| " + " | ".join(row_vals) + " |")

            # 첫 번째 줄 밑에 구분선 추가
            if row_idx == 0:
                lines.append("| " + " | ".join(["---"] * (max_col + 1)) + " |")

        return "\n" + "\n".join(lines) + "\n"

    # =========================================================
    # [Helper] 차트 처리
    # =========================================================
    def _process_chart(self, chart_data: Dict[str, Any]) -> str:
        """차트 데이터를 마크다운 표로 변환"""
        result_parts = []

        chart_name = chart_data.get("name", "Chart")
        result_parts.append(f"\n### Chart: {chart_name}\n")

        # chartData에서 정보 추출
        chart_info = chart_data.get("chartData", {})

        # 차트 제목
        if chart_info.get("title"):
            result_parts.append(f"**{chart_info['title']}**\n")

        # 차트 타입
        if chart_info.get("chartType"):
            result_parts.append(f"Type: {chart_info['chartType']}\n")

        # CSV 데이터가 있으면 표로 변환
        if "csv" in chart_data:
            result_parts.append(self._csv_to_markdown(chart_data["csv"]))
        elif "csv" in chart_info:
            result_parts.append(self._csv_to_markdown(chart_info["csv"]))

        # markdown 데이터가 있으면 그대로 사용
        if "markdown" in chart_data:
            result_parts.append(chart_data["markdown"])
        elif "markdown" in chart_info:
            result_parts.append(chart_info["markdown"])

        return "\n".join(result_parts) if len(result_parts) > 1 else ""

    def _csv_to_markdown(self, csv_data: str) -> str:
        """CSV 문자열을 마크다운 표로 변환"""
        if not csv_data:
            return ""

        lines = []
        rows = csv_data.strip().split("\n")

        for i, row in enumerate(rows):
            # CSV 파싱 (간단한 경우만 처리, 복잡한 CSV는 csv 모듈 사용 고려)
            cells = row.split(",")
            # 셀 내용 정리 (따옴표 제거, 파이프 이스케이프)
            cells = [c.strip().strip('"').replace("|", "\\|") for c in cells]
            lines.append("| " + " | ".join(cells) + " |")

            if i == 0:
                lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

        return "\n" + "\n".join(lines) + "\n"

    # =========================================================
    # [Helper] 수식(Equation) 처리
    # =========================================================
    def _process_equation(self, eq_data: Dict[str, Any]) -> str:
        """수식 데이터를 텍스트로 변환"""
        raw_math = eq_data.get("rawMath", {})

        if isinstance(raw_math, dict):
            fmt = raw_math.get("format", "")
            value = raw_math.get("value", "")

            if value:
                # LaTeX 형식이면 $로 감싸기
                if fmt.lower() == "latex":
                    return f"${value}$"
                # OMML이나 기타 형식은 그대로 출력 (또는 변환 로직 추가 가능)
                else:
                    return f"[Equation: {value}]"
        elif isinstance(raw_math, str) and raw_math:
            return f"[Equation: {raw_math}]"

        return ""
