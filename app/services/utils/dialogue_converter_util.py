import csv
import re
import time
import logging
from io import StringIO

logger = logging.getLogger(__name__)

def clean_content(content: str, original_filename: str) -> tuple[str, bool]:
    """
    메시지 내용을 정제합니다.
    - <FONT ...> 태그 제거
    - ATTACH://... 부분을 원본 파일명으로 교체

    Returns:
        (정제된 내용, 첨부파일 여부)
    """
    if not content:
        return "", False

    content = re.sub(r'<FONT[^>]*>', '', content)
    is_attachment = False
    if content.startswith('ATTACH://') and original_filename:
        content = f"[첨부파일: {original_filename}]"
        is_attachment = True

    return content.strip(), is_attachment

def _parse_formatted_date(formatted_date: str) -> tuple[str, str]:
    """날짜 문자열에서 (date_only, time_only) 파싱 (두 가지 포맷 지원)

    - 컴팩트: YYYYMMDDHHmmss[SSS] (숫자만, 14자 이상)
    - ISO:    YYYY-MM-DD[ HH:mm:ss]
    """
    if not formatted_date:
        return '', ''

    # 컴팩트 포맷: 숫자만 14자 이상
    if re.match(r'^\d{14,}$', formatted_date):
        return (
            f"{formatted_date[0:4]}-{formatted_date[4:6]}-{formatted_date[6:8]}",
            f"{formatted_date[8:10]}:{formatted_date[10:12]}:{formatted_date[12:14]}"
        )

    # ISO 포맷: YYYY-MM-DD 또는 YYYY-MM-DD HH:mm:ss
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}):(\d{2}))?', formatted_date)
    if m:
        date_only = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        time_only = f"{m.group(4)}:{m.group(5)}:{m.group(6)}" if m.group(4) else ''
        return date_only, time_only

    return '', ''


def convert_csv_to_dialogue(csv_content: bytes) -> str:
    """
    CSV 파일 내용을 대화록 형식으로 변환합니다.

    :param csv_content: CSV 파일의 바이트 내용
    :return: 변환된 대화록 문자열
    """
    start_time = time.time()
    logger.info(f"[Dialogue] CSV 파싱 시작 (size: {len(csv_content)} bytes)")

    try:
        # UTF-8로 디코딩 시도, 실패 시 CP949로 시도
        try:
            csv_string = csv_content.decode('utf-8-sig')
        except UnicodeDecodeError:
            csv_string = csv_content.decode('cp949')
    except UnicodeDecodeError as e:
        raise ValueError(f"파일 인코딩을 확인해주세요 (UTF-8, CP949): {e}")

    # StringIO를 사용하여 문자열을 파일처럼 다룸
    f = StringIO(csv_string)
    reader = csv.DictReader(f)
    
    # 필수 컬럼 확인
    required_columns = {'sTalkerName', 'formatted_date', 'sTalkerContent', 'sOriginalFileName'}
    if not required_columns.issubset(reader.fieldnames):
        raise ValueError(f"CSV 파일에 필수 컬럼('sTalkerName', 'formatted_date', 'sTalkerContent', 'sOriginalFileName')이 없습니다. 현재 컬럼: {reader.fieldnames}")

    dialogue_lines = []
    current_date = None

    prev_talker = None
    prev_time = None
    prev_contents = []
    prev_is_attachment = False

    def flush_message():
        """누적된 메시지를 dialogue_lines에 추가"""
        nonlocal prev_talker, prev_time, prev_contents, prev_is_attachment
        if prev_talker and prev_contents:
            combined = '\n'.join(prev_contents)
            if prev_time:
                dialogue_lines.append(f'[{prev_time}] {prev_talker}: {combined}')
            else:
                dialogue_lines.append(f'{prev_talker}: {combined}')
        prev_talker = None
        prev_time = None
        prev_contents = []
        prev_is_attachment = False

    for row in reader:
        talker_name = (row.get('sTalkerName') or '').strip()
        formatted_date = (row.get('formatted_date') or '').strip()
        content = row.get('sTalkerContent') or ''
        original_filename = (row.get('sOriginalFileName') or '').strip()

        cleaned_content, is_attachment = clean_content(content, original_filename)

        if not cleaned_content:
            continue

        date_only, time_only = _parse_formatted_date(formatted_date)

        if date_only and date_only != current_date:
            flush_message()
            if current_date is not None:
                dialogue_lines.append('')
            dialogue_lines.append(f'===== {date_only} =====')
            current_date = date_only

        if (talker_name == prev_talker and not is_attachment and not prev_is_attachment):
            prev_contents.append(cleaned_content)
        else:
            flush_message()
            prev_talker = talker_name
            prev_time = time_only
            prev_contents = [cleaned_content]
            prev_is_attachment = is_attachment

    flush_message()

    duration = time.time() - start_time
    logger.info(f"[Dialogue] CSV 파싱 완료: {len(dialogue_lines)}줄, {duration:.2f}초")

    return '\n'.join(dialogue_lines)


def split_dialogue_by_date(csv_content: bytes) -> list[dict[str, str]]:
    """
    CSV 파일을 대화록으로 변환한 뒤, 날짜별로 분리하여 반환합니다.

    :param csv_content: CSV 파일의 바이트 내용
    :return: [{"date": "2025-01-15", "text": "대화 내용..."}, ...] 형태의 리스트
    """
    logger.info("[Dialogue] split_dialogue_by_date 시작")
    dialogue_text = convert_csv_to_dialogue(csv_content)
    logger.info("[Dialogue] 날짜별 분리 시작")

    result = []
    current_date = None
    current_lines = []

    for line in dialogue_text.split('\n'):
        match = re.match(r'^={5}\s+(\S+)\s+={5}$', line)
        if match:
            if current_date is not None and current_lines:
                result.append({
                    "date": current_date,
                    "text": '\n'.join(current_lines).strip()
                })
            current_date = match.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_date is not None and current_lines:
        result.append({
            "date": current_date,
            "text": '\n'.join(current_lines).strip()
        })

    logger.info(f"[Dialogue] 날짜별 분리 완료: {len(result)}개 세그먼트")
    return result
