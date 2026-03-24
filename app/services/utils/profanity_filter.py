"""욕설/비속어 필터링 유틸리티"""
import csv
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DATASET_PATH = Path(__file__).parent.parent.parent / "data" / "dataset.csv"

_PATTERN = None


def _load_banned_words() -> list[str]:
    """dataset.csv에서 금칙어 목록 로드 (slang 컬럼 기준)"""
    if not _DATASET_PATH.exists():
        logger.warning(f"금칙어 파일 없음: {_DATASET_PATH}")
        return []

    words = []
    with open(_DATASET_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get("slang", "").strip()
            if word:
                words.append(word)

    logger.info(f"금칙어 {len(words)}개 로드 완료")
    return words


def _build_pattern(words: list[str]):
    if not words:
        return None
    return re.compile(
        "|".join(re.escape(w) for w in words),
        flags=re.IGNORECASE
    )


def reload():
    """금칙어 목록을 다시 로드 (런타임 갱신용)"""
    global _PATTERN
    words = _load_banned_words()
    _PATTERN = _build_pattern(words)
    logger.info("금칙어 패턴 갱신 완료")


def filter_profanity(text: str, replacement: str = "***") -> str:
    """텍스트에서 욕설/비속어를 replacement로 치환"""
    if not _PATTERN:
        return text
    return _PATTERN.sub(replacement, text)


# 서버 시작 시 초기 로드
reload()
