"""RAG 텍스트 처리 유틸리티"""

_CONTEXTUAL_PREFIX_TEMPLATE = "[파일: {filename}] "


def add_source_prefix(text: str, filename: str) -> str:
    """임베딩/리랭킹용 텍스트에 출처 파일명 프리픽스를 추가"""
    if not filename:
        return text
    return _CONTEXTUAL_PREFIX_TEMPLATE.format(filename=filename) + text
