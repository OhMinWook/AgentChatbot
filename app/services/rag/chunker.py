"""
텍스트 청킹 서비스

IncrementalChunker: 페이지 단위로 텍스트를 추가하며 청크를 생성
build_chunks: 추출된 텍스트를 청크로 변환하고 prev/next ID 연결
"""

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.api_clients.model_server_client import model_server_client

logger = logging.getLogger(__name__)


class IncrementalChunker:
    """
    점진적 청킹을 위한 클래스.
    페이지 단위로 텍스트를 추가하면 청크를 생성합니다.
    """

    def __init__(self, file_name: str, chunk_size: int, chunk_overlap: int):
        self.file_name = file_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.buffer = ""
        self.chunk_idx = 0
        self.current_page = 1
        self.page_ranges: List[Tuple[int, int, int]] = []  # (start, end, page_num)

    def add_page(self, page_num: int, text: str) -> List[Dict]:
        """페이지 텍스트 추가 후 생성 가능한 청크 반환"""
        if not text.strip():
            return []

        if self.buffer:
            self.buffer += "\n\n"
        start_idx = len(self.buffer)
        self.buffer += text
        end_idx = len(self.buffer)
        self.page_ranges.append((start_idx, end_idx, page_num))
        self.current_page = page_num

        return self._extract_chunks()

    def _extract_chunks(self) -> List[Dict]:
        """버퍼에서 완성된 청크 추출"""
        chunks = []

        while len(self.buffer) >= self.chunk_size:
            chunk_text = self.buffer[:self.chunk_size]

            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            cut_point = max(last_period, last_newline)

            if cut_point > self.chunk_size * 0.7:
                chunk_text = chunk_text[:cut_point + 1]
                actual_end = cut_point + 1
            else:
                actual_end = self.chunk_size

            if chunk_text.strip():
                page = self._get_page_for_position(0, actual_end)
                chunk_id = f"{self.file_name}_p{page}_{self.chunk_idx}"

                chunks.append({
                    "id": chunk_id,
                    "content": chunk_text.strip(),
                    "metadata": {
                        "source": self.file_name,
                        "page": page
                    }
                })
                self.chunk_idx += 1

            remove_len = actual_end - self.chunk_overlap
            if remove_len > 0:
                self.buffer = self.buffer[remove_len:]
                self.page_ranges = [
                    (max(0, s - remove_len), max(0, e - remove_len), p)
                    for s, e, p in self.page_ranges
                    if e > remove_len
                ]
            else:
                break

        return chunks

    def _get_page_for_position(self, start: int, end: int) -> int:
        """위치에 해당하는 페이지 번호 반환"""
        for ps, pe, pn in self.page_ranges:
            if ps < end and pe > start:
                return pn
        return self.current_page

    def flush(self) -> List[Dict]:
        """남은 버퍼를 청크로 변환(마지막 청크가 청크 사이즈보다 작을 때 실행)"""
        chunks = []
        remaining = self.buffer.strip()

        if remaining and not (self.chunk_idx > 0 and len(remaining) <= self.chunk_overlap):
            page = self._get_page_for_position(0, len(self.buffer))
            chunk_id = f"{self.file_name}_p{page}_{self.chunk_idx}"

            chunks.append({
                "id": chunk_id,
                "content": remaining,
                "metadata": {
                    "source": self.file_name,
                    "page": page
                }
            })
            self.chunk_idx += 1
            self.buffer = ""

        return chunks


_KO_SENTENCE_END = re.compile(r'[다요죠]\.|니다\.|십시오\.|하였다\.|되었다\.|있다\.|없다\.')


def _find_korean_cut(chunk: str, threshold: int) -> int:
    """한국어 종결어미 기준 cut point 탐색. threshold 이후에서 가장 마지막 매칭 반환."""
    best = -1
    for m in _KO_SENTENCE_END.finditer(chunk):
        if m.end() > threshold:
            best = m.end() - 1
    return best


def _split_by_size(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """텍스트가 chunk_size를 초과하면 문장/개행 경계에서 분할"""
    if len(text) <= chunk_size:
        return [text]

    threshold = int(chunk_size * 0.7)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        chunk = text[start:end]
        ko_cut = _find_korean_cut(chunk, threshold)
        newline_cut = chunk.rfind('\n')
        cut = max(ko_cut, newline_cut if newline_cut > threshold else -1)
        if cut > threshold:
            end = start + cut + 1
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap

    return [c for c in chunks if c.strip()]


def _split_section_with_tables(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """섹션 내에서 표를 독립 청크로 분리하고, 긴 본문은 추가 분할"""
    table_pattern = re.compile(r'((?:\|[^\n]*\n?)+)', re.MULTILINE)
    parts = table_pattern.split(text)

    result = []
    current_text = ""
    for part in parts:
        if part.strip().startswith("|"):
            if current_text.strip():
                result.extend(_split_by_size(current_text.strip(), chunk_size, chunk_overlap))
                current_text = ""
            result.append(part.strip())
        else:
            current_text += part

    if current_text.strip():
        result.extend(_split_by_size(current_text.strip(), chunk_size, chunk_overlap))

    return result


def _merge_small_chunks(
    raw: List[Tuple[str, int]], chunk_size: int, min_chunk_size: int
) -> List[Tuple[str, int]]:
    """min_chunk_size 미만 청크를 인접 청크와 병합.

    - 표 청크(| 시작)는 병합하지 않고 독립 유지
    - 병합 시 chunk_size 초과하면 병합 중단
    - page 번호는 첫 번째 청크의 page 유지
    """
    if not raw:
        return []

    merged: List[Tuple[str, int]] = []
    pending_text, pending_page = raw[0]

    for text, page in raw[1:]:
        is_table = text.lstrip().startswith("|")
        pending_is_table = pending_text.lstrip().startswith("|")

        if pending_is_table:
            # 표 청크는 그대로 확정
            merged.append((pending_text, pending_page))
            pending_text, pending_page = text, page
            continue

        if is_table:
            # 다음이 표이면 pending 그대로 확정 후 표를 pending으로
            merged.append((pending_text, pending_page))
            pending_text, pending_page = text, page
            continue

        combined = pending_text + "\n\n" + text
        if len(pending_text) < min_chunk_size and len(combined) <= chunk_size:
            # 작은 청크 + 다음 청크 병합
            pending_text = combined
        else:
            merged.append((pending_text, pending_page))
            pending_text, pending_page = text, page

    merged.append((pending_text, pending_page))
    return merged


def _build_structured_chunks(file_name: str, page_results: List[Tuple[int, str]]) -> List[Dict]:
    """마크다운 헤더 기반 구조 인식 청킹 (PDF 전용)"""
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP
    min_chunk_size = max(chunk_size // 2, 300)  # 기본 350자 (CHUNK_SIZE=700 기준)

    # 1. 페이지 마커와 함께 전체 텍스트 합치기
    combined_lines = []
    for page_num, text in page_results:
        combined_lines.append(f"<!-- PAGE:{page_num} -->")
        combined_lines.append(text)
    combined_text = "\n".join(combined_lines)

    # 2. 헤더(# ## ###) 앞에서 섹션 분리
    section_pattern = re.compile(r'(?=\n#{1,3} )')
    sections = section_pattern.split(combined_text)

    # 3. 섹션별 sub-chunk 생성 → (text, page) 목록으로 수집
    raw_chunks: List[Tuple[str, int]] = []
    current_page = 1
    header_stack: List[str] = ["", "", ""]  # [h1, h2, h3]

    for section in sections:
        if not section.strip():
            continue

        page_markers = re.findall(r'<!-- PAGE:(\d+) -->', section)
        if page_markers:
            current_page = int(page_markers[-1])
        clean_section = re.sub(r'<!-- PAGE:\d+ -->\n?', '', section).strip()

        if not clean_section:
            continue

        # 헤더 추출 및 스택 갱신
        header_match = re.match(r'^(#{1,3})\s+(.+)', clean_section)
        if header_match:
            level = len(header_match.group(1))
            header_text = header_match.group(2).strip()
            header_stack[level - 1] = header_text
            for i in range(level, 3):
                header_stack[i] = ""

        # 헤더 컨텍스트 prefix 생성 (빈 항목 제외)
        breadcrumb = " > ".join(h for h in header_stack if h)
        header_prefix = f"[{breadcrumb}]\n" if breadcrumb else ""

        sub_chunks = _split_section_with_tables(clean_section, chunk_size, chunk_overlap)
        for sub_text in sub_chunks:
            if sub_text.strip():
                # 표 청크는 prefix 미적용
                content = sub_text.strip()
                if header_prefix and not content.startswith("|"):
                    content = header_prefix + content
                raw_chunks.append((content, current_page))

    # 4. 소형 청크 병합
    merged_chunks = _merge_small_chunks(raw_chunks, chunk_size, min_chunk_size)

    # 5. 최종 Dict 생성
    chunks = []
    for chunk_idx, (text, page) in enumerate(merged_chunks):
        chunk_id = f"{file_name}_p{page}_{chunk_idx}"
        chunks.append({
            "id": chunk_id,
            "content": text,
            "metadata": {
                "source": file_name,
                "page": page,
            }
        })

    return chunks


_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[다요죠]\.)\s+|(?<=니다\.)\s+|(?<=십시오\.)\s+|(?<=[.!?])\s{2,}|\n{2,}'
)


def _split_sentences(text: str) -> List[str]:
    """텍스트를 문장 단위로 분리 (단일 줄바꿈은 공백으로 정규화)"""
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    sentences = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """두 벡터의 코사인 유사도 계산"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _build_semantic_chunks(
    file_name: str,
    text: str,
    default_page: int = 1,
) -> List[Dict]:
    """문장 임베딩 유사도 기반 Semantic Chunking

    1. 텍스트를 문장 단위로 분리
    2. 전체 문장 임베딩
    3. 인접 문장 간 코사인 유사도 계산
    4. 유사도가 threshold 미만인 지점을 청크 경계로 결정
    5. 청크 크기 초과 시 _split_by_size로 추가 분할
    """
    chunk_size = settings.CHUNK_SIZE
    threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD

    sentences = _split_sentences(text)
    if not sentences:
        return []

    embeddings = await model_server_client.embed_texts(sentences, is_query=False)

    # 유사도 급락 지점 = 청크 경계
    breakpoints = [0]
    for i in range(len(sentences) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < threshold:
            breakpoints.append(i + 1)
    breakpoints.append(len(sentences))

    chunks = []
    chunk_idx = 0
    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        chunk_text = " ".join(sentences[start:end]).strip()
        if not chunk_text:
            continue
        if len(chunk_text) > chunk_size:
            for sub in _split_by_size(chunk_text, chunk_size, settings.CHUNK_OVERLAP):
                if sub.strip():
                    chunks.append({
                        "id": f"{file_name}_p{default_page}_{chunk_idx}",
                        "content": sub,
                        "metadata": {"source": file_name, "page": default_page},
                    })
                    chunk_idx += 1
        else:
            chunks.append({
                "id": f"{file_name}_p{default_page}_{chunk_idx}",
                "content": chunk_text,
                "metadata": {"source": file_name, "page": default_page},
            })
            chunk_idx += 1

    return chunks


async def build_chunks(
    file_name: str,
    page_results: Optional[List[Tuple[int, str]]],
    markdown_content: Optional[str],
    doc_type: str = "structured",
) -> Tuple[List[Dict], int]:
    """텍스트를 청크로 분할하고 prev/next ID 연결(청크 간의 순서 관계를 명시)

    Returns:
        (all_chunks, max_page)
    """
    if doc_type == "unstructured":
        raw_text = "\n".join(t for _, t in page_results) if page_results else (markdown_content or "")
        default_page = page_results[0][0] if page_results else 1
        all_chunks = await _build_semantic_chunks(file_name, raw_text, default_page)
    elif page_results:
        # PDF: 마크다운 헤더 기반 구조 인식 청킹
        all_chunks = _build_structured_chunks(file_name, page_results)
    else:
        # 나머지 포맷(HWPX, DOCX 등): 기존 슬라이딩 윈도우
        chunker = IncrementalChunker(
            file_name=file_name,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        all_chunks = chunker.add_page(1, markdown_content)
        all_chunks.extend(chunker.flush())

    max_page = max((c["metadata"]["page"] for c in all_chunks), default=1)

    for i, chunk in enumerate(all_chunks):
        chunk["metadata"]["prev_chunk_id"] = all_chunks[i - 1]["id"] if i > 0 else None
        chunk["metadata"]["next_chunk_id"] = all_chunks[i + 1]["id"] if i < len(all_chunks) - 1 else None

    logger.debug(f"[Chunker] doc_type={doc_type}, 총 {len(all_chunks)}개 청크 생성")
    for chunk in all_chunks[:3]:
        logger.debug(f"[Chunker] 샘플 청크 ({chunk['id']}): {chunk['content'][:100]!r}")

    return all_chunks, max_page
