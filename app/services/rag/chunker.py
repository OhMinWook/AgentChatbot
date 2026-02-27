"""
텍스트 청킹 서비스

IncrementalChunker: 페이지 단위로 텍스트를 추가하며 청크를 생성
build_chunks: 추출된 텍스트를 청크로 변환하고 prev/next ID 연결
"""

from typing import Dict, List, Optional, Tuple

from app.core.config import settings


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


def build_chunks(
    file_name: str,
    page_results: Optional[List[Tuple[int, str]]],
    markdown_content: Optional[str],
) -> Tuple[List[Dict], int]:
    """텍스트를 청크로 분할하고 prev/next ID 연결(청크 간의 순서 관계를 명시)

    Returns:
        (all_chunks, max_page)
    """
    chunker = IncrementalChunker(
        file_name=file_name,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    all_chunks: List[Dict] = []
    max_page = 0

    if page_results:
        for page_num, page_text in page_results:
            max_page = max(max_page, page_num)
            all_chunks.extend(chunker.add_page(page_num, page_text))
    else:
        max_page = 1
        all_chunks.extend(chunker.add_page(1, markdown_content))

    all_chunks.extend(chunker.flush())

    for i, chunk in enumerate(all_chunks):
        chunk["metadata"]["prev_chunk_id"] = all_chunks[i - 1]["id"] if i > 0 else None
        chunk["metadata"]["next_chunk_id"] = all_chunks[i + 1]["id"] if i < len(all_chunks) - 1 else None

    return all_chunks, max_page
