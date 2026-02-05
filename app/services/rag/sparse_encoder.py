"""
BM25 스타일 Sparse Vector 생성

Qdrant Sparse Vector 형식으로 변환
"""

from collections import Counter
from typing import Dict, List
import re


class SparseEncoder:
    """
    BM25 스타일 Sparse Vector 생성기

    한국어 텍스트를 sparse vector (indices, values)로 변환
    """

    def __init__(self, vocab_size: int = 100000):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> Dict[str, List]:
        """
        텍스트를 Qdrant Sparse Vector 형식으로 변환

        Returns:
            {"indices": [int, ...], "values": [float, ...]}
        """
        tokens = self._tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}

        tf = Counter(tokens)

        # 해시 충돌 시 값을 합산하기 위해 딕셔너리 사용
        index_values: Dict[int, float] = {}
        for token, count in tf.items():
            # 토큰을 해시하여 인덱스로 변환
            idx = abs(hash(token)) % self.vocab_size
            # 해시 충돌 시 값 합산 (indices는 unique해야 함)
            index_values[idx] = index_values.get(idx, 0.0) + float(count)

        indices = list(index_values.keys())
        values = list(index_values.values())

        return {"indices": indices, "values": values}

    def encode_batch(self, texts: List[str]) -> List[Dict[str, List]]:
        """배치 인코딩"""
        return [self.encode(text) for text in texts]

    def _tokenize(self, text: str) -> List[str]:
        """
        간단한 한국어 토크나이징

        - 공백/특수문자 기준 분리
        - 소문자 변환
        - 2글자 이상만 유지
        """
        tokens = re.findall(r'[\w가-힣]+', text.lower())
        return [t for t in tokens if len(t) >= 2]


# 싱글톤
sparse_encoder = SparseEncoder()
