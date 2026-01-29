"""경로 검증 유틸리티 - Path Traversal 방지"""

import os
import re


def sanitize_path_component(component: str) -> str:
    """
    경로 구성 요소(파일명, 디렉토리명)에서 위험한 문자를 검증합니다.
    Path Traversal 공격을 방지합니다.

    :param component: 검증할 경로 구성 요소
    :return: 검증된 경로 구성 요소
    :raises ValueError: 위험한 문자가 포함된 경우
    """
    if not component or not component.strip():
        raise ValueError("경로 구성 요소가 비어있습니다.")

    component = component.strip()

    # .. 경로 탐색 차단
    if ".." in component:
        raise ValueError(f"허용되지 않는 경로 패턴입니다: {component}")

    # 경로 구분자 차단 (/, \)
    if "/" in component or "\\" in component:
        raise ValueError(f"경로 구분자가 포함되어 있습니다: {component}")

    # null byte 차단
    if "\x00" in component:
        raise ValueError("허용되지 않는 문자가 포함되어 있습니다.")

    return component


def safe_join(base_dir: str, *components: str) -> str:
    """
    base_dir 하위로만 경로를 생성합니다.
    결과 경로가 base_dir 바깥이면 거부합니다.

    :param base_dir: 허용된 최상위 디렉토리
    :param components: 하위 경로 구성 요소들
    :return: 안전한 절대 경로
    :raises ValueError: 결과 경로가 base_dir 바깥인 경우
    """
    for comp in components:
        sanitize_path_component(comp)

    joined = os.path.join(base_dir, *components)
    resolved = os.path.normpath(os.path.abspath(joined))
    base_resolved = os.path.normpath(os.path.abspath(base_dir))

    if not resolved.startswith(base_resolved + os.sep) and resolved != base_resolved:
        raise ValueError(f"허용된 디렉토리 범위를 벗어나는 경로입니다.")

    return resolved
