## 여기서는 LLM 모델으로 부터 받아온 output을 검증하고 API 명세서에 적힌 형태대로 왔는지 재검증을 합니다.


def validate_bullets(output: str) -> str:
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if not lines:
        return output

    # 최소 규칙: 첫 줄이 '- '로 시작하지 않으면 강제로 bullet로 래핑(아주 약한 보정)
    if not lines[0].startswith("- "):
        lines = [f"- {ln}" if not ln.startswith("- ") else ln for ln in lines]

    return "\n".join(lines)
