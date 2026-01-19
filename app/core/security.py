from fastapi import Header, HTTPException

# [기능 추가 포인트 #1] 인증(AuthN)을 여기에 구현
# 예) X-API-Key, JWT Bearer 토큰, 사내 SSO 프록시 헤더 등

async def require_api_key(x_api_key: str | None = Header(default=None)):
    # MVP에서는 비활성화(항상 통과)
    # 운영에서는 아래처럼 강제:
    #
    # if x_api_key != "YOUR_SECRET":
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    return True
