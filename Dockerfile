# 1. Python 3.12 슬림 버전 사용
FROM python:3.12-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필수 패키지 설치 (git: 라이브러리 설치, ffmpeg: 오디오 처리)
RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 복사
COPY . .

# 6. 실행 명령어 (FastAPI 실행)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
