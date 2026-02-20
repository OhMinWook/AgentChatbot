import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 로깅 설정 (환경 변수로 레벨 조정 가능)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from app.api.routes_health import router as health_router
from app.api.routes_chatbot import router as chatbot_router
from app.api.routes_callsummary import router as callsummary_router
from app.api.routes_dialogue_converter import router as dialogue_converter_router
from app.api.routes_document_automation import router as document_automation_router
from app.api.routes_admin import router as admin_router

# 클라이언트 인스턴스들 (종료 시 정리용)
from app.services.api_clients.llm_client import llm_client
from app.services.api_clients.model_server_client import model_server_client
from app.services.api_clients.stt_client import stt_client
from app.services.utils.memory_service import memory_service
from app.services.rag.extractors import _pdf_process_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행할 코드 (필요하면 여기에 추가)
    yield
    # 앱 종료 시 클라이언트 정리
    await llm_client.close()
    await model_server_client.close()
    await stt_client.close()
    await memory_service.redis.aclose()
    # 프로세스 풀 정리
    if _pdf_process_pool is not None:
        _pdf_process_pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

# CORS 설정 (환경 변수 CORS_ORIGINS로 허용 도메인 설정, 쉼표 구분)
# 예: CORS_ORIGINS="http://localhost:3000,https://example.com"
# 기본값: 모든 도메인 허용 (개발용)
_cors_origins_str = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = ["*"] if _cors_origins_str == "*" else [o.strip() for o in _cors_origins_str.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test", include_in_schema=False)
async def test_ui():
    return FileResponse("test_ui.html")


app.include_router(health_router)
app.include_router(chatbot_router)
app.include_router(callsummary_router)
app.include_router(dialogue_converter_router)
app.include_router(document_automation_router)
app.include_router(admin_router)
