from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_health import router as health_router
from app.api.routes_chatbot import router as chatbot_router
from app.api.routes_callsummary import router as callsummary_router
from app.api.routes_dialogue_converter import router as dialogue_converter_router
from app.api.routes_document_automation import router as document_automation_router

# 클라이언트 인스턴스들 (종료 시 정리용)
from app.services.clients.llm_client import llm_client
from app.services.clients.model_server_client import model_server_client
from app.services.clients.stt_client import stt_client
from app.services.utils.memory_service import memory_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행할 코드 (필요하면 여기에 추가)
    yield
    # 앱 종료 시 클라이언트 정리
    await llm_client.close()
    await model_server_client.close()
    await stt_client.close()
    await memory_service.redis.aclose()


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)

# CORS 설정 (테스트용 - 운영에서는 origins 제한 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(chatbot_router)
app.include_router(callsummary_router)
app.include_router(dialogue_converter_router)
app.include_router(document_automation_router)
