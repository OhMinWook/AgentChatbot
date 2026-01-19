from fastapi import FastAPI
from app.api.routes_health import router as health_router
from app.api.routes_chatbot import router as chatbot_router
from app.api.routes_callsummary import router as callsummary_router

app = FastAPI(title="LLM Gateway", version="0.1.0")

app.include_router(health_router)
app.include_router(chatbot_router)
app.include_router(callsummary_router)