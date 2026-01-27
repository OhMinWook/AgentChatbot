import os
import torch
import numpy as np
import json
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import outlines
from vllm import LLM, SamplingParams
from outlines import Generator
from pylate import models as pylate_models

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================================== 
# 설정 (Settings)
# ======================================== 
class Settings(BaseSettings):
    COLBERT_MODEL_NAME: str = "jinaai/jina-colbert-v2"
    QWEN_MODEL_NAME: str = "Qwen/Qwen3-4B-Instruct-2507"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # vLLM 설정
    GPU_MEMORY_UTILIZATION: float = 0.6
    MAX_MODEL_LEN: int = 4096
    ATTENTION_BACKEND: str = "FLASHINFER"
    ENFORCE_EAGER: bool = True
    
    # 인코딩 설정
    COLBERT_BATCH_SIZE_QUERY: int = 32
    COLBERT_BATCH_SIZE_DOC: int = 128

settings = Settings()

# 전역 변수 (모델 인스턴스)
colbert_model: Optional[pylate_models.ColBERT] = None
qwen_model = None
graph_generator = None
keyword_generator = None

# ======================================== 
# Pydantic 스키마 (Outlines용)
# ======================================== 
class Entity(BaseModel):
    name: str
    type: str
    description: str

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    description: str

class GraphSchema(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

# ======================================== 
# API Request/Response 스키마
# ======================================== 
class EncodeRequest(BaseModel):
    texts: List[str]
    is_query: bool = False

class EncodeResponse(BaseModel):
    embeddings: List[List[List[float]]]

class KeywordRequest(BaseModel):
    texts: List[str]

class GraphExtractRequest(BaseModel):
    texts: List[str]

# ======================================== 
# 프롬프트 템플릿
# ======================================== 
KEYWORD_SYSTEM_PROMPT = """You are a keyword extraction assistant. Extract 5-10 key concepts, entities, or important terms from the given text. Return only comma-separated keywords in Korean. Do not include explanations."""

KEYWORD_USER_TEMPLATE = """다음 본문에서 핵심 키워드를 5~10개 정도 추출해줘. 한국어로만 출력하고 쉼표로 구분해줘.

본문:
{text}

키워드:"""

GRAPH_SYSTEM_PROMPT = """You are an expert at extracting entities and relationships from text to build a knowledge graph. 
For the given text, extract all important entities and the relationships between them in JSON format."""

GRAPH_USER_TEMPLATE = """다음 텍스트에서 지식 그래프 구성을 위한 엔티티와 관계를 추출해줘.

텍스트:
{text}

결과:"""

# ======================================== 
# 모델 로드 함수
# ======================================== 
def load_colbert():
    global colbert_model
    logger.info(f"Loading Jina ColBERT v2 on {settings.DEVICE}...")
    try:
        colbert_model = pylate_models.ColBERT(
            model_name_or_path=settings.COLBERT_MODEL_NAME,
            device=settings.DEVICE,
            trust_remote_code=True
        )
        logger.info("ColBERT loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load ColBERT: {e}")

def load_qwen_with_outlines():
    global qwen_model, graph_generator, keyword_generator
    logger.info(f"Loading Qwen3-4B with Outlines + vLLM({settings.ATTENTION_BACKEND}) on {settings.DEVICE}...")

    try:
        # vLLM 최적화 설정으로 모델 로드
        llm = LLM(
            model=settings.QWEN_MODEL_NAME,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
            max_model_len=settings.MAX_MODEL_LEN,
            attention_backend=settings.ATTENTION_BACKEND,
            enforce_eager=settings.ENFORCE_EAGER,
        )

        # Outlines v1: vLLM offline wrapper
        qwen_model = outlines.from_vllm_offline(llm)

        # Generator 초기화
        graph_generator = Generator(qwen_model, GraphSchema)  # 구조화된 JSON 응답용
        keyword_generator = Generator(qwen_model)             # 일반 텍스트 응답용

        logger.info("Qwen3-4B + Outlines Generator loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Qwen/Outlines: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 모델 로드
    load_colbert()
    load_qwen_with_outlines()
    yield
    # 앱 종료 시 정리 (필요한 경우)
    logger.info("Shutting down model server...")

app = FastAPI(title="AI Model Server", lifespan=lifespan)

# ======================================== 
# API 엔드포인트
# ======================================== 
@app.post("/encode", response_model=EncodeResponse)
async def encode_texts(request: EncodeRequest):
    if colbert_model is None:
        raise HTTPException(status_code=503, detail="ColBERT model not loaded")
    if not request.texts:
        return EncodeResponse(embeddings=[])
    
    try:
        batch_size = settings.COLBERT_BATCH_SIZE_QUERY if request.is_query else settings.COLBERT_BATCH_SIZE_DOC
        embeddings = colbert_model.encode(
            request.texts,
            batch_size=batch_size,
            is_query=request.is_query,
            show_progress_bar=False
        )
        
        # 임베딩 결과 변환 (numpy/torch -> list)
        if hasattr(embeddings, 'cpu'):
            embeddings_list = embeddings.cpu().numpy().tolist()
        elif isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings
            
        return EncodeResponse(embeddings=embeddings_list)
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/encode/query")
async def encode_query(request: EncodeRequest):
    request.is_query = True
    return await encode_texts(request)

@app.post("/encode/documents")
async def encode_documents(request: EncodeRequest):
    request.is_query = False
    return await encode_texts(request)

@app.post("/extract_keywords")
async def extract_keywords(request: KeywordRequest):
    if keyword_generator is None:
        raise HTTPException(status_code=503, detail="Keyword generator not initialized")

    if not request.texts:
        return {"keywords": []}

    # 프롬프트 구성
    prompts = []
    for text in request.texts:
        truncated = text[:1500]  # 컨텍스트 길이 제한
        full_prompt = (
            f"<|im_start|>system\n{KEYWORD_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{KEYWORD_USER_TEMPLATE.format(text=truncated)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(full_prompt)

    try:
        sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.0,
            stop=["<|im_end|>", "\n"]
        )

        outputs = keyword_generator.batch(prompts, sampling_params=sampling_params)
        keywords_list = [out.strip() for out in outputs]
        return {"keywords": keywords_list}

    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        return {"keywords": [""] * len(request.texts)}

@app.post("/extract_graph")
async def extract_graph(request: GraphExtractRequest):
    if graph_generator is None:
        raise HTTPException(status_code=503, detail="Graph generator not initialized")

    if not request.texts:
        return {"results": []}

    logger.info(f"Extracting graph from {len(request.texts)} texts...")

    # 구조화된 데이터 추출을 위한 프롬프트 구성
    prompts = []
    for text in request.texts:
        truncated = text[:2000]
        full_prompt = (
            f"<|im_start|>system\n{GRAPH_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{GRAPH_USER_TEMPLATE.format(text=truncated)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompts.append(full_prompt)

    try:
        sampling_params = SamplingParams(
            max_tokens=1024,
            temperature=0.0,
        )

        # Outlines를 사용한 구조화된 JSON 배치 생성
        json_strings = graph_generator.batch(prompts, sampling_params=sampling_params)

        results = []
        for s in json_strings:
            try:
                # Pydantic 모델로 검증 및 파싱
                obj = GraphSchema.model_validate_json(s)
                results.append(obj.model_dump())
            except Exception as ve:
                logger.warning(f"JSON validation failed: {ve}")
                results.append({"entities": [], "relationships": []})

        return {"results": results}

    except Exception as e:
        logger.error(f"Graph extraction error: {e}")
        return {"results": [{"entities": [], "relationships": []}] * len(request.texts)}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "colbert_loaded": colbert_model is not None,
        "outlines_loaded": graph_generator is not None,
        "device": settings.DEVICE,
        "config": {
            "colbert_model": settings.COLBERT_MODEL_NAME,
            "llm_model": settings.QWEN_MODEL_NAME
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)