import os
import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from pylate import models

# ======================================== 
# 설정
# ======================================== 
COLBERT_MODEL_NAME = "jinaai/jina-colbert-v2"
QWEN_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

colbert_model: Optional[models.ColBERT] = None
qwen_model: Optional[LLM] = None
qwen_tokenizer = None

# ======================================== 
# Request/Response 스키마
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
# 프롬프트 정의
# ======================================== 
KEYWORD_SYSTEM_PROMPT = """You are a keyword extraction assistant. Extract 5-10 key concepts, entities, or important terms from the given text. Return only comma-separated keywords in Korean. Do not include explanations."""

KEYWORD_USER_TEMPLATE = """텍스트 본문에서 핵심 키워드 5~10개를 추출해주세요. 답변은 쉼표로 구분된 리스트만 포함해야 합니다.

본문:
{text}

키워드:"""

ENTITY_EXTRACTION_SYSTEM_PROMPT = "You are a knowledge graph extractor. Extract entities and their relationships from the given text."

ENTITY_EXTRACTION_USER_TEMPLATE = """주어진 텍스트에서 중요한 엔티티(Entity)와 그들 간의 관계(Relationship)를 추출하세요.

## 추출 가이드라인
1. 엔티티: 사람, 조직, 장소, 개념, 사건, 날짜 등 중요한 명사구.
2. 관계: 엔티티 사이의 상호작용이나 속성을 나타내는 동사구.
3. 형식: (주체) -[관계]-> (목적어)

## 텍스트
{text}

## 출력 형식 (JSON)
{{ 
  "entities": [
    {{"name": "엔티티이름", "type": "유형", "description": "한줄설명"}}
  ],
  "relationships": [
    {{"source": "주체", "target": "목적어", "type": "관계유형", "description": "관계설명"}}
  ]
}}"""

# JSON 스키마 (vLLM guided decoding용)
GRAPH_JSON_SCHEMA = { 
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "type"]
            }
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["source", "target", "type"]
            }
        }
    },
    "required": ["entities", "relationships"]
}

# ======================================== 
# 모델 로드
# ======================================== 
def load_colbert():
    global colbert_model
    print(f"🚀 Loading Jina ColBERT v2 on {DEVICE}...")
    colbert_model = models.ColBERT(
        model_name_or_path=COLBERT_MODEL_NAME,
        device=DEVICE,
        trust_remote_code=True
    )
    print("✅ ColBERT loaded!")

def load_qwen():
    global qwen_model, qwen_tokenizer
    print(f"🚀 Loading Qwen3-4B with vLLM on {DEVICE}...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    qwen_model = LLM(
        model=QWEN_MODEL_NAME,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096
    )
    print("✅ Qwen3-4B loaded with vLLM!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_colbert()
    load_qwen()
    yield
    print("Shutting down model server...")

app = FastAPI(lifespan=lifespan)

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
        embeddings = colbert_model.encode(
            request.texts,
            batch_size=32 if request.is_query else 128,
            is_query=request.is_query,
            show_progress_bar=True
        )
        if hasattr(embeddings, 'cpu'):
            embeddings_list = embeddings.cpu().numpy().tolist()
        elif isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = embeddings
        return EncodeResponse(embeddings=embeddings_list)
    except Exception as e:
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
    if qwen_model is None or qwen_tokenizer is None:
        raise HTTPException(status_code=503, detail="Qwen model not loaded")
    
    prompts = []
    for text in request.texts:
        truncated = text[:1500]
        messages = [
            {"role": "system", "content": KEYWORD_SYSTEM_PROMPT},
            {"role": "user", "content": KEYWORD_USER_TEMPLATE.format(text=truncated)}
        ]
        prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0, max_tokens=100)
    outputs = qwen_model.generate(prompts, sampling_params)
    return {"keywords": [o.outputs[0].text.strip().split('\n')[0] for o in outputs]}

@app.post("/extract_graph")
async def extract_graph(request: GraphExtractRequest):
    """지식 그래프용 엔티티 및 관계 추출 (경량 모델 활용)"""
    if qwen_model is None or qwen_tokenizer is None:
        raise HTTPException(status_code=503, detail="Qwen model not loaded")
    
    if not request.texts:
        return {"results": []}

    prompts = []
    for text in request.texts:
        truncated = text[:2000]
        messages = [
            {"role": "system", "content": ENTITY_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": ENTITY_EXTRACTION_USER_TEMPLATE.format(text=truncated)}
        ]
        prompt = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    # Guided Decoding (JSON) 적용
    sampling_params = SamplingParams(
        temperature=0, 
        max_tokens=2048,
        guided_json=GRAPH_JSON_SCHEMA
    )

    try:
        outputs = qwen_model.generate(prompts, sampling_params)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            try:
                # JSON 파싱 시도
                graph_data = json.loads(generated_text)
                results.append(graph_data)
            except json.JSONDecodeError:
                print(f"❌ JSON Parsing failed for: {generated_text[:100]}...")
                results.append({"entities": [], "relationships": []})
        
        return {"results": results}

    except Exception as e:
        print(f"❌ Graph extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "colbert_loaded": colbert_model is not None,
        "qwen_loaded": qwen_model is not None,
        "device": DEVICE
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)