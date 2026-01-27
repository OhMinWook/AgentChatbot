
import os
import json
import asyncio
import re
import time
from typing import List, Dict
import sys

# 프로젝트 루트 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from app.services.rag.rag_ingestion_service import rag_ingestion_service
from app.services.rag.lightrag_service import lightrag_service

# 설정
DB_DIR_PATH = os.path.join(SCRIPT_DIR, "RAG_test_DB")
INVOKE_ID = "RAG_Test"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "uploaded_files", INVOKE_ID)
TEST_CASES_PATH = os.path.join(SCRIPT_DIR, "korquad_test_questions.json")
MAX_DOCS_PER_FILE = float('inf')
CONCURRENT_LIMIT = 1  # 순차 실행으로 변경

def clean_html(raw_html):
    """HTML 태그 제거 및 텍스트 정제"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = cleantext.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    cleantext = re.sub(r'\n\s*\n', '\n\n', cleantext)
    return cleantext.strip()

async def ingest_doc(semaphore, item, doc_index, total_count, all_test_cases, json_file):
    async with semaphore:
        title = item.get('title', 'No Title').replace('/', '_').replace('\\', '_')
        context_html = item.get('context', '')
        qas = item.get('qas', [])

        print(f"\n{'='*60}")
        print(f"[DEBUG] Document {doc_index + 1}/{total_count}")
        print(f"[DEBUG] Title: {title}")

        if not context_html:
            print(f"[DEBUG] SKIP - Empty context")
            return

        # 1. HTML -> Markdown 변환 및 저장
        start_time = time.time()
        markdown_content = clean_html(context_html)
        file_name = f"{title}.md"
        output_file_path = os.path.join(OUTPUT_DIR, file_name)

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n{markdown_content}")
        
        # 2. 테스트 케이스 수집
        doc_test_cases = {
            "source_file": json_file,
            "title": title,
            "questions": [
                {
                    "q": qa.get('question'),
                    "a": qa.get('answer', {}).get('text')
                } for qa in qas
            ]
        }
        all_test_cases.append(doc_test_cases)

        # 3. ColBERT 및 LightRAG 인덱싱
        try:
            print(f"[DEBUG] >>> Starting Ingestion for: {title}")
            
            # (1) ColBERT (ingest_file 사용 - LightRAG는 내부에서 꺼져있음)
            await rag_ingestion_service.ingest_file(
                file_name=file_name,
                invoke_id=INVOKE_ID,
                file_path=output_file_path
            )
            print(f"[DEBUG] ✅ ColBERT Done")

            # (2) LightRAG (수동 실행)
            # 청크 생성 (rag_ingestion_service의 내부 메서드 활용)
            # ingest_file 내에서 이미 생성했지만, 외부에서 접근 어려우므로 다시 생성 (비용 거의 없음)
            chunks = await rag_ingestion_service._create_chunks_with_keywords(
                f"# {title}\n\n{markdown_content}", 
                file_name
            )
            
            if chunks:
                print(f"[DEBUG] >>> Starting LightRAG indexing ({len(chunks)} chunks)...")
                await lightrag_service.index_chunks(chunks, INVOKE_ID)
                print(f"[DEBUG] ✅ LightRAG Done")
            else:
                print(f"[DEBUG] ⚠️ No chunks created for LightRAG")

        except Exception as e:
            print(f"[DEBUG] !!! Ingestion FAILED: {e}")
            import traceback
            traceback.print_exc()

        total_time = time.time() - start_time
        print(f"[DEBUG] Total time: {total_time:.2f}s")
        print(f"{'='*60}\n")

async def process_korquad():
    print("\n" + "#"*60)
    print("# LightRAG + ColBERT Ingestion Test")
    print("#"*60)
    # ... (rest of the main logic remains similar, relying on modified ingest_doc)
    
    if not os.path.exists(DB_DIR_PATH):
        print(f"[ERROR] Directory not found: {DB_DIR_PATH}")
        return

    json_files = [f for f in os.listdir(DB_DIR_PATH) if f.endswith('.json')]
    if not json_files:
        print("[ERROR] No JSON files found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    all_test_cases = []
    # semaphore = asyncio.Semaphore(CONCURRENT_LIMIT) # 순차 실행이므로 세마포어 불필요
    
    # tasks = []
    total_start_time = time.time()

    processed_count = 0
    for json_file in json_files:
        file_path = os.path.join(DB_DIR_PATH, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file}: {e}")
            continue

        korquad_data = data.get('data', [])
        docs_to_process = min(len(korquad_data), MAX_DOCS_PER_FILE)

        for i, item in enumerate(korquad_data):
            if i >= MAX_DOCS_PER_FILE:
                break
            
            # 순차 실행 (await로 바로 실행)
            # ingest_doc 함수 서명 변경 필요 없이 semaphore에 None 전달하거나 더미 사용
            # 하지만 ingest_doc 내부에서 semaphore를 쓰므로, 더미 세마포어를 하나 만듭니다.
            dummy_semaphore = asyncio.Semaphore(1)
            await ingest_doc(dummy_semaphore, item, i, docs_to_process, all_test_cases, json_file)
            processed_count += 1
            
            if processed_count >= 70:
                print(f"\n[STOP] Reached limit of 70 documents.")
                break
        
        if processed_count >= 70:
            break

    # print(f"\n[DEBUG] Created {len(tasks)} async tasks. Processing...")
    # await asyncio.gather(*tasks)

    # 결과 저장
    with open(TEST_CASES_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_test_cases, f, ensure_ascii=False, indent=2)

    total_time = time.time() - total_start_time
    print(f"\n[SUMMARY] Total documents: {len(all_test_cases)}")
    print(f"[SUMMARY] Total time: {total_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(process_korquad())
