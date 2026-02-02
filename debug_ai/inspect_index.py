import asyncio
import os
import sys

# 프로젝트 루트 경로 추가 (모듈 임포트용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.api_clients.model_server_client import model_server_client

async def inspect_document_chunks(invoke_id: str, target_filename: str):
    print(f"=== 문서 조회 시작: {target_filename} (Room: {invoke_id}) ===")

    # 1. 문서 메타데이터 조회
    metadata = await model_server_client.colbert_get_document_metadata(invoke_id, target_filename)
    if not metadata:
        print(f"Error: 문서를 찾을 수 없습니다.")
        return

    print(f"\n[문서 정보]")
    print(f"- 파일명: {metadata['file_name']}")
    print(f"- 총 페이지: {metadata['total_pages']}")
    print(f"- 총 청크 수: {metadata['total_chunks']}")
    print(f"- 저장 시각: {metadata['ingested_at']}")

    # 2. 문서 목록 조회
    documents = await model_server_client.colbert_list_documents(invoke_id)

    print(f"\n[인덱싱된 문서 목록 ({len(documents)}개)]")
    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1} ---")
        print(f"- 파일명: {doc.get('file_name', '(없음)')}")
        print(f"- 총 페이지: {doc.get('total_pages', 0)}")
        print(f"- 총 청크 수: {doc.get('total_chunks', 0)}")
        print("-" * 50)

if __name__ == "__main__":
    # 테스트할 방 ID와 파일명 설정
    # (사용자가 업로드했던 방 ID와 파일명을 여기에 입력하세요)
    INVOKE_ID = "test-room-001"
    TARGET_FILENAME = "KorQuAD_2.0_paper.pdf"

    asyncio.run(inspect_document_chunks(INVOKE_ID, TARGET_FILENAME))
