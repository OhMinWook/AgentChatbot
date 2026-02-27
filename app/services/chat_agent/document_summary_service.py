"""
문서 요약 서비스

대용량/소형 문서 요약 비즈니스 로직을 담당합니다.
- summarize_large: 질문 분해 → 병렬 검색 → 병렬 LLM → 통합 payload 반환
- summarize_small: 단순 검색 → 요약 payload 반환
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

from app.core.config import settings
from app.services.api_clients.llm_client import llm_client
from app.services.prompt_builders.document_summary_prompt_builder import document_summary_prompt_builder

logger = logging.getLogger(__name__)


class DocumentSummaryService:

    async def summarize_large(
        self,
        search_tool,
        filename: str,
        on_progress=None,
    ) -> Tuple[Dict, List]:
        """대용량 문서 요약 (질문 분해 → 병렬 검색 → 병렬 LLM → 통합)
            병렬 LLM => 문서 처리를 위해 같은 LLM을 여러 번 호출해서 처리하는 방식
        Returns:
            (merge_payload, all_references)
        """
        # 프롬프트 빌더 호출
        questions = document_summary_prompt_builder.get_questions()

        # 진행도
        if on_progress:
            await on_progress(10, "문서 내용을 분석하고 있습니다")
        if on_progress:
            await on_progress(20, "관련 문서 검색 중...")

        query_texts = [q["question"] for q in questions]
        batch_results = await search_tool.search_batch(
            query_texts, top_k=settings.SEARCH_TOP_K, filter_filename=filename
        )

        if on_progress:
            await on_progress(40, "문서 내용을 정리하고 있습니다")

        all_references = []
        llm_tasks = []

        for i, q in enumerate(questions):
            search_result = batch_results[i] if i < len(batch_results) else {"results": [], "references": []}
            filtered_chunks = search_result.get("results", [])
            refs = search_result.get("references", [])

            for ref in refs:
                if ref not in all_references:
                    all_references.append(ref)

            if filtered_chunks:
                context = "\n\n---\n\n".join([c.get("content", "") for c in filtered_chunks])
                payload = document_summary_prompt_builder.build_qa_payload(q["question"], context)
                llm_tasks.append((q["key"], llm_client.chat_completions(payload)))
            else:
                llm_tasks.append((q["key"], None))

        # 병렬 LLM 실행(같은 LLM을 여러 번 호출하는 방식)
        qa_results = {}
        async_tasks = [task for key, task in llm_tasks if task is not None]
        task_keys = [key for key, task in llm_tasks if task is not None]

        if async_tasks:
            responses = await asyncio.gather(*async_tasks, return_exceptions=True)
            for key, response in zip(task_keys, responses):
                if isinstance(response, Exception):
                    logger.error(f"[DocSummary] LLM 호출 실패 ({key}): {response}")
                    qa_results[key] = "분석 실패"
                else:
                    qa_results[key] = llm_client.extract_content(response)

        for key, task in llm_tasks:
            if task is None:
                qa_results[key] = "해당 정보 없음"

        if on_progress:
            await on_progress(85, "요약을 작성하고 있습니다")

        merge_payload = document_summary_prompt_builder.build_merge_payload(qa_results, filename)

        return merge_payload, all_references

    async def summarize_small(
        self,
        search_tool,
        filename: str,
        on_progress=None,
    ) -> Tuple[Optional[Dict], List]:
        """소형 문서 요약 (단순 검색 → 요약 payload)

        Returns:
            (summary_payload, all_references) - payload가 None이면 검색 결과 없음
        """
        if on_progress:
            await on_progress(20, "문서 내용을 검색하고 있습니다")

        search_query = f"{filename} 요약"
        search_result = await search_tool.search(
            search_query, top_k=settings.SEARCH_TOP_K, filter_filename=filename
        )

        filtered_chunks = search_result.get("results", [])
        all_references = search_result.get("references", [])

        if not filtered_chunks:
            return None, []

        context = "\n\n---\n\n".join([c.get("content", "") for c in filtered_chunks])

        if on_progress:
            await on_progress(50, "요약 생성 중...")

        payload = document_summary_prompt_builder.build_simple_summary_payload(context, filename)

        return payload, all_references


# 싱글톤 인스턴스
document_summary_service = DocumentSummaryService()
