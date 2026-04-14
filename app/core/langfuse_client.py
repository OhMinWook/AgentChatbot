"""
Langfuse 비활성화 스텁
langfuse v4에서 decorators 모듈이 제거됨 — 서버 실행을 위해 no-op으로 대체
"""

import functools
import logging

logger = logging.getLogger(__name__)


class _NoopLangfuse:
    def get_current_trace_id(self):
        return None

    def create_score(self, **kwargs):
        pass

    def score(self, **kwargs):
        pass

    def flush(self):
        pass


class _NoopContext:
    def get_current_trace_id(self):
        return None

    def update_current_observation(self, **kwargs):
        pass


langfuse = _NoopLangfuse()
langfuse_context = _NoopContext()


def observe(*args, **kwargs):
    """No-op observe 데코레이터 (langfuse.decorators.observe 대체)"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*a, **kw):
            return await func(*a, **kw)

        @functools.wraps(func)
        def sync_wrapper(*a, **kw):
            return func(*a, **kw)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    # @observe() 또는 @observe(capture_input=False) 형태 모두 지원
    if len(args) == 1 and callable(args[0]):
        return decorator(args[0])
    return decorator
