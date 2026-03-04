"""
Retry helpers for LLM calls.
"""

import random
import time
from typing import Callable, Iterator, TypeVar

from core.utils import get_console_logger

logger = get_console_logger()

T = TypeVar("T")


def is_retryable_llm_exception(exc: Exception) -> bool:
    """
    Return True for transient/safety-filter style errors that are worth retrying.
    """
    msg = str(exc).lower()
    retryable_markers = (
        "inappropriate",
        "content filter",
        "safety",
        "temporarily unavailable",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "rate limit",
        "too many requests",
        "429",
        "502",
        "503",
        "504",
    )
    return any(marker in msg for marker in retryable_markers)


def run_with_retry(
    operation: Callable[[], T],
    max_attempts: int,
    operation_name: str,
) -> T:
    """
    Run operation with bounded retries for retryable exceptions.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_llm_exception(exc):
                raise
            wait_s = min(2 ** (attempt - 1), 4) + random.uniform(0, 0.3)
            logger.warning(
                "%s failed on attempt %d/%d, retrying in %.2fs. Error: %s",
                operation_name,
                attempt,
                max_attempts,
                wait_s,
                exc,
            )
            time.sleep(wait_s)

    raise RuntimeError(f"{operation_name} failed after retries")


def stream_with_retry(
    stream_factory: Callable[[], Iterator[T]],
    max_attempts: int,
    operation_name: str,
) -> Iterator[T]:
    """
    Retry stream creation/early failure before first token is emitted.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    for attempt in range(1, max_attempts + 1):
        emitted_any_chunk = False
        try:
            for chunk in stream_factory():
                emitted_any_chunk = True
                yield chunk
            return
        except Exception as exc:
            # If we already emitted content, retry would duplicate output.
            if emitted_any_chunk:
                raise
            if attempt >= max_attempts or not is_retryable_llm_exception(exc):
                raise
            wait_s = min(2 ** (attempt - 1), 4) + random.uniform(0, 0.3)
            logger.warning(
                "%s failed on attempt %d/%d, retrying in %.2fs. Error: %s",
                operation_name,
                attempt,
                max_attempts,
                wait_s,
                exc,
            )
            time.sleep(wait_s)
