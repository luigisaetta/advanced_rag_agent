"""
Unit tests for core.retry_utils.
"""

import pytest

import core.retry_utils as retry_module


def test_is_retryable_llm_exception_detects_markers():
    """Known transient markers should be classified as retryable."""
    assert retry_module.is_retryable_llm_exception(Exception("429 too many requests"))
    assert retry_module.is_retryable_llm_exception(
        Exception("Content filter triggered")
    )
    assert not retry_module.is_retryable_llm_exception(Exception("syntax error"))


def test_run_with_retry_returns_on_first_success():
    """No retry is performed when operation succeeds immediately."""
    out = retry_module.run_with_retry(
        operation=lambda: "ok",
        max_attempts=3,
        operation_name="op",
    )
    assert out == "ok"


def test_run_with_retry_retries_then_succeeds(monkeypatch):
    """Retryable failure should sleep and retry until success."""
    attempts = {"n": 0}
    sleeps = []
    monkeypatch.setattr(retry_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)

    def _operation():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("timeout while calling model")
        return "done"

    out = retry_module.run_with_retry(
        operation=_operation,
        max_attempts=4,
        operation_name="op",
    )

    assert out == "done"
    assert attempts["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_run_with_retry_raises_non_retryable_without_sleep(monkeypatch):
    """Non-retryable failures should be raised immediately."""
    sleeps = []
    monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="bad input"):
        retry_module.run_with_retry(
            operation=lambda: (_ for _ in ()).throw(RuntimeError("bad input")),
            max_attempts=4,
            operation_name="op",
        )

    assert not sleeps


def test_run_with_retry_raises_after_max_attempts(monkeypatch):
    """Retryable failures should stop at max attempts and re-raise."""
    attempts = {"n": 0}
    monkeypatch.setattr(retry_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(retry_module.time, "sleep", lambda _seconds: None)

    def _operation():
        attempts["n"] += 1
        raise RuntimeError("503 service unavailable")

    with pytest.raises(RuntimeError, match="503"):
        retry_module.run_with_retry(
            operation=_operation,
            max_attempts=2,
            operation_name="op",
        )

    assert attempts["n"] == 2


def test_run_with_retry_rejects_invalid_max_attempts():
    """max_attempts must be >= 1."""
    with pytest.raises(ValueError, match="max_attempts"):
        retry_module.run_with_retry(
            operation=lambda: "x",
            max_attempts=0,
            operation_name="op",
        )


def test_stream_with_retry_retries_before_first_chunk(monkeypatch):
    """Stream creation can be retried when nothing was emitted yet."""
    attempts = {"n": 0}
    sleeps = []
    monkeypatch.setattr(retry_module.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)

    def _factory():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("temporarily unavailable")
        yield "a"
        yield "b"

    out = list(
        retry_module.stream_with_retry(
            stream_factory=_factory,
            max_attempts=3,
            operation_name="stream-op",
        )
    )

    assert out == ["a", "b"]
    assert attempts["n"] == 2
    assert sleeps == [1.0]


def test_stream_with_retry_does_not_retry_after_partial_emit(monkeypatch):
    """If a chunk was already emitted, exception must propagate directly."""
    attempts = {"n": 0}
    sleeps = []
    monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)

    def _factory():
        attempts["n"] += 1
        yield "first"
        raise RuntimeError("503 downstream")

    gen = retry_module.stream_with_retry(
        stream_factory=_factory,
        max_attempts=3,
        operation_name="stream-op",
    )
    assert next(gen) == "first"
    with pytest.raises(RuntimeError, match="503"):
        next(gen)

    assert attempts["n"] == 1
    assert not sleeps


def test_stream_with_retry_rejects_invalid_max_attempts():
    """max_attempts must be >= 1 for stream retry as well."""
    with pytest.raises(ValueError, match="max_attempts"):
        list(
            retry_module.stream_with_retry(
                stream_factory=lambda: iter(()),
                max_attempts=0,
                operation_name="stream-op",
            )
        )
