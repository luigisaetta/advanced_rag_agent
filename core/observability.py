"""
Langfuse observability adapter.

This module replaces the previous py-zipkin integration while keeping a
compatible `zipkin_span(...)` API for existing decorators.
"""

from __future__ import annotations

import os
import importlib
from functools import wraps
from typing import Any, Callable

import config

try:
    from config_private import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
except Exception:  # pragma: no cover - optional private config in some contexts
    LANGFUSE_PUBLIC_KEY = ""
    LANGFUSE_SECRET_KEY = ""

langfuse_context = None
observe = None
_get_langfuse_client = None

try:
    decorators_module = importlib.import_module("langfuse.decorators")
    langfuse_context = getattr(decorators_module, "langfuse_context")
    observe = getattr(decorators_module, "observe")
    from langfuse import get_client as _get_langfuse_client

    _LANGFUSE_AVAILABLE = True
    _LANGFUSE_USE_DECORATORS = True
except Exception:  # pragma: no cover - package may not be installed in all envs
    try:
        # Langfuse >=4 exposes observe/get_client at package top-level.
        from langfuse import observe, get_client as _get_langfuse_client

        _LANGFUSE_AVAILABLE = True
        _LANGFUSE_USE_DECORATORS = False
    except Exception:
        _LANGFUSE_AVAILABLE = False
        _LANGFUSE_USE_DECORATORS = False


class Encoding:
    """Compatibility shim for old py-zipkin imports."""

    V2_JSON = "v2_json"


def _is_enabled() -> bool:
    return bool(config.ENABLE_TRACING)


def _is_configured() -> bool:
    return bool(config.LANGFUSE_HOST and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


def _bootstrap_langfuse_env() -> None:
    """Expose Langfuse settings through env vars expected by SDK internals."""
    os.environ.setdefault("LANGFUSE_HOST", config.LANGFUSE_HOST)
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", LANGFUSE_PUBLIC_KEY)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", LANGFUSE_SECRET_KEY)


def _get_client():
    """Return current Langfuse client instance if available/configured."""
    if not (_LANGFUSE_AVAILABLE and _is_enabled() and _is_configured()):
        return None
    _bootstrap_langfuse_env()
    if _get_langfuse_client is None:
        return None
    try:
        # v4 initializes default client from LANGFUSE_* env vars.
        return _get_langfuse_client()
    except Exception:
        try:
            # Backward-compatible fallback for variants expecting explicit key.
            return _get_langfuse_client(public_key=LANGFUSE_PUBLIC_KEY)
        except Exception:
            return None


def annotate_current_observation(
    *,
    input_data: Any | None = None,
    output_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
    level: str | None = None,
    status_message: str | None = None,
) -> None:
    """Attach structured annotations to the active Langfuse observation."""
    if not (_LANGFUSE_AVAILABLE and _is_enabled() and _is_configured()):
        return
    _bootstrap_langfuse_env()

    payload: dict[str, Any] = {}
    if input_data is not None:
        payload["input"] = input_data
    if output_data is not None:
        payload["output"] = output_data
    if metadata is not None:
        payload["metadata"] = metadata
    if level:
        payload["level"] = level
    if status_message:
        payload["status_message"] = status_message

    try:
        if _LANGFUSE_USE_DECORATORS and langfuse_context is not None:
            langfuse_context.update_current_observation(**payload)
            return

        client = _get_client()
        if client is not None:
            try:
                # Avoid noisy warnings when called outside an active observation context.
                if not client.get_current_observation_id():
                    return
            except Exception:
                return
            client.update_current_span(**payload)
    except Exception:
        # Never break agent execution because of observability.
        return


def flush_observability() -> None:
    """Force flush buffered observability events to Langfuse."""
    if not (_LANGFUSE_AVAILABLE and _is_enabled() and _is_configured()):
        return
    try:
        client = _get_client()
        if client is not None:
            client.flush()
    except Exception:
        return


def zipkin_span(
    *,
    service_name: str,
    span_name: str,
    transport_handler: Callable[..., Any] | None = None,
    encoding: Any | None = None,
    sample_rate: float = 100,
) -> Any:
    """
    Backward-compatible span decorator/context manager.

    The old zipkin-only params are accepted for compatibility and ignored.
    """
    del transport_handler, encoding, sample_rate

    class _SpanContext:
        def __init__(self, enabled: bool):
            self.enabled = enabled
            self._ctx_manager = None

        def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
            if not self.enabled:
                return func

            traced = observe(name=span_name, capture_input=True, capture_output=True)(
                func
            )

            @wraps(traced)
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                try:
                    if _LANGFUSE_USE_DECORATORS and langfuse_context is not None:
                        langfuse_context.update_current_trace(tags=[service_name])
                        langfuse_context.update_current_observation(
                            metadata={"service_name": service_name}
                        )
                except Exception:
                    pass
                return traced(*args, **kwargs)

            return _wrapped

        def __enter__(self) -> Any:
            if not self.enabled:
                return None
            if _LANGFUSE_USE_DECORATORS:
                # v3/decorators path does not provide an explicit context manager
                # in this adapter, so keep this as best-effort no-op.
                return None
            client = _get_client()
            if client is None:
                return None
            try:
                self._ctx_manager = client.start_as_current_observation(
                    name=span_name,
                    as_type="span",
                    metadata={"service_name": service_name},
                    end_on_exit=True,
                )
                return self._ctx_manager.__enter__()
            except Exception:
                self._ctx_manager = None
                return None

        def __exit__(self, exc_type, exc, tb) -> bool:
            if self._ctx_manager is not None:
                try:
                    self._ctx_manager.__exit__(exc_type, exc, tb)
                except Exception:
                    pass
            flush_observability()
            return False

    enabled = _LANGFUSE_AVAILABLE and _is_enabled() and _is_configured()
    if enabled:
        _bootstrap_langfuse_env()
    return _SpanContext(enabled=enabled)
