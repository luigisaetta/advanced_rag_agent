"""
Unit tests for core.transport.
"""

import core.transport as transport_module


class _FakeResponse:
    def __init__(self, raise_exc=None):
        """Initialize fake response."""
        self._raise_exc = raise_exc

    def raise_for_status(self):
        """Raise configured exception or succeed."""
        if self._raise_exc is not None:
            raise self._raise_exc


def test_http_transport_success(monkeypatch):
    """Sends span payload when tracing is enabled and config is valid."""
    called = {}

    def _fake_post(url, data, headers, timeout):
        called["url"] = url
        called["data"] = data
        called["headers"] = headers
        called["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(transport_module.config, "ENABLE_TRACING", True)
    monkeypatch.setattr(transport_module.config, "APM_BASE_URL", "https://apm.example")
    monkeypatch.setattr(transport_module.config, "APM_CONTENT_TYPE", "application/json")
    monkeypatch.setattr(transport_module, "APM_PUBLIC_KEY", "public-key")
    monkeypatch.setattr(transport_module.requests, "post", _fake_post)

    payload = b"zipkin-span"
    out = transport_module.http_transport(payload)

    assert isinstance(out, _FakeResponse)
    assert called["url"].startswith("https://apm.example/observations/public-span")
    assert "dataKey=public-key" in called["url"]
    assert called["data"] == payload
    assert called["headers"] == {"Content-Type": "application/json"}
    assert called["timeout"] == 30


def test_http_transport_returns_none_when_tracing_disabled(monkeypatch):
    """Does not call requests.post when tracing is disabled."""
    calls = {"post": 0}

    def _fake_post(*_args, **_kwargs):
        calls["post"] += 1
        return _FakeResponse()

    monkeypatch.setattr(transport_module.config, "ENABLE_TRACING", False)
    monkeypatch.setattr(transport_module.config, "APM_BASE_URL", "https://apm.example")
    monkeypatch.setattr(transport_module.config, "APM_CONTENT_TYPE", "application/json")
    monkeypatch.setattr(transport_module, "APM_PUBLIC_KEY", "public-key")
    monkeypatch.setattr(transport_module.requests, "post", _fake_post)

    out = transport_module.http_transport(b"x")
    assert out is None
    assert calls["post"] == 0


def test_http_transport_returns_none_when_base_url_missing(monkeypatch):
    """Invalid base URL is handled gracefully."""
    monkeypatch.setattr(transport_module.config, "ENABLE_TRACING", True)
    monkeypatch.setattr(transport_module.config, "APM_BASE_URL", "")
    monkeypatch.setattr(transport_module, "APM_PUBLIC_KEY", "public-key")

    assert transport_module.http_transport(b"x") is None


def test_http_transport_returns_none_when_public_key_missing(monkeypatch):
    """Missing public key is handled gracefully."""
    monkeypatch.setattr(transport_module.config, "ENABLE_TRACING", True)
    monkeypatch.setattr(transport_module.config, "APM_BASE_URL", "https://apm.example")
    monkeypatch.setattr(transport_module, "APM_PUBLIC_KEY", "")

    assert transport_module.http_transport(b"x") is None


def test_http_transport_handles_request_exception(monkeypatch):
    """Request exceptions should be caught and return None."""

    def _fake_post(*_args, **_kwargs):
        raise transport_module.requests.RequestException("network error")

    monkeypatch.setattr(transport_module.config, "ENABLE_TRACING", True)
    monkeypatch.setattr(transport_module.config, "APM_BASE_URL", "https://apm.example")
    monkeypatch.setattr(transport_module.config, "APM_CONTENT_TYPE", "application/json")
    monkeypatch.setattr(transport_module, "APM_PUBLIC_KEY", "public-key")
    monkeypatch.setattr(transport_module.requests, "post", _fake_post)

    assert transport_module.http_transport(b"x") is None


def test_http_transport_handles_http_error_from_raise_for_status(monkeypatch):
    """HTTP errors from response.raise_for_status are handled."""

    def _fake_post(*_args, **_kwargs):
        return _FakeResponse(
            raise_exc=transport_module.requests.RequestException("http 500")
        )

    monkeypatch.setattr(transport_module.config, "ENABLE_TRACING", True)
    monkeypatch.setattr(transport_module.config, "APM_BASE_URL", "https://apm.example")
    monkeypatch.setattr(transport_module.config, "APM_CONTENT_TYPE", "application/json")
    monkeypatch.setattr(transport_module, "APM_PUBLIC_KEY", "public-key")
    monkeypatch.setattr(transport_module.requests, "post", _fake_post)

    assert transport_module.http_transport(b"x") is None
