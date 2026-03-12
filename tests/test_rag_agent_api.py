"""
Unit tests for rag_agent_api helpers and endpoint behavior.
"""

import asyncio
import json

import rag_agent_api as api_module


def test_safe_json_serializes_unserializable_objects():
    """safe_json should fallback to string conversion for unknown objects."""

    class _Obj:
        def __str__(self):
            return "obj-as-string"

    out = api_module.safe_json({"k": _Obj()})
    parsed = json.loads(out)
    assert parsed == {"k": "obj-as-string"}


def test_generate_request_id_returns_non_empty_unique_values():
    """Generated request ids should be non-empty and unique."""
    a = api_module.generate_request_id()
    b = api_module.generate_request_id()
    assert isinstance(a, str) and a
    assert isinstance(b, str) and b
    assert a != b


def test_stream_graph_updates_yields_json_lines(monkeypatch):
    """Graph stream updates are emitted as JSONL chunks."""

    class _FakeGraph:
        async def astream(self, state, config=None):  # noqa: ARG002
            assert state["user_request"] == "hello"
            assert state["chat_history"] == []
            yield {"StepA": {"ok": True}}
            yield {"StepB": {"value": 2}}

    monkeypatch.setattr(api_module, "agent_graph", _FakeGraph())

    async def _collect():
        out = []
        async for chunk in api_module.stream_graph_updates(
            "hello", config={"configurable": {"k": "v"}}
        ):
            out.append(chunk)
        return out

    chunks = asyncio.run(_collect())
    assert len(chunks) == 2
    assert chunks[0].endswith("\n")
    assert json.loads(chunks[0].strip()) == {"StepA": {"ok": True}}
    assert json.loads(chunks[1].strip()) == {"StepB": {"value": 2}}


def test_invoke_returns_streaming_response(monkeypatch):
    """invoke endpoint should return a StreamingResponse on success."""
    monkeypatch.setattr(api_module, "generate_request_id", lambda: "req-123")

    req = api_module.InvokeRequest(user_input="Explain this")
    response = asyncio.run(api_module.invoke(req))

    assert response.__class__.__name__ == "StreamingResponse"
    assert response.media_type == api_module.MEDIA_TYPE


def test_invoke_returns_error_payload_on_stream_setup_failure(monkeypatch):
    """invoke should return error payload when stream setup raises."""

    def _boom(_user_input, _config=None):
        raise RuntimeError("broken stream")

    monkeypatch.setattr(api_module, "stream_graph_updates", _boom)

    req = api_module.InvokeRequest(user_input="Explain this")
    response = asyncio.run(api_module.invoke(req))

    assert isinstance(response, dict)
    assert response["error"] == "broken stream"
