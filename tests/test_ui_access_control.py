"""
Unit tests for ui.access_control helpers.
"""

import pytest

import ui.access_control as access_module


class _SessionState(dict):
    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    def __init__(self):
        self.session_state = _SessionState()
        self.markdown_calls = []
        self.error_calls = []
        self.stop_calls = 0

    def markdown(self, text, unsafe_allow_html=False):
        self.markdown_calls.append((text, unsafe_allow_html))

    def error(self, text):
        self.error_calls.append(text)

    def stop(self):
        self.stop_calls += 1
        raise RuntimeError("st.stop")


def test_resolve_session_user_profile_caches_until_user_changes(monkeypatch):
    """Profile is cached and refreshed only when username changes."""
    fake_st = _FakeStreamlit()
    fake_st.session_state.authenticated_user = "ALICE"

    calls = []
    monkeypatch.setattr(access_module, "st", fake_st)
    monkeypatch.setattr(
        access_module, "get_authenticated_user", lambda username: str(username).lower()
    )
    monkeypatch.setattr(
        access_module,
        "get_user_profile",
        lambda username: calls.append(username) or ("ADMIN" if username == "alice" else "USER"),
    )

    assert access_module._resolve_session_user_profile() == "ADMIN"
    assert access_module._resolve_session_user_profile() == "ADMIN"
    assert calls == ["alice"]

    fake_st.session_state.authenticated_user = "BOB"
    assert access_module._resolve_session_user_profile() == "USER"
    assert calls == ["alice", "bob"]


def test_hide_admin_only_pages_in_sidebar_for_non_admin(monkeypatch):
    """Non-admin users get CSS rules to hide admin-only pages."""
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(access_module, "st", fake_st)
    monkeypatch.setattr(access_module, "is_admin_user", lambda: False)

    access_module.hide_admin_only_pages_in_sidebar()

    assert len(fake_st.markdown_calls) == 1
    css, unsafe = fake_st.markdown_calls[0]
    assert unsafe is True
    assert "loader_ui" in css
    assert "post_answer_eval_ui" in css


def test_require_admin_page_access_blocks_non_admin(monkeypatch):
    """Non-admin users are blocked via st.error + st.stop."""
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(access_module, "st", fake_st)
    monkeypatch.setattr(access_module, "is_admin_user", lambda: False)

    with pytest.raises(RuntimeError, match="st.stop"):
        access_module.require_admin_page_access()

    assert fake_st.error_calls == [
        "Access denied. This page is available only for ADMIN users."
    ]
    assert fake_st.stop_calls == 1
