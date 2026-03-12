"""
Unit tests for core.user_profile.
"""

import core.user_profile as user_profile_module


class _FakeCursor:
    def __init__(self, row=None, execute_exc=None):
        """Configure returned row or execute exception."""
        self._row = row
        self._execute_exc = execute_exc
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, **kwargs):
        """Capture execute calls and optionally raise."""
        self.executed.append((sql, kwargs))
        if self._execute_exc is not None:
            raise self._execute_exc

    def fetchone(self):
        """Return configured row."""
        return self._row


class _FakeConnection:
    def __init__(self, cursor):
        """Wrap provided cursor in a context-managed connection."""
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        """Return configured cursor."""
        return self._cursor


def test_get_user_profile_returns_default_for_empty_username():
    """Empty username returns default profile."""
    out = user_profile_module.get_user_profile("", default_profile="ADMIN")
    assert out == "ADMIN"


def test_get_user_profile_sanitizes_invalid_default_profile():
    """Unsupported default profile should fallback to USER."""
    out = user_profile_module.get_user_profile("", default_profile="unknown")
    assert out == "USER"


def test_get_user_profile_returns_default_on_db_exception(monkeypatch):
    """Database errors should return default profile."""
    cursor = _FakeCursor(row=None, execute_exc=RuntimeError("db error"))
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(user_profile_module, "get_connection", lambda: conn)

    out = user_profile_module.get_user_profile("alice", default_profile="ADMIN")
    assert out == "ADMIN"


def test_get_user_profile_returns_default_when_no_row(monkeypatch):
    """Missing user row returns default profile."""
    cursor = _FakeCursor(row=None)
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(user_profile_module, "get_connection", lambda: conn)

    out = user_profile_module.get_user_profile("alice", default_profile="USER")
    assert out == "USER"


def test_get_user_profile_returns_default_when_disabled(monkeypatch):
    """Disabled profile should fallback to default."""
    cursor = _FakeCursor(row=("ADMIN", 0))
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(user_profile_module, "get_connection", lambda: conn)

    out = user_profile_module.get_user_profile("alice", default_profile="USER")
    assert out == "USER"


def test_get_user_profile_returns_default_when_invalid_profile_code(monkeypatch):
    """Unknown profile_code should fallback to default."""
    cursor = _FakeCursor(row=("SUPERUSER", 1))
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(user_profile_module, "get_connection", lambda: conn)

    out = user_profile_module.get_user_profile("alice", default_profile="ADMIN")
    assert out == "ADMIN"


def test_get_user_profile_returns_uppercase_valid_profile(monkeypatch):
    """Valid enabled profile should be returned in canonical uppercase."""
    cursor = _FakeCursor(row=("admin", 1))
    conn = _FakeConnection(cursor)
    monkeypatch.setattr(user_profile_module, "get_connection", lambda: conn)

    out = user_profile_module.get_user_profile("alice", default_profile="USER")
    assert out == "ADMIN"
