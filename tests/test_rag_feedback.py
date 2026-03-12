"""
Unit tests for core.rag_feedback.
"""

import pytest

from core.rag_feedback import RagFeedback


class _FakeCursor:
    def __init__(self, fetchone_value=(0,)):
        """Store fetch value and executed statements."""
        self._fetchone_value = fetchone_value
        self.executed = []
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, *args, **kwargs):
        """Capture SQL execution calls."""
        self.executed.append((sql, args, kwargs))

    def fetchone(self):
        """Return configured fetchone value."""
        return self._fetchone_value

    def close(self):
        """Mark cursor as closed."""
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor_obj: _FakeCursor):
        """Connection wrapper returning a deterministic cursor."""
        self._cursor_obj = cursor_obj
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        """Return configured cursor."""
        return self._cursor_obj

    def commit(self):
        """Record commit calls."""
        self.committed = True


def test_table_exists_true(monkeypatch):
    """table_exists returns True when USER_TABLES count is positive."""
    cursor = _FakeCursor(fetchone_value=(1,))
    conn = _FakeConnection(cursor)
    node = RagFeedback()
    monkeypatch.setattr(node, "get_connection", lambda: conn)

    assert node.table_exists("rag_feedback") is True
    assert len(cursor.executed) == 1
    _sql, _args, kwargs = cursor.executed[0]
    assert kwargs["tn"] == "RAG_FEEDBACK"


def test_table_exists_false(monkeypatch):
    """table_exists returns False when USER_TABLES count is zero."""
    cursor = _FakeCursor(fetchone_value=(0,))
    conn = _FakeConnection(cursor)
    node = RagFeedback()
    monkeypatch.setattr(node, "get_connection", lambda: conn)

    assert node.table_exists("RAG_FEEDBACK") is False


def test_insert_feedback_rejects_out_of_range_values():
    """Feedback must be in [1, 5]."""
    node = RagFeedback()
    for invalid in (0, 6):
        with pytest.raises(ValueError, match="between 1 and 5"):
            node.insert_feedback("q", "a", invalid)


def test_insert_feedback_creates_table_when_missing(monkeypatch):
    """When table is missing, insert_feedback creates table then inserts row."""
    cursor = _FakeCursor()
    conn = _FakeConnection(cursor)
    node = RagFeedback()
    calls = {"create_table": 0}

    monkeypatch.setattr(node, "table_exists", lambda _table: False)
    monkeypatch.setattr(node, "get_connection", lambda: conn)
    monkeypatch.setattr(
        node, "_create_table", lambda: calls.__setitem__("create_table", 1)
    )

    node.insert_feedback("question", "answer", 5)

    assert calls["create_table"] == 1
    assert conn.committed is True
    assert cursor.closed is True
    assert len(cursor.executed) == 1
    sql, args, kwargs = cursor.executed[0]
    assert "INSERT INTO RAG_FEEDBACK" in sql
    assert not kwargs
    params = args[0]
    assert params["question"] == "question"
    assert params["answer"] == "answer"
    assert params["feedback"] == 5


def test_insert_feedback_skips_create_table_when_present(monkeypatch):
    """When table exists, insert_feedback should not call _create_table."""
    cursor = _FakeCursor()
    conn = _FakeConnection(cursor)
    node = RagFeedback()
    calls = {"create_table": 0}

    monkeypatch.setattr(node, "table_exists", lambda _table: True)
    monkeypatch.setattr(node, "get_connection", lambda: conn)
    monkeypatch.setattr(
        node, "_create_table", lambda: calls.__setitem__("create_table", 1)
    )

    node.insert_feedback("q", "a", 1)

    assert calls["create_table"] == 0
    assert conn.committed is True
    assert len(cursor.executed) == 1
