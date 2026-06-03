import pytest
import json
from unittest.mock import MagicMock, patch, PropertyMock
from fastapi.testclient import TestClient


@pytest.fixture
def test_db(tmp_path):
    from app.db.database import Database
    db = Database(tmp_path / "test.db")
    return db


@pytest.fixture
def test_repo(test_db):
    from app.db.repo import Repo
    return Repo(test_db)


class TestMessageStatus:
    def test_add_message_default_complete(self, test_repo):
        sid = test_repo.create_session("test")
        msg_id = test_repo.add_message(sid, "user", "hello")
        msg = test_repo.get_message(msg_id)
        assert msg is not None
        assert msg["status"] == "complete"

    def test_add_message_processing(self, test_repo):
        sid = test_repo.create_session("test")
        msg_id = test_repo.add_message(sid, "assistant", "", status="processing")
        msg = test_repo.get_message(msg_id)
        assert msg is not None
        assert msg["status"] == "processing"

    def test_update_message_status(self, test_repo):
        sid = test_repo.create_session("test")
        msg_id = test_repo.add_message(sid, "assistant", "", status="processing")
        test_repo.update_message(msg_id, status="complete", content="Hello!")
        msg = test_repo.get_message(msg_id)
        assert msg["status"] == "complete"
        assert msg["content"] == "Hello!"

    def test_update_message_citations(self, test_repo):
        sid = test_repo.create_session("test")
        msg_id = test_repo.add_message(sid, "assistant", "answer", status="processing")
        cits = [{"label": 1, "text": "chunk text"}]
        test_repo.update_message(msg_id, citations=cits, status="complete")
        msg = test_repo.get_message(msg_id)
        assert msg["status"] == "complete"
        assert len(msg["citations"]) == 1

    def test_update_message_failed(self, test_repo):
        sid = test_repo.create_session("test")
        msg_id = test_repo.add_message(sid, "assistant", "", status="processing")
        test_repo.update_message(msg_id, status="failed", error_message="timeout")
        msg = test_repo.get_message(msg_id)
        assert msg["status"] == "failed"
        assert msg["error_message"] == "timeout"

    def test_get_processing_messages(self, test_repo):
        sid = test_repo.create_session("test")
        test_repo.add_message(sid, "user", "hello")
        test_repo.add_message(sid, "assistant", "", status="processing")
        test_repo.add_message(sid, "assistant", "done answer", status="complete")

        processing = test_repo.get_processing_messages(sid)
        assert len(processing) == 1
        assert processing[0]["status"] == "processing"

    def test_no_processing_messages_when_all_complete(self, test_repo):
        sid = test_repo.create_session("test")
        test_repo.add_message(sid, "user", "hello")
        test_repo.add_message(sid, "assistant", "answer", status="complete")

        processing = test_repo.get_processing_messages(sid)
        assert len(processing) == 0

    def test_get_messages_includes_status(self, test_repo):
        sid = test_repo.create_session("test")
        test_repo.add_message(sid, "user", "hello")
        test_repo.add_message(sid, "assistant", "", status="processing")

        msgs = test_repo.get_messages(sid)
        assert len(msgs) == 2
        assert msgs[0]["status"] == "complete"
        assert msgs[1]["status"] == "processing"


class TestDocumentStatus:
    def test_add_document_default_ready(self, test_repo):
        doc_id = test_repo.add_document("test.pdf", "test.pdf", "pdf")
        doc = test_repo.get_document(doc_id)
        assert doc["status"] == "ready"

    def test_add_document_processing(self, test_repo):
        doc_id = test_repo.add_document("test.pdf", "test.pdf", "pdf", status="processing")
        doc = test_repo.get_document(doc_id)
        assert doc["status"] == "processing"

    def test_update_document_status(self, test_repo):
        doc_id = test_repo.add_document("test.pdf", "test.pdf", "pdf", status="processing")
        test_repo.update_document_status(doc_id, "ready")
        doc = test_repo.get_document(doc_id)
        assert doc["status"] == "ready"

    def test_update_document_status_failed(self, test_repo):
        doc_id = test_repo.add_document("test.pdf", "test.pdf", "pdf", status="processing")
        test_repo.update_document_status(doc_id, "failed", error_message="parse error")
        doc = test_repo.get_document(doc_id)
        assert doc["status"] == "failed"
        assert doc["error_message"] == "parse error"

    def test_list_documents_includes_status(self, test_repo):
        test_repo.add_document("a.pdf", "a.pdf", "pdf", status="processing")
        test_repo.add_document("b.pdf", "b.pdf", "pdf", status="ready")
        docs = test_repo.list_documents()
        statuses = {d["status"] for d in docs}
        assert "processing" in statuses
        assert "ready" in statuses


class TestChatEndpointBackground:
    def test_chat_returns_409_when_processing(self):
        from app.main import app, kb
        original_add_message = kb.repo.add_message
        original_get_processing = kb.repo.get_processing_messages
        original_recent_history = kb.repo.recent_history
        original_create_session = kb.repo.create_session
        original_ensure_session = kb.repo.ensure_session
        original_touch_session = kb.repo.touch_session
        original_get_summary = kb.repo.get_summary
        original_get_messages = kb.repo.get_messages
        try:
            kb.repo.create_session = lambda title="": "test-sid"
            kb.repo.ensure_session = lambda sid: None
            kb.repo.recent_history = lambda sid, limit=6: []
            kb.repo.get_summary = lambda sid: None
            kb.repo.add_message = lambda *a, **kw: 1
            kb.repo.touch_session = lambda *a, **kw: None
            kb.repo.get_messages = lambda sid: []
            kb.repo.get_processing_messages = lambda sid: [{"id": 1, "status": "processing"}]

            client = TestClient(app)
            resp = client.post("/api/chat", json={"session_id": "test-sid", "message": "hello"})
            assert resp.status_code == 409
        finally:
            kb.repo.add_message = original_add_message
            kb.repo.get_processing_messages = original_get_processing
            kb.repo.recent_history = original_recent_history
            kb.repo.create_session = original_create_session
            kb.repo.ensure_session = original_ensure_session
            kb.repo.touch_session = original_touch_session
            kb.repo.get_summary = original_get_summary
            kb.repo.get_messages = original_get_messages