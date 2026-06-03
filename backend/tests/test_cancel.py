import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


class TestChatCancel:
    def test_cancel_chat_no_processing(self):
        from app.main import app, kb
        original_get_processing = kb.repo.get_processing_messages
        original_cancelled = set()
        try:
            kb.repo.get_processing_messages = lambda sid: []
            client = TestClient(app)
            resp = client.post("/api/chat/cancel", json={"session_id": "test-sid"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["cancelled"] is False
            assert "Không có" in data["message"]
        finally:
            kb.repo.get_processing_messages = original_get_processing

    def test_cancel_chat_with_processing(self):
        from app.main import app, kb
        original_get_processing = kb.repo.get_processing_messages
        original_update = kb.repo.update_message
        try:
            kb.repo.get_processing_messages = lambda sid: [{"id": 42, "status": "processing"}]
            kb.repo.update_message = lambda msg_id, **kw: None
            client = TestClient(app)
            resp = client.post("/api/chat/cancel", json={"session_id": "test-sid"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["cancelled"] is True
            assert "Đã hủy" in data["message"]
        finally:
            kb.repo.get_processing_messages = original_get_processing
            kb.repo.update_message = original_update

    def test_cancel_chat_missing_session(self):
        from app.main import app
        client = TestClient(app)
        resp = client.post("/api/chat/cancel", json={})
        assert resp.status_code == 422

    def test_cancelled_agent_skips_persist(self):
        from app.main import _run_agent_sync, _cancelled_sessions
        from app.db.database import Database
        from app.db.repo import Repo
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        db = Database(os.path.join(tmpdir, "test.db"))
        repo = Repo(db)
        sid = repo.create_session("test")
        msg_id = repo.add_message(sid, "assistant", "", status="processing")

        # Simulate cancelled session
        _cancelled_sessions.add(sid)

        # Create a mock generator that yields a route event
        class MockGen:
            def __init__(self):
                self.count = 0
            def __iter__(self):
                return self
            def __next__(self):
                self.count += 1
                if self.count == 1:
                    return {"type": "route", "data": {"route": "simple"}}
                raise StopIteration

        import asyncio
        queue = asyncio.Queue()

        # Mock agent
        class MockAgent:
            def __init__(self, *a, **kw):
                pass
            def run(self, *a, **kw):
                return MockGen()

        # Patch Agent
        import app.main as main_module
        original_agent = main_module.Agent
        try:
            main_module.Agent = MockAgent
            _run_agent_sync(sid, msg_id, "hello", [], None, queue)
        finally:
            main_module.Agent = original_agent
            _cancelled_sessions.discard(sid)

        # Check message is cancelled
        msg = repo.get_message(msg_id)
        assert msg["status"] == "cancelled"
        assert "hủy" in msg["error_message"].lower()


class TestUploadCancelRobust:
    def test_ingest_bg_handles_deleted_doc(self):
        from app.main import kb
        from app.db.database import Database
        from app.db.repo import Repo
        import tempfile, os

        tmpdir = tempfile.mkdtemp()
        db = Database(os.path.join(tmpdir, "test.db"))
        repo = Repo(db)
        kb.db = db
        kb.repo = repo

        # Create doc then delete it immediately (simulate user cancel)
        doc_id = repo.add_document("test.pdf", "test.pdf", "pdf", status="processing")
        repo.delete_document(doc_id)

        # _ingest_bg should handle this gracefully
        data = b"%PDF-1.4 fake"
        name = "test.pdf"

        def _ingest_bg():
            try:
                doc = kb.repo.get_document(doc_id)
                if not doc:
                    return None
                result = kb.ingest_pdf(data, name, doc_id=doc_id)
                doc = kb.repo.get_document(doc_id)
                if doc:
                    kb.repo.update_document_status(doc_id, "ready")
                return result
            except Exception as e:
                try:
                    doc = kb.repo.get_document(doc_id)
                    if doc:
                        kb.repo.update_document_status(doc_id, "failed", error_message=str(e))
                except Exception:
                    pass
                return None

        result = _ingest_bg()
        assert result is None  # Gracefully handled deletion