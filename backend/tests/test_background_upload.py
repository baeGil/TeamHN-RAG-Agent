import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


@pytest.fixture
def mock_kb():
    with patch("app.main.kb") as mock:
        mock.repo.list_documents.return_value = []
        mock.repo.create_session.return_value = "test-session"
        mock.repo.ensure_session.return_value = None
        mock.repo.recent_history.return_value = []
        mock.repo.get_summary.return_value = None
        mock.repo.add_message.return_value = 1
        mock.repo.touch_session.return_value = None
        mock.repo.get_processing_messages.return_value = []
        mock.stats.return_value = {"documents": 0, "chunks": 0, "vector_ready": False, "reranker": False}
        yield mock


class TestBackgroundUpload:
    def test_upload_returns_processing_status(self, client, mock_kb):
        mock_kb.repo.add_document.return_value = 42
        with patch("app.main.kb", mock_kb):
            pdf_data = b"%PDF-1.4 test content"
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("test.pdf", pdf_data, "application/pdf")},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "processing"
            assert data["document_id"] == 42
            mock_kb.repo.add_document.assert_called_once_with("test.pdf", "test.pdf", "pdf", status="processing")

    def test_upload_rejects_oversized_file(self, client, mock_kb):
        from app.config import Settings
        settings = Settings()
        big_data = b"%PDF-1.4" + b"\x00" * (settings.max_upload_size + 1)
        with patch("app.main.kb", mock_kb):
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("big.pdf", big_data, "application/pdf")},
            )
            assert resp.status_code == 413

    def test_upload_rejects_non_pdf_magic_bytes(self, client, mock_kb):
        with patch("app.main.kb", mock_kb):
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("test.pdf", b"Not a PDF", "application/pdf")},
            )
            assert resp.status_code == 400

    def test_upload_rejects_non_pdf_extension(self, client, mock_kb):
        pdf_data = b"%PDF-1.4 test content"
        with patch("app.main.kb", mock_kb):
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", pdf_data, "application/pdf")},
            )
            assert resp.status_code == 400

    def test_documents_include_status_field(self, client, mock_kb):
        mock_kb.repo.list_documents.return_value = [
            {"id": 1, "title": "test.pdf", "source": "test.pdf", "source_type": "pdf",
             "n_chunks": 10, "status": "ready", "error_message": None, "created_at": "2024-01-01"},
            {"id": 2, "title": "processing.pdf", "source": "processing.pdf", "source_type": "pdf",
             "n_chunks": 0, "status": "processing", "error_message": None, "created_at": "2024-01-02"},
        ]
        with patch("app.main.kb", mock_kb):
            resp = client.get("/api/documents")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["status"] == "ready"
            assert data[1]["status"] == "processing"