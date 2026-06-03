import pytest
from unittest.mock import MagicMock, patch
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
        mock.stats.return_value = {"documents": 0, "chunks": 0, "vector_ready": False, "reranker": False}
        yield mock


class TestConfigMaxUploadSize:
    def test_max_upload_size_default(self):
        from app.config import Settings
        s = Settings()
        assert s.max_upload_size == 5 * 1024 * 1024

    def test_max_upload_size_custom(self):
        from app.config import Settings
        with patch.dict("os.environ", {"MAX_UPLOAD_SIZE": "10485760"}):
            s = Settings()
            assert s.max_upload_size == 10 * 1024 * 1024

    def test_config_endpoint_returns_max_upload_size(self, client, mock_kb):
        with patch("app.main.kb", mock_kb):
            resp = client.get("/api/config")
            assert resp.status_code == 200
            data = resp.json()
            assert "max_upload_size" in data
            assert data["max_upload_size"] > 0


class TestUploadValidation:
    def test_upload_rejects_oversized_file(self, client, mock_kb):
        big_data = b"%PDF-1.4 fake" + b"\x00" * (6 * 1024 * 1024)
        with patch("app.main.kb", mock_kb):
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("large.pdf", big_data, "application/pdf")},
            )
            assert resp.status_code == 413
            assert "quá lớn" in resp.json()["detail"].lower() or "too large" in resp.text.lower()

    def test_upload_rejects_non_pdf_magic_bytes(self, client, mock_kb):
        fake_data = b"Not a PDF file content" + b"\x00" * 100
        with patch("app.main.kb", mock_kb):
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("test.pdf", fake_data, "application/pdf")},
            )
            assert resp.status_code == 400
            assert "pdf" in resp.json()["detail"].lower()

    def test_upload_rejects_non_pdf_extension(self, client, mock_kb):
        pdf_data = b"%PDF-1.4 fake content for test"
        with patch("app.main.kb", mock_kb):
            resp = client.post(
                "/api/documents/upload",
                files={"file": ("test.txt", pdf_data, "application/pdf")},
            )
            assert resp.status_code == 400

    def test_config_endpoint_includes_max_upload_size(self):
        from app.config import Settings
        s = Settings()
        assert hasattr(s, "max_upload_size")
        assert s.max_upload_size == 5 * 1024 * 1024