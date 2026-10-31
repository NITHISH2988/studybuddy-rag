import pytest
from fastapi.testclient import TestClient

from studybuddy import api
from studybuddy.service import Settings, StudyBuddyService


@pytest.fixture
def client(tmp_path, monkeypatch, keyword_embedder):
    service = StudyBuddyService(
        Settings(
            vector_db_path=str(tmp_path / "vectors.sqlite3"), minimum_similarity=0.2
        ),
        embedder=keyword_embedder,
    )
    original_service = getattr(api.app.state, "service", None)
    api.app.state.service = service
    yield TestClient(api.app)
    api.app.state.service = original_service


def test_health_and_frontend_are_available(client):
    assert client.get("/health").json()["status"] == "ok"
    response = client.get("/")
    assert response.status_code == 200
    assert "StudyBuddy" in response.text


def test_document_question_and_retrieval_evaluation(client):
    added = client.post(
        "/documents",
        json={
            "title": "Solar guide",
            "text": "Solar panels convert sunlight into electricity.",
        },
    )
    assert added.status_code == 200
    chunk_id = added.json()["chunk_ids"][0]

    answer = client.post("/ask", json={"question": "How do sun panels make power?"})
    assert answer.status_code == 200
    assert answer.json()["citations"][0]["chunk_id"] == chunk_id
    assert "[S1]" in answer.json()["answer"]

    evaluation = client.post(
        "/evaluate",
        json={
            "k": 2,
            "queries": [
                {"query": "solar power panels", "relevant_chunk_ids": [chunk_id]}
            ],
        },
    )
    assert evaluation.status_code == 200
    assert evaluation.json()["recall@2"] == 1.0
    assert evaluation.json()["mrr@2"] == 1.0


def test_upload_accepts_text_file_and_rejects_unknown_extension(client):
    good = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"Solar panels use the sun to make energy.",
                "text/plain",
            )
        },
    )
    assert good.status_code == 200
    assert good.json()["chunks_added"] == 1
    bad = client.post("/upload", files={"file": ("notes.csv", b"a,b", "text/csv")})
    assert bad.status_code == 415


def test_request_validation_rejects_blank_and_invalid_k(client):
    assert (
        client.post("/documents", json={"title": "", "text": "x" * 20}).status_code
        == 422
    )
    assert client.post("/ask", json={"question": "?", "top_k": 0}).status_code == 422
