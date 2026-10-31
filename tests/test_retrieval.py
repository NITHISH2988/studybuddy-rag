import pytest

from studybuddy.core import chunk_text
from studybuddy.evaluation import evaluate_retrieval
from studybuddy.store import SQLiteVectorStore


def test_chunks_overlap_and_have_stable_offsets():
    chunks = chunk_text(
        "doc", "notes", "one two three four five six", size=4, overlap=1
    )
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six"
    assert chunks[0].chunk_id == "doc:0"
    assert chunks[1].chunk_id == "doc:1"


def test_semantic_retrieval_ranks_relevant_chunk_and_persists(
    tmp_path, keyword_embedder
):
    path = tmp_path / "vectors.sqlite3"
    store = SQLiteVectorStore(path, keyword_embedder)
    solar = store.add_document(
        "Solar energy", "Photovoltaic solar panels convert sunlight into electricity."
    )
    store.add_document("Cooking", "Boil pasta in a pot of water in the kitchen.")

    result = store.search("How do sun panels create power?", top_k=2, min_score=0.2)

    assert result[0]["document_id"] == solar["document_id"]
    assert result[0]["score"] > 0.9
    assert result[0]["chunk_id"] in solar["chunk_ids"]
    reopened = SQLiteVectorStore(path, keyword_embedder)
    assert (
        reopened.search("photovoltaic panels", top_k=1)[0]["document_id"]
        == solar["document_id"]
    )


def test_retrieval_evaluation_computes_recall_and_mrr(tmp_path, keyword_embedder):
    store = SQLiteVectorStore(tmp_path / "vectors.sqlite3", keyword_embedder)
    relevant = store.add_document(
        "Solar", "Solar panels use sunlight to generate power."
    )
    examples = [{"query": "sun panels", "relevant_chunk_ids": relevant["chunk_ids"]}]
    metrics = evaluate_retrieval(
        examples, lambda query, k: store.search(query, top_k=k), k=3
    )
    assert metrics == {"queries": 1, "recall@3": 1.0, "mrr@3": 1.0}


def test_recall_at_k_counts_all_labeled_relevant_chunks():
    examples = [{"query": "concept", "relevant_chunk_ids": ["first", "second"]}]
    search = lambda query, k: [{"chunk_id": "first"}]
    metrics = evaluate_retrieval(examples, search, k=1)
    assert metrics["recall@1"] == 0.5
    assert metrics["mrr@1"] == 1.0


def test_empty_queries_and_empty_text_are_rejected(tmp_path, keyword_embedder):
    with pytest.raises(ValueError, match="cannot be empty"):
        chunk_text("doc", "title", " ")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_retrieval([], lambda query, k: [], k=2)


def test_database_rejects_different_embedding_model(tmp_path, keyword_embedder):
    path = tmp_path / "vectors.sqlite3"
    SQLiteVectorStore(path, keyword_embedder)

    class OtherEmbedder:
        model_id = "incompatible-model"

        def embed(self, texts):
            return keyword_embedder.embed(texts)

    with pytest.raises(ValueError, match="different embedding model"):
        SQLiteVectorStore(path, OtherEmbedder())


def test_database_connections_close_after_operations(tmp_path, keyword_embedder):
    path = tmp_path / "vectors.sqlite3"
    store = SQLiteVectorStore(path, keyword_embedder)
    store.add_document("Solar", "Solar panels use sunlight to generate electricity.")
    store.search("solar power")
    path.rename(tmp_path / "renamed.sqlite3")
