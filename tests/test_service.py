from studybuddy.service import Settings, StudyBuddyService


class InvalidCitationGenerator:
    model_id = "test-generator"

    def answer(self, question, sources):
        return "Solar panels make electricity [S8]."


def test_answer_citations_reference_returned_sources(tmp_path, keyword_embedder):
    service = StudyBuddyService(
        Settings(
            vector_db_path=str(tmp_path / "vectors.sqlite3"), minimum_similarity=0.2
        ),
        embedder=keyword_embedder,
        generator=InvalidCitationGenerator(),
    )
    service.add_document("Solar", "Photovoltaic panels turn sunlight into electricity.")

    result = service.answer("How do solar panels make power?")

    assert result["citations"]
    assert "[S8]" not in result["answer"]
    assert "[S1]" in result["answer"]
    for citation in result["citations"]:
        assert citation["chunk_id"].startswith(citation["document_id"] + ":")


def test_answer_abstains_when_no_source_meets_threshold(tmp_path, keyword_embedder):
    service = StudyBuddyService(
        Settings(
            vector_db_path=str(tmp_path / "vectors.sqlite3"), minimum_similarity=0.8
        ),
        embedder=keyword_embedder,
    )
    service.add_document("Solar", "Photovoltaic panels turn sunlight into electricity.")

    answer = service.answer("How do I cook pasta?")

    assert answer["citations"] == []
    assert "could not find evidence" in answer["answer"]
