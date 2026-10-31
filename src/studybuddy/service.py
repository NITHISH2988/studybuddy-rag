"""Application wiring and source-checked question answering."""

import re
from dataclasses import dataclass

from .embeddings import create_embedder
from .evaluation import evaluate_retrieval
from .generation import create_generator
from .store import SQLiteVectorStore

ABSTENTION = (
    "I could not find evidence in the indexed documents to answer that question."
)


@dataclass
class Settings:
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_backend: str = "extractive"
    generator_model: str = "gemma3"
    ollama_url: str = "http://localhost:11434"
    vector_db_path: str = "artifacts/studybuddy.sqlite3"
    minimum_similarity: float = 0.2


class StudyBuddyService:
    def __init__(self, settings: Settings, embedder=None, generator=None):
        self.settings = settings
        self.embedder = embedder or create_embedder(
            settings.embedding_backend, settings.embedding_model, settings.ollama_url
        )
        self.generator = generator or create_generator(
            settings.generator_backend, settings.generator_model, settings.ollama_url
        )
        self.store = SQLiteVectorStore(settings.vector_db_path, self.embedder)

    @property
    def chunk_count(self) -> int:
        return self.store.count

    def add_document(self, title: str, text: str) -> dict:
        return self.store.add_document(title, text)

    def answer(self, question: str, top_k: int = 4) -> dict:
        sources = self.store.search(
            question,
            top_k=top_k,
            min_score=self.settings.minimum_similarity,
        )
        if not sources:
            return {
                "answer": ABSTENTION,
                "citations": [],
                "generator": self.generator.model_id,
            }
        answer = self.generator.answer(question, sources)
        valid_labels = {f"S{index}" for index in range(1, len(sources) + 1)}
        labels = re.findall(r"\[(S\d+)\]", answer)
        answer = re.sub(
            r"\[(S\d+)\]",
            lambda match: match.group(0) if match.group(1) in valid_labels else "",
            answer,
        ).strip()
        if not any(label in valid_labels for label in labels):
            answer = f"{answer}\n\nSources: " + ", ".join(
                f"[{label}]" for label in sorted(valid_labels, key=lambda x: int(x[1:]))
            )
        return {
            "answer": answer,
            "citations": sources,
            "generator": self.generator.model_id,
        }

    def evaluate(self, examples: list[dict], k: int) -> dict:
        return evaluate_retrieval(
            examples,
            lambda query, limit: self.store.search(
                query,
                top_k=limit,
                min_score=self.settings.minimum_similarity,
            ),
            k=k,
        )
