"""Embedding backends for local Sentence Transformers or local Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx
import numpy as np


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


@dataclass
class SentenceTransformerEmbedder:
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class OllamaEmbedder:
    model_id: str = "embeddinggemma"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    transport: httpx.BaseTransport | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        with httpx.Client(
            timeout=self.timeout_seconds, transport=self.transport
        ) as client:
            response = client.post(
                f"{self.base_url.rstrip('/')}/api/embed",
                json={"model": self.model_id, "input": texts, "truncate": False},
            )
            response.raise_for_status()
            vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("Ollama returned an invalid embedding matrix")
        return normalize(vectors)


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding model returned a zero-length vector")
    return vectors / norms


def create_embedder(kind: str, model_id: str, ollama_url: str) -> Embedder:
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_id)
    if kind == "ollama":
        return OllamaEmbedder(model_id=model_id, base_url=ollama_url)
    raise ValueError("embedding backend must be 'sentence-transformers' or 'ollama'")
