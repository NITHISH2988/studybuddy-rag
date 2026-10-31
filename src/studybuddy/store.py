"""Persistent, small-corpus cosine search backed by SQLite and normalized vectors."""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .core import chunk_text
from .embeddings import Embedder, normalize


class SQLiteVectorStore:
    def __init__(self, path: Path, embedder: Embedder) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS chunks "
                "(chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, title TEXT NOT NULL, "
                "text TEXT NOT NULL, vector TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key='embedding_model'"
            ).fetchone()
            if existing and existing[0] != embedder.model_id:
                raise ValueError(
                    "Vector database was created with a different embedding model; "
                    "use the original model or a new database path"
                )
            connection.execute(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES('embedding_model', ?)",
                (embedder.model_id,),
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @property
    def count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def add_document(
        self, title: str, text: str, *, size: int = 180, overlap: int = 35
    ) -> dict:
        doc_id = str(uuid.uuid4())
        chunks = chunk_text(doc_id, title, text, size=size, overlap=overlap)
        vectors = normalize(self.embedder.embed([chunk.text for chunk in chunks]))
        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedder returned a different number of vectors than chunks"
            )
        with self._connection() as connection:
            connection.executemany(
                "INSERT INTO chunks(chunk_id, doc_id, title, text, vector) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.text,
                        json.dumps(vector.tolist()),
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
            )
        return {
            "document_id": doc_id,
            "chunks_added": len(chunks),
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
        }

    def search(
        self, query: str, *, top_k: int = 4, min_score: float = 0.2
    ) -> list[dict]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT chunk_id, doc_id, title, text, vector FROM chunks"
            ).fetchall()
        if not rows:
            return []
        query_vector = normalize(self.embedder.embed([query]))[0]
        matrix = np.asarray([json.loads(row[4]) for row in rows], dtype=np.float32)
        if matrix.shape[1] != query_vector.shape[0]:
            raise ValueError("Query embedding dimension differs from stored vectors")
        scores = matrix @ query_vector
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "chunk_id": rows[index][0],
                "document_id": rows[index][1],
                "title": rows[index][2],
                "excerpt": rows[index][3],
                "score": float(scores[index]),
            }
            for index in ranked
            if float(scores[index]) >= min_score
        ]
