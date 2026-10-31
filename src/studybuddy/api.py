"""FastAPI interface and a small local browser UI for StudyBuddy RAG."""

import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .ingestion import extract_text
from .service import Settings, StudyBuddyService

logger = logging.getLogger(__name__)
app = FastAPI(title="StudyBuddy RAG", version="0.2.0")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=20, max_length=500_000)

    @field_validator("title", "text")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must contain non-whitespace characters")
        return value


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)
    top_k: int = Field(default=4, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        return value


class EvaluationQuery(BaseModel):
    query: str = Field(min_length=3, max_length=2_000)
    relevant_chunk_ids: list[str] = Field(min_length=1)


class EvaluationRequest(BaseModel):
    queries: list[EvaluationQuery] = Field(min_length=1, max_length=500)
    k: int = Field(default=5, ge=1, le=20)


def _settings() -> Settings:
    return Settings(
        embedding_backend=os.getenv(
            "STUDYBUDDY_EMBEDDING_BACKEND", "sentence-transformers"
        ),
        embedding_model=os.getenv(
            "STUDYBUDDY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        generator_backend=os.getenv("STUDYBUDDY_GENERATOR", "extractive"),
        generator_model=os.getenv("STUDYBUDDY_OLLAMA_CHAT_MODEL", "gemma3"),
        ollama_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        vector_db_path=os.getenv(
            "STUDYBUDDY_VECTOR_DB", "artifacts/studybuddy.sqlite3"
        ),
        minimum_similarity=float(os.getenv("STUDYBUDDY_MIN_SIMILARITY", "0.2")),
    )


def get_service() -> StudyBuddyService:
    if not hasattr(app.state, "service") or app.state.service is None:
        app.state.service = StudyBuddyService(_settings())
    return app.state.service


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    service = getattr(app.state, "service", None)
    return {"status": "ok", "chunks": service.chunk_count if service else None}


@app.post("/documents")
def add_document(document: DocumentRequest) -> dict:
    try:
        return get_service().add_document(document.title, document.text)
    except Exception as error:
        logger.exception("Document indexing failed")
        raise HTTPException(
            status_code=503, detail="Document could not be indexed"
        ) from error


@app.post("/upload")
async def upload_document(file: Annotated[UploadFile, File()]) -> dict:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 10 MiB limit")
    try:
        text = extract_text(file.filename or "", content)
        return get_service().add_document(file.filename or "Untitled", text)
    except ValueError as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    except Exception as error:
        logger.exception("Document upload indexing failed")
        raise HTTPException(
            status_code=503, detail="Document could not be indexed"
        ) from error


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    try:
        return get_service().answer(request.question, request.top_k)
    except Exception as error:
        logger.exception("Question answering failed")
        raise HTTPException(
            status_code=503, detail="Retrieval or generation is unavailable"
        ) from error


@app.post("/evaluate")
def evaluate(request: EvaluationRequest) -> dict:
    examples = [query.model_dump() for query in request.queries]
    try:
        return get_service().evaluate(examples, request.k)
    except Exception as error:
        logger.exception("Retrieval evaluation failed")
        raise HTTPException(
            status_code=503, detail="Retrieval evaluation failed"
        ) from error
