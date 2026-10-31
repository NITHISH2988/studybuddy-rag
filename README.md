# StudyBuddy RAG

StudyBuddy indexes study notes, retrieves semantically similar passages, and answers questions from those passages with source citations. It is a local-first teaching project; retrieval helps expose evidence but does not eliminate unsupported or incorrect model output.

## Architecture

1. Text, Markdown, PDF, and DOCX ingestion extracts text.
2. Paragraph text is split into overlapping word chunks.
3. A configurable embedding backend creates normalized dense vectors: Sentence Transformers (default) or Ollama.
4. SQLite stores chunk metadata and vectors. Cosine similarity is calculated with NumPy for a modest personal corpus; no external vector database is required.
5. Search returns ranked chunks and scores. A similarity threshold abstains when no chunk is sufficiently close.
6. The default generator returns retrieved excerpts directly. Optional Ollama chat generation receives only those excerpts and is asked to answer with `[S#]` references.
7. The response returns citation records containing document ID, chunk ID, excerpt, and similarity score. Invalid citation labels from a generator are removed; source IDs are determined by retrieval, not by generated text.

The SQLite database records the embedding model ID and rejects reopening it with a different embedding model. Start a new database path if you intentionally change embedding models.

The default `sentence-transformers/all-MiniLM-L6-v2` embedding model is listed as Apache-2.0 on its [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Model weights are downloaded at runtime and are not included in this repository.

## Local setup

Python 3.10+ is supported. Install the project and test dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn studybuddy.api:app --reload
```

The first Sentence Transformers use downloads `sentence-transformers/all-MiniLM-L6-v2` to the local model cache. It runs locally after download. To use Ollama for both embeddings and generation, install and start Ollama separately, pull the configured models, then set:

```powershell
$env:STUDYBUDDY_EMBEDDING_BACKEND = "ollama"
$env:STUDYBUDDY_EMBEDDING_MODEL = "embeddinggemma"
$env:STUDYBUDDY_GENERATOR = "ollama"
$env:STUDYBUDDY_OLLAMA_CHAT_MODEL = "gemma3"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
```

The API uses Ollama's `/api/embed` and non-streaming `/api/chat` endpoints. Check the [Ollama embedding API](https://docs.ollama.com/api/embed) and [chat API](https://docs.ollama.com/api/chat). The default generation mode is `extractive` and needs no chat model.

## Use the API or frontend

Open `http://127.0.0.1:8000/` for the included small frontend, or `/docs` for OpenAPI.

- `POST /documents`: JSON `{ "title": "...", "text": "..." }`
- `POST /upload`: multipart TXT, Markdown, PDF, or DOCX file (10 MiB maximum)
- `POST /ask`: JSON `{ "question": "...", "top_k": 4 }`
- `POST /evaluate`: query set with known `relevant_chunk_ids`; returns Recall@k and MRR@k
- `GET /health`: process health and indexed chunk count, if the service has initialized

Ingestion returns `document_id`, `chunk_ids`, and `chunks_added`; keep those IDs for retrieval evaluation.

## Retrieval evaluation

Prepare labeled queries whose relevant chunk IDs are known, for example:

```json
{
  "k": 5,
  "queries": [
    {"query": "How do solar panels convert sunlight?", "relevant_chunk_ids": ["DOC_ID:0"]}
  ]
}
```

`recall@k` is the mean fraction of each query's labeled relevant chunks found in the top k. `mrr@k` measures how highly the first relevant chunk is ranked. The current tests use deterministic fake embeddings to verify ranking and metric calculations; those test vectors are not retrieval-quality results for a real embedding model. Build a representative human-labeled query set before drawing conclusions about an embedding model.

### Small real-encoder smoke check

The default `all-MiniLM-L6-v2` encoder was run against five short hand-labeled passages and five paraphrased queries. This smoke set returned Recall@3 **1.00** and MRR@3 **1.00**. A separate two-passage check returned cosine similarity 0.855 for the related solar passage and 0.006 for an unrelated cooking passage. These checks only confirm that model loading, vector creation, persistence, ranking, and the evaluator connect end to end; five simple queries are not a representative quality benchmark.

## Tests

```powershell
python -m pytest -q
```

Tests cover chunking, dense-vector ranking, SQLite persistence/model compatibility, retrieval metrics, abstention, citation/source mapping, text upload, API validation, and mocked Ollama embedding/chat calls. They do not download model weights or require an Ollama service.

CI on Python 3.11 runs Ruff lint and formatting checks, this fixture-only test suite, and a pip-audit scan of runtime dependencies; it does not exercise a live model download or Ollama service.

## Configuration and limitations

Environment variables: `STUDYBUDDY_EMBEDDING_BACKEND`, `STUDYBUDDY_EMBEDDING_MODEL`, `STUDYBUDDY_GENERATOR`, `STUDYBUDDY_OLLAMA_CHAT_MODEL`, `OLLAMA_BASE_URL`, `STUDYBUDDY_VECTOR_DB`, and `STUDYBUDDY_MIN_SIMILARITY`. Defaults store the SQLite index at `artifacts/studybuddy.sqlite3`.

The service has no authentication or user isolation and must stay on a trusted local network. Similarity thresholds need evaluation on the intended documents. PDF extraction can fail on scanned/image-only documents. Generated answers can still contain unsupported claims; returned citations establish which passages were retrieved, not that every sentence is entailed. Do not upload confidential notes to a remote model.

## Docker

Build with `docker build -t studybuddy-rag .` and run with a persistent local artifact volume, for example `docker run --rm -p 8000:8000 -v "${PWD}/artifacts:/app/artifacts" studybuddy-rag`. For Ollama in another container or on the host, set `OLLAMA_BASE_URL` to a reachable address; `localhost` inside a container refers to that container. Docker Engine was unavailable during this quality pass, so the image build and container run instructions remain unverified.

## License

MIT for the code. Uploaded document rights remain with their respective owners; only ingest files you are permitted to use.
