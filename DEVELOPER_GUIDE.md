# Developer Guide: StudyBuddy RAG

StudyBuddy is a local document question-answering application. It demonstrates dense retrieval and source handling, while explicitly not claiming that citations guarantee entailment or eliminate hallucinations.

## Architecture and data flow

```text
text/TXT/MD/PDF/DOCX → extract_text (uploads) → chunk_text
                                           → Embedder.embed → normalize
                                           → SQLiteVectorStore.add_document
question → embed → SQLiteVectorStore.search (cosine, top-k, threshold)
          → StudyBuddyService.answer → extractive or Ollama generator
          → citations tied to retrieved rows + filtered [S#] labels
evaluation queries → same search → Recall@k + MRR@k
FastAPI routes + static/index.html expose the workflow
```

`ingestion.extract_text()` supports text, Markdown, PDF, and DOCX. `core.chunk_text()` splits on whitespace into overlapping word windows; IDs are derived from document ID and window position. `embeddings.Embedder` is a protocol implemented by `SentenceTransformerEmbedder` and `OllamaEmbedder`. Both output normalized float32 vectors. `SQLiteVectorStore` stores chunk metadata and JSON vectors, and stores the embedding model ID in metadata so it refuses a database opened with a different model name. Search embeds the query, checks dimensions, computes dot products (cosine similarity for normalized vectors), sorts descending, and applies `min_score`.

`StudyBuddyService.answer()` retrieves first. It abstains if nothing passes the threshold. Otherwise it asks the configured generator for an answer. The extractive generator returns passages directly; Ollama receives a system instruction to answer only from excerpts and cite `[S#]`. Service code strips labels outside the returned source set and guarantees a source list when none are cited. Citation records come from retrieval itself, not from the LLM. This verifies source association, not that every generated statement follows from a source.

`evaluation.evaluate_retrieval()` scores labeled relevant chunk IDs using Recall@k and MRR@k. These metrics evaluate ranked retrieval only, not answer faithfulness. The API wires settings from environment variables, serves HTML at `/`, and has ingestion, question, evaluation, and health routes. The included frontend uses those endpoints and displays the source excerpts.

## Directory map and entry points

- `src/studybuddy/api.py`: Pydantic request schemas, environment settings, lazy service initialization, routes, upload limit, and frontend serving; run `uvicorn studybuddy.api:app --reload`.
- `src/studybuddy/ingestion.py`: format-specific extraction.
- `src/studybuddy/core.py`: `Chunk` and `chunk_text`.
- `src/studybuddy/embeddings.py`: embedder protocol, local Sentence Transformers/Ollama backends, normalization, and factory.
- `src/studybuddy/store.py`: SQLite schema, persistence, document addition, and vector ranking.
- `src/studybuddy/generation.py`: extractive and Ollama chat generators.
- `src/studybuddy/service.py`: orchestration, abstention, citation label filtering, and evaluation wiring.
- `src/studybuddy/evaluation.py`: retrieval metrics.
- `src/studybuddy/static/index.html`: single-page browser UI.
- `tests/`: chunking/ingestion, retrieval relevance, persistence, citations, API behavior, and Ollama calls using deterministic fixtures/mocks.
- `.github/workflows/ci.yml`: Python 3.11, lint/format, fixture-only tests, runtime audit.
- `Dockerfile`: API container; persistent SQLite storage should be mounted as a volume.

## Run locally and complete a walkthrough

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
uvicorn studybuddy.api:app --reload
```

The first default embedding call downloads `all-MiniLM-L6-v2`; it needs internet access once and local disk for the model cache. Open `http://127.0.0.1:8000/`, add a title and at least 20 characters of notes, then ask a question about those notes. The page reports indexed chunk count, displays the answer and retrieved citations. For an explicit API walk-through, POST text such as `{"title":"Solar notes","text":"Solar panels convert sunlight into electricity using photovoltaic cells. The cells generate direct current, which an inverter converts to alternating current."}` to `/documents`, then ask `{"question":"How does a solar panel produce usable electricity?","top_k":4}` at `/ask`.

Returned IDs, excerpts, scores, answer text, and citation positions depend on the encoder and indexed text; there is no fixed expected generated sentence. In default extractive mode, the answer is the retrieved text labeled `[S1]`, `[S2]`, etc. Ollama mode can phrase an answer and may still make unsupported statements; inspect citations against the text.

For retrieval evaluation, save the `chunk_ids` returned by ingestion and submit labeled queries to `/evaluate`. Example request:

```json
{"k": 3, "queries": [{"query":"What converts direct current?","relevant_chunk_ids":["YOUR_DOCUMENT_ID:0"]}]}
```

Replace the placeholder with a real ID returned by your run. The API returns `recall@3` and `mrr@3`. These values are meaningful only relative to a human-labeled representative query set.

To use Ollama for both embeddings and generation, install/start Ollama and pull the configured models, then set `STUDYBUDDY_EMBEDDING_BACKEND=ollama`, `STUDYBUDDY_EMBEDDING_MODEL=embeddinggemma`, `STUDYBUDDY_GENERATOR=ollama`, `STUDYBUDDY_OLLAMA_CHAT_MODEL=gemma3`, and `OLLAMA_BASE_URL=http://localhost:11434`. Changing embedding model requires a new SQLite database path because vectors must share one embedding space.

## Concepts to be ready to explain

1. **Embeddings:** dense vectors encode semantic features; model choice, language/domain, and query/document mismatch affect quality.
2. **Normalization and cosine similarity:** unit-length vectors make dot product equal cosine similarity; higher scores are treated as closer, not as calibrated relevance probabilities.
3. **Chunking:** fixed word windows with overlap preserve some local continuity but may split semantic boundaries and count words rather than tokens.
4. **Top-k and threshold:** top-k bounds context size; the similarity threshold offers a simple abstention rule that must be tuned on representative queries.
5. **Persistence/model compatibility:** SQLite stores vectors and model identifier; identifier matching prevents obvious incompatible model reuse but does not pin an immutable model revision.
6. **Recall@k:** fraction of known relevant chunks retrieved in the first k results.
7. **MRR@k:** reciprocal rank of the first relevant result; it rewards finding a relevant chunk early.
8. **Grounded generation:** retrieval constrains the supplied context; a prompt is not a formal guarantee of factual entailment.
9. **Citation mapping:** citation records are generated from search results; filtering unknown `[S#]` labels prevents references to nonretrieved sources, but does not verify claim-to-source entailment.
10. **RAG evaluation layers:** retrieval metrics and answer-faithfulness checks are separate; current automated evaluation covers retrieval only.

## Tests, CI, and extension points

Run `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .`. Fake embedding vectors give deterministic ranking tests; `httpx` mock transports test Ollama request/response handling. CI intentionally avoids model downloads and a live Ollama server. The small README smoke check is not a representative benchmark.

Sensible extensions include model revision and vector dimension metadata, a human-labeled evaluation set, hybrid lexical+dense retrieval if measurements justify it, and request isolation/authentication before shared-network deployment. Current exact cosine search reads all stored vectors into memory for each query and is suited to a modest personal corpus. Avoid adding a vector database until corpus size and latency measurements justify it.

## Current limits and troubleshooting

- No authentication or tenant isolation: keep the service on a trusted local machine/network.
- The upload cap limits raw request bytes but is not a comprehensive defense against pathological compressed documents or extraction cost.
- Scanned PDFs need OCR, which is not implemented.
- Exact search is linear in number of stored chunks and vector dimension.
- Citation presence proves a retrieved source was referenced, not that every sentence is supported.
- If model loading fails, check the model cache/network, Ollama availability, and the configured model names.
- If opening an existing DB raises a model mismatch, retain its original embedding model or point `STUDYBUDDY_VECTOR_DB` to a new file.
- If answers abstain, check that text was indexed, inspect retrieved similarity scores, and evaluate an appropriate threshold rather than lowering it blindly.

