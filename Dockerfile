FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd -m app && mkdir -p /app/artifacts && chown -R app:app /app/artifacts
USER app
ENV STUDYBUDDY_VECTOR_DB=/app/artifacts/studybuddy.sqlite3
EXPOSE 8000
CMD ["uvicorn","studybuddy.api:app","--host","0.0.0.0","--port","8000"]
