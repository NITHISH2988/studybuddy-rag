import httpx
import numpy as np

from studybuddy.embeddings import OllamaEmbedder
from studybuddy.generation import OllamaGenerator


def test_ollama_embed_uses_official_endpoint_and_normalizes():
    def handler(request):
        assert request.url.path == "/api/embed"
        assert request.read()
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]]})

    embedder = OllamaEmbedder(transport=httpx.MockTransport(handler))
    vectors = embedder.embed(["text"])
    assert np.allclose(vectors, [[0.6, 0.8]])


def test_ollama_generator_requests_non_streaming_chat():
    def handler(request):
        assert request.url.path == "/api/chat"
        body = request.read().decode("utf-8")
        assert '"stream":false' in body
        return httpx.Response(200, json={"message": {"content": "Panel output [S1]."}})

    generator = OllamaGenerator(transport=httpx.MockTransport(handler))
    answer = generator.answer("What is this?", [{"excerpt": "Solar panel source"}])
    assert answer == "Panel output [S1]."
