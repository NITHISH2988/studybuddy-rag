import numpy as np
import pytest


class KeywordEmbedder:
    """Deterministic semantic fixture; never represents a production embedding model."""

    model_id = "test-keyword-vectors-v1"

    def embed(self, texts):
        vectors = []
        for text in texts:
            value = text.casefold()
            if any(term in value for term in ("solar", "sun", "photovoltaic", "panel")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(term in value for term in ("cooking", "pasta", "boil", "kitchen")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def keyword_embedder():
    return KeywordEmbedder()
