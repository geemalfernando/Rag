"""Thin wrapper around the Gemini SDK: embeddings and text generation with retries."""

import numpy as np
from google import genai
from google.genai import types

from rag.config import Settings

# Gemini returns 429/503 when a model is busy; those usually clear up within seconds.
_RETRY = types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=30.0, http_status_codes=[429, 500, 502, 503, 504])
_EMBED_BATCH = 100
EMBED_DIM = 768


class Gemini:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(
            api_key=settings.require_key(),
            http_options=types.HttpOptions(retry_options=_RETRY),
        )

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            result = self.client.models.embed_content(
                model=self.settings.embed_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
            )
            vectors.extend(e.values for e in result.embeddings)
        arr = np.asarray(vectors, dtype=np.float32).reshape(len(texts), EMBED_DIM)
        # Truncated gemini-embedding vectors aren't unit length, so normalise for cosine similarity.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.clip(norms, 1e-12, None)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]

    def generate(self, prompt: str, system: str | None = None) -> str:
        response = self.client.models.generate_content(
            model=self.settings.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=system),
        )
        return response.text or ""
