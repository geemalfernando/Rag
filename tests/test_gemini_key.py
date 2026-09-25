import numpy as np
import pytest

from rag.gemini import EMBED_DIM, Gemini

pytestmark = pytest.mark.gemini


def test_key_can_talk_to_gemini(gemini_settings):
    reply = Gemini(gemini_settings).generate("Reply with just the word: pong")
    assert "pong" in reply.lower()


def test_embeddings_rank_related_text_higher(gemini_settings):
    gem = Gemini(gemini_settings)
    docs = gem.embed_documents(["The cat sat on the warm windowsill.", "Quarterly revenue grew by 12 percent."])
    assert docs.shape == (2, EMBED_DIM)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-5)

    query = gem.embed_query("Where was the kitty sitting?")
    assert docs[0] @ query > docs[1] @ query
