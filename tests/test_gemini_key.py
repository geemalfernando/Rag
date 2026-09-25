import pytest
from google import genai


@pytest.mark.gemini
def test_key_can_talk_to_gemini(gemini_settings):
    client = genai.Client(api_key=gemini_settings.api_key)
    reply = client.models.generate_content(model=gemini_settings.chat_model, contents="Reply with just the word: pong")
    assert "pong" in reply.text.lower()
