import pytest

from rag.config import Settings


@pytest.fixture
def settings():
    return Settings.from_env()


@pytest.fixture
def gemini_settings(settings):
    if not settings.api_key:
        pytest.skip("GEMINI_API_KEY not set")
    return settings
