"""Embedding adapter; importing config first disables LangSmith tracing."""

from app.config import Settings

# isort: split
from langchain_openai import OpenAIEmbeddings


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key or not settings.openai_api_key.get_secret_value():
            raise RuntimeError("OPENAI_API_KEY is required to embed content")
        self._client = OpenAIEmbeddings(
            model=settings.embed_model,
            dimensions=settings.embed_dims,
            api_key=settings.openai_api_key,
            timeout=30,
            max_retries=2,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)
