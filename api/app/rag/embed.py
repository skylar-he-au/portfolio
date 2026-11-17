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
            # Send raw text instead of tiktoken token ids. The default first
            # downloads tiktoken's encoding table, which made the first call in
            # a fresh container take about 8 seconds. Sections (largest is
            # under 500 tokens) and 500-character questions stay far below the
            # model's 8191-token limit, so no local splitting is needed.
            check_embedding_ctx_length=False,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts)
