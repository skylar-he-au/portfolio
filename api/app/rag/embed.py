"""Embedding adapter; importing config first disables LangSmith tracing."""

from app.config import AZURE_OPENAI_API_VERSION, Settings

# isort: split
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import SecretStr

from app.guard.ledger import current_scope
from app.guard.usage import metered, raise_if_budget_exhausted


def estimate_tokens(texts: list[str]) -> int:
    """A deliberately high token estimate (about 4 characters make a token)."""
    return (sum(len(text) for text in texts) + 2) // 3


def _present(secret: SecretStr | None) -> SecretStr | None:
    return secret if secret is not None and secret.get_secret_value() else None


class AzureEmbedder:
    """text-embedding-3-small on an Azure OpenAI deployment.

    Every connection setting is passed explicitly: given nothing, `AzureOpenAIEmbeddings`
    falls back to environment variables (`AZURE_OPENAI_API_KEY`, then `OPENAI_API_KEY`),
    and given an Entra token it silently drops the key.
    """

    def __init__(self, settings: Settings) -> None:
        key = _present(settings.azure_openai_api_key)
        token = _present(settings.azure_openai_ad_token)
        missing = [
            name
            for name, value in (
                ("AZURE_OPENAI_ENDPOINT", settings.azure_openai_endpoint),
                ("AZURE_OPENAI_DEPLOYMENT", settings.azure_openai_deployment),
                ("AZURE_OPENAI_API_KEY or AZURE_OPENAI_AD_TOKEN", key or token),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing embedding settings: {', '.join(missing)}")
        self._model = settings.embed_model
        self._client = AzureOpenAIEmbeddings(
            model=settings.embed_model,
            dimensions=settings.embed_dims,
            azure_endpoint=settings.azure_openai_endpoint,
            azure_deployment=settings.azure_openai_deployment,
            api_version=AZURE_OPENAI_API_VERSION,
            api_key=key,
            azure_ad_token=None if key else token,
            # Not read from OPENAI_API_BASE / OPENAI_ORG_ID either.
            base_url=None,
            organization=None,
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
        if current_scope.get() is None:
            return await self._client.aembed_documents(texts)
        try:
            async with metered(
                "openai", self._model, input_tokens=estimate_tokens(texts), embedding=True
            ):
                return await self._client.aembed_documents(texts)
        except Exception as error:
            await raise_if_budget_exhausted("openai", error)
            raise
