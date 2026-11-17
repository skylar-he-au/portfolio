"""Deterministic, network-free embeddings shared by ingest and retrieval tests."""

import hashlib

DIMENSIONS = 1536


def unit_vector(index: int = 0) -> list[float]:
    vector = [0.0] * DIMENSIONS
    vector[index] = 1.0
    return vector


class FakeEmbedder:
    def __init__(
        self,
        vectors: dict[str, list[float]] | None = None,
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self.vectors = vectors or {}
        self.fail_on_call = fail_on_call
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("fake embedding failure")
        result = []
        for text in texts:
            if text in self.vectors:
                result.append(list(self.vectors[text]))
            else:
                digest = hashlib.sha256(text.encode()).digest()
                result.append(
                    [(digest[index % len(digest)] - 127.5) / 128 for index in range(DIMENSIONS)]
                )
        return result
