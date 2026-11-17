"""Types shared by ingest, retrieval, the Q&A graph and the API."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Section:
    page: str  # site path: "/projects/roleradar", "/about", "/privacy"
    section_id: str  # github-slugger rules, same as the `##` heading id on the page
    heading: str
    text: str  # the section's Markdown body, without the `##` line
    score: float = 0.0  # retrieval score (RRF); 0 when ingesting

    @property
    def url(self) -> str:
        return f"{self.page}#{self.section_id}"


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class Retriever(Protocol):
    """Retrieval as the graph uses it: db and embedder already bound (3b2 binds retrieve())."""

    async def __call__(
        self, query: str, *, pages: Sequence[str] | None = None
    ) -> list[Section]: ...


@dataclass(frozen=True)
class RecalledTurn:
    turn_id: str
    question: str  # the visitor's original question
    answer: str  # the final answer text
    score: float  # memory ranking score


class ModelsUnavailable(Exception):
    """Both providers failed for one node; the API sends `error` `unavailable`."""
