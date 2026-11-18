"""Recognize supported job links and fetch bounded plain text from public vendor APIs."""

import asyncio
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal

import httpx

Source = Literal["greenhouse", "lever"]
FetchErrorCode = Literal["not_found", "rate_limited", "timeout", "too_large", "bad_response"]

_GREENHOUSE_HOST = "boards-api.greenhouse.io"
_LEVER_HOSTS = ("api.lever.co", "api.eu.lever.co")
_BOARD = r"[A-Za-z0-9_-]+"
_GREENHOUSE_ID = r"[0-9]+"
_LEVER_ID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_SUFFIX = r"(?:\?[^\s#]*)?(?:#[^\s]*)?"
_GREENHOUSE_URL = re.compile(
    r"https://(?ai:boards\.greenhouse\.io|job-boards\.greenhouse\.io|"
    r"job-boards\.anz\.greenhouse\.io)/"
    rf"(?P<board>{_BOARD})/jobs/(?P<job_id>{_GREENHOUSE_ID})/?{_SUFFIX}"
)
_LEVER_URL = re.compile(
    r"https://(?P<host>(?ai:jobs\.lever\.co|jobs\.eu\.lever\.co))/"
    rf"(?P<board>{_BOARD})/(?P<job_id>{_LEVER_ID})(?:/apply)?/?{_SUFFIX}"
)
_MAX_BODY_BYTES = 1_048_576
_MAX_TEXT_CHARS = 20_000
_TIMEOUT = 8.0
_BLOCK_TAGS = frozenset(
    {
        "p",
        "div",
        "br",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "tr",
        "table",
        "section",
        "blockquote",
    }
)


@dataclass(frozen=True)
class JobRef:
    source: Source
    host: str
    board: str
    job_id: str


@dataclass(frozen=True)
class FetchedJob:
    source: Source
    title: str | None
    text: str
    truncated: bool


class FetchError(Exception):
    def __init__(self, code: FetchErrorCode) -> None:
        self.code: FetchErrorCode = code
        super().__init__(code)


def parse_job_url(text: str) -> JobRef | None:
    if len(text) > 2048:
        return None
    text = text.strip()
    if match := _GREENHOUSE_URL.fullmatch(text):
        return JobRef("greenhouse", _GREENHOUSE_HOST, match["board"], match["job_id"])
    if match := _LEVER_URL.fullmatch(text):
        host = "api.eu.lever.co" if match["host"].lower() == "jobs.eu.lever.co" else "api.lever.co"
        return JobRef("lever", host, match["board"], match["job_id"].lower())
    return None


def _matches(pattern: str, value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _request_url(ref: JobRef) -> str:
    if not _matches(_BOARD, ref.board):
        raise FetchError("bad_response")
    if ref.source == "greenhouse":
        if ref.host != _GREENHOUSE_HOST or not _matches(_GREENHOUSE_ID, ref.job_id):
            raise FetchError("bad_response")
        return f"https://{_GREENHOUSE_HOST}/v1/boards/{ref.board}/jobs/{ref.job_id}"
    if ref.source == "lever":
        if ref.host not in _LEVER_HOSTS or not _matches(_LEVER_ID, ref.job_id):
            raise FetchError("bad_response")
        return f"https://{ref.host}/v0/postings/{ref.board}/{ref.job_id}"
    raise FetchError("bad_response")


def _clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden += 1
        if self.hidden:
            return
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag == "li":
            self.parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self.hidden:
            self.hidden -= 1
            return
        if not self.hidden and tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _html_text(markup: str) -> str:
    parser = _TextParser()
    parser.feed(markup)
    parser.close()
    return _clean_text("".join(parser.parts))


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _job_text(source: Source, payload: dict[str, object]) -> tuple[str | None, str]:
    if source == "greenhouse":
        title = _string(payload.get("title")) or None
        # Greenhouse escapes the markup itself, in addition to entities inside it.
        text = _html_text(html.unescape(_string(payload.get("content"))))
        return title, text

    title = _string(payload.get("text")) or None
    pieces = [_string(payload.get("descriptionPlain"))]
    sections = payload.get("lists")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            heading = _string(section.get("text"))
            content = _html_text(_string(section.get("content")))
            pieces.append("\n".join(piece for piece in (heading, content) if piece))
    additional = _string(payload.get("additionalPlain"))
    pieces.append(additional or _html_text(_string(payload.get("additional"))))
    return title, _clean_text("\n\n".join(piece for piece in pieces if piece))


async def fetch_job(ref: JobRef, client: httpx.AsyncClient) -> FetchedJob:
    url = _request_url(ref)
    body = bytearray()
    try:
        async with asyncio.timeout(_TIMEOUT):
            async with client.stream(
                "GET",
                url,
                follow_redirects=False,
                headers={"Accept": "application/json"},
                timeout=_TIMEOUT,
            ) as response:
                if response.status_code == 404:
                    raise FetchError("not_found")
                if response.status_code == 429:
                    raise FetchError("rate_limited")
                if response.status_code != 200:
                    raise FetchError("bad_response")
                try:
                    length = int(response.headers.get("Content-Length", "0"))
                except ValueError:
                    length = 0
                if length > _MAX_BODY_BYTES:
                    raise FetchError("too_large")
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > _MAX_BODY_BYTES:
                        raise FetchError("too_large")
                    body.extend(chunk)
    except (httpx.TimeoutException, TimeoutError):
        raise FetchError("timeout") from None
    except (httpx.RequestError, OSError):
        raise FetchError("bad_response") from None

    try:
        payload = json.loads(body)
    except (ValueError, RecursionError):
        raise FetchError("bad_response") from None
    if not isinstance(payload, dict):
        raise FetchError("bad_response")
    title, text = _job_text(ref.source, payload)
    if not text:
        raise FetchError("bad_response")
    return FetchedJob(ref.source, title, text[:_MAX_TEXT_CHARS], len(text) > _MAX_TEXT_CHARS)
