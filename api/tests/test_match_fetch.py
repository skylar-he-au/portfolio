"""Strict job-link parsing and bounded public API fetching without network access."""

import asyncio
import html
import json
import re
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.match.fetch import FetchedJob, FetchError, JobRef, fetch_job, parse_job_url

pytestmark = pytest.mark.anyio

GREENHOUSE = JobRef("greenhouse", "boards-api.greenhouse.io", "quantium", "4005542201")
LEVER = JobRef("lever", "api.lever.co", "mable", "40cc617b-b8f7-4734-952a-80f95235e159")
LEVER_EU = JobRef("lever", "api.eu.lever.co", "xtb", "a8512df6-0322-416c-aee2-72fb6bee12e9")
FIXTURES = Path(__file__).parent / "fixtures" / "match"
CAP = 1_048_576


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def fixture_body(filename: str) -> bytes:
    return (FIXTURES / filename).read_bytes()


@pytest.mark.parametrize(
    ("host", "ref"),
    [
        ("boards.greenhouse.io", GREENHOUSE),
        ("job-boards.greenhouse.io", GREENHOUSE),
        ("job-boards.anz.greenhouse.io", GREENHOUSE),
        ("jobs.lever.co", LEVER),
        ("jobs.eu.lever.co", LEVER_EU),
    ],
)
@pytest.mark.parametrize("suffix", ["", "?source=example", "/", "#details", "/?a=b#details"])
@pytest.mark.parametrize("surround", ["", " \n\t"])
def test_parse_allowed_urls(host: str, ref: JobRef, suffix: str, surround: str) -> None:
    path = (
        f"{ref.board}/jobs/{ref.job_id}"
        if ref.source == "greenhouse"
        else f"{ref.board}/{ref.job_id}"
    )
    assert parse_job_url(f"{surround}https://{host}/{path}{suffix}{surround}") == ref


@pytest.mark.parametrize("ref", [LEVER, LEVER_EU])
@pytest.mark.parametrize("suffix", ["/apply", "/apply/", "/apply/?source=test#apply"])
def test_parse_lever_apply(ref: JobRef, suffix: str) -> None:
    host = "jobs.eu.lever.co" if ref == LEVER_EU else "jobs.lever.co"
    assert parse_job_url(f"https://{host}/{ref.board}/{ref.job_id}{suffix}") == ref


def test_parse_preserves_board_case_and_normalizes_host_and_uuid() -> None:
    assert parse_job_url("https://BOARDS.GREENHOUSE.IO/Mixed_Case-1/jobs/001") == JobRef(
        "greenhouse", "boards-api.greenhouse.io", "Mixed_Case-1", "001"
    )
    assert parse_job_url(f"https://JOBS.LEVER.CO/Mixed_Case-1/{LEVER.job_id.upper()}") == replace(
        LEVER, board="Mixed_Case-1"
    )


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com/quantium/jobs/123",
        "https://job-boards.eu.greenhouse.io/quantium/jobs/123",
        "https://company.com/careers?gh_jid=123",
        "https://boards.greenhouse.io.evil.com/quantium/jobs/123",
        "https://evilboards.greenhouse.io/quantium/jobs/123",
        "https://boards.greenhouse.io./quantium/jobs/123",
        "https://boards.greenhouse.\u0131o/quantium/jobs/123",
        f"https://job\u017fs.lever.co/mable/{LEVER.job_id}",
        "text https://boards.greenhouse.io/quantium/jobs/123",
        "https://boards.greenhouse.io/quantium/jobs/123 text",
        "https://jobs.lever.co/a/one https://jobs.lever.co/a/two",
        "http://boards.greenhouse.io/quantium/jobs/123",
        "https://boards.greenhouse.io:443/quantium/jobs/123",
        "https://boards.greenhouse.io:/quantium/jobs/123",
        "https://user@boards.greenhouse.io/quantium/jobs/123",
        "https://boards.greenhouse.io/../jobs/123",
        "https://boards.greenhouse.io/a/../quantium/jobs/123",
        "https://boards.greenhouse.io/a%2Fb/jobs/123",
        "https://boards.greenhouse.io/quantium/jobs/%31",
        "https://boards.greenhouse.io/a\\b/jobs/123",
        "https://boards.greenhouse.io/a.b/jobs/123",
        "https://boards.greenhouse.io/a b/jobs/123",
        "https://boards.greenhouse.io/é/jobs/123",
        "https://boards.greenhouse.io/quantium/jobs/abc",
        "https://boards.greenhouse.io/quantium/jobs/\uff11\uff12\uff13",
        "https://boards.greenhouse.io/quantium/jobs/123/extra",
        "https://boards.greenhouse.io/quantium/jobs/123?source=a\nb",
        "https://boards.greenhouse.io/quantium/jobs/123#some\ttext",
        "https://jobs.lever.co/mable/not-a-uuid",
        f"https://jobs.lever.co/a.b/{LEVER.job_id}",
        f"https://jobs.lever.co/mable/{LEVER.job_id}/other",
        f"https://jobs.lever.co/mable/{LEVER.job_id[:-1]}g",
        "",
        " \n\t",
        "x" * 3000,
        "https://boards.greenhouse.io/quantium/jobs/123?q=" + "x" * 3000,
    ],
)
def test_parse_rejects_other_inputs(value: str) -> None:
    assert parse_job_url(value) is None


@pytest.mark.parametrize(
    ("ref", "url", "payload"),
    [
        (
            GREENHOUSE,
            "https://boards-api.greenhouse.io/v1/boards/quantium/jobs/4005542201",
            {"title": "Engineer", "content": "&lt;p&gt;Build things&lt;/p&gt;"},
        ),
        (
            LEVER,
            "https://api.lever.co/v0/postings/mable/40cc617b-b8f7-4734-952a-80f95235e159",
            {"text": "Engineer", "descriptionPlain": "Build things"},
        ),
        (
            LEVER_EU,
            "https://api.eu.lever.co/v0/postings/xtb/a8512df6-0322-416c-aee2-72fb6bee12e9",
            {"text": "Engineer", "descriptionPlain": "Build things"},
        ),
    ],
)
async def test_fetch_exact_api_url(ref: JobRef, url: str, payload: dict[str, str]) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.method == "GET"
        assert request.url == httpx.URL(url)
        assert request.headers["accept"] == "application/json"
        assert set(request.extensions["timeout"].values()) == {8.0}
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await fetch_job(ref, client)
    assert result == FetchedJob(ref.source, "Engineer", "Build things", False)
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [(302, "bad_response"), (404, "not_found"), (429, "rate_limited"), (500, "bad_response")],
)
async def test_status_errors_without_redirects_or_retries(status: int, code: str) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://example.com/redirect"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle), follow_redirects=True
    ) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(GREENHOUSE, client)
    assert exc.value.code == code
    assert len(requests) == 1


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("declared_length", [None, str(CAP + 1)])
async def test_size_limit_stops_stream_early(declared_length: str | None) -> None:
    stream = Chunks([b"x" * (CAP // 2), b"x" * (CAP // 2), b"x", b"unread"])
    headers = {} if declared_length is None else {"Content-Length": declared_length}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, headers=headers, stream=stream))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(GREENHOUSE, client)
    assert exc.value.code == "too_large"
    assert stream.consumed == (3 if declared_length is None else 0)
    assert stream.closed


async def test_body_at_size_limit_is_accepted() -> None:
    body = json.dumps({"content": "Short"}).encode()
    stream = Chunks([body, b" " * (CAP - len(body))])
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await fetch_job(GREENHOUSE, client)).text == "Short"
    assert stream.closed


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectError, "bad_response"),
        (OSError, "bad_response"),
    ],
)
@pytest.mark.parametrize("during_body", [False, True])
async def test_transport_errors(error: type[Exception], code: str, during_body: bool) -> None:
    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"{"
            raise error("failed")

    def handle(request: httpx.Request) -> httpx.Response:
        if during_body:
            return httpx.Response(200, stream=BrokenBody())
        raise error("failed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(GREENHOUSE, client)
    assert exc.value.code == code


@pytest.mark.parametrize("during_body", [False, True])
async def test_total_deadline_covers_request_and_body(
    monkeypatch: pytest.MonkeyPatch, during_body: bool
) -> None:
    original_timeout = asyncio.timeout
    deadlines: list[float] = []

    def short_timeout(delay: float) -> asyncio.Timeout:
        deadlines.append(delay)
        return original_timeout(0.01)

    class SlowBody(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"{"
            await asyncio.Event().wait()

    async def handle(request: httpx.Request) -> httpx.Response:
        if not during_body:
            await asyncio.Event().wait()
        return httpx.Response(200, stream=SlowBody())

    monkeypatch.setattr(asyncio, "timeout", short_timeout)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(GREENHOUSE, client)
    assert exc.value.code == "timeout"
    assert deadlines == [8.0]


@pytest.mark.parametrize("body", [b"not JSON", b"[]", b"null", b'"text"', b"\xff"])
async def test_bad_json(body: bytes) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(GREENHOUSE, client)
    assert exc.value.code == "bad_response"


@pytest.mark.parametrize(
    ("ref", "payload"),
    [
        (GREENHOUSE, {}),
        (GREENHOUSE, {"content": 123}),
        (GREENHOUSE, {"content": "&lt;p&gt; &lt;/p&gt;&lt;script&gt;hidden&lt;/script&gt;"}),
        (LEVER, {}),
        (LEVER, {"descriptionPlain": " \n", "lists": [None, 1, {"content": False}]}),
        (LEVER, {"descriptionPlain": [], "lists": "wrong", "additional": 123}),
    ],
)
async def test_empty_or_malformed_text(ref: JobRef, payload: dict[str, Any]) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(ref, client)
    assert exc.value.code == "bad_response"


@pytest.mark.parametrize(
    "ref",
    [
        replace(GREENHOUSE, host="evil.com"),
        replace(GREENHOUSE, host=LEVER.host),
        replace(LEVER, host=GREENHOUSE.host),
        replace(LEVER, host="api.lever.co:443"),
        replace(GREENHOUSE, board="a/b"),
        replace(LEVER, board="../secret"),
        replace(GREENHOUSE, job_id="\uff11\uff12\uff13"),
        replace(LEVER, job_id="not-a-uuid"),
        replace(GREENHOUSE, source="unknown"),  # type: ignore[arg-type]
    ],
)
async def test_invalid_ref_never_requests(ref: JobRef) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"content": "Must not reach here"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(FetchError) as exc:
            await fetch_job(ref, client)
    assert exc.value.code == "bad_response"
    assert requests == []


@pytest.mark.parametrize("ref", [GREENHOUSE, LEVER])
@pytest.mark.parametrize("length", [10, 20_000, 20_001])
async def test_text_truncation(ref: JobRef, length: int) -> None:
    field = "content" if ref.source == "greenhouse" else "descriptionPlain"
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={field: "é" * length}))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch_job(ref, client)
    assert result.text == "é" * min(length, 20_000)
    assert result.truncated is (length > 20_000)
    assert result.title is None


@pytest.mark.parametrize("title", [None, "", " \n", 123, []])
@pytest.mark.parametrize("ref", [GREENHOUSE, LEVER])
async def test_invalid_title_is_none(ref: JobRef, title: Any) -> None:
    payload = {"title": title, "text": title, "content": "Body", "descriptionPlain": "Body"}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await fetch_job(ref, client)).title is None


@pytest.mark.parametrize("additional", [None, "", " \n", 123])
async def test_lever_sections_and_html_fallback(additional: Any) -> None:
    payload = {
        "descriptionPlain": "Plain intro",
        "description": "<p>Ignore duplicate HTML</p>",
        "lists": [
            None,
            {"text": "Responsibilities", "content": "<ul><li>Build &amp; test</li></ul>"},
            {"text": 123, "content": "<p>Content only</p>"},
            {"text": "Heading only", "content": None},
        ],
        "additionalPlain": additional,
        "additional": "<p>HTML ending</p>",
    }
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch_job(LEVER, client)
    assert result.text == (
        "Plain intro\n\nResponsibilities\n- Build & test\n\nContent only"
        "\n\nHeading only\n\nHTML ending"
    )


async def test_lever_prefers_plain_additional() -> None:
    payload = {"descriptionPlain": "Intro", "additionalPlain": "Plain", "additional": "HTML"}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    async with httpx.AsyncClient(transport=transport) as client:
        assert (await fetch_job(LEVER, client)).text == "Intro\n\nPlain"


async def test_html_blocks_entities_whitespace_and_hidden_content() -> None:
    blocks = [
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
    ]
    markup = "".join(f"before<{tag}>inside</{tag}>after" for tag in blocks)
    markup += (
        "<script>secret <b>script</b></script><style>secret style</style>"
        "<p>A\t  B&nbsp; C &amp; D &#39;quoted&#39;</p>\n\n\n\n<p>End</p>"
    )
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"content": html.escape(markup)})
    )
    async with httpx.AsyncClient(transport=transport) as client:
        text = (await fetch_job(GREENHOUSE, client)).text
    assert "secret" not in text
    assert "A B C & D 'quoted'" in text
    assert "- inside" in text
    assert "beforeinside" not in text
    assert "insideafter" not in text
    assert "\n\n\n" not in text
    assert all(line == line.strip() for line in text.splitlines())
    assert re.search(r"<[^>]*>|&(?:#\d+|#x[0-9a-fA-F]+|\w+);", text) is None


@pytest.mark.parametrize(
    ("filename", "ref", "title_field", "phrases"),
    [
        (
            "greenhouse.json",
            GREENHOUSE,
            "title",
            [
                "Who is Quantium?",
                "designing and delivering production-grade generative AI",
                "- Technical ownership: Take end-to-end ownership",
            ],
        ),
        (
            "lever.json",
            LEVER,
            "text",
            [
                "About Mable",
                "Key Responsibilities",
                "- Work as part of a cross-functional squad to deliver high-quality features",
            ],
        ),
        (
            "lever-eu.json",
            LEVER_EU,
            "text",
            [
                "XTB is a leading European online broker",
                "Responsibilities:",
                "- Ongoing account management and support for your active client base",
            ],
        ),
    ],
)
async def test_real_fixtures(
    filename: str, ref: JobRef, title_field: str, phrases: list[str]
) -> None:
    body = fixture_body(filename)
    payload = json.loads(body)
    transport = httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    async with httpx.AsyncClient(transport=transport) as client:
        result = await fetch_job(ref, client)
    assert result.source == ref.source
    assert result.title == payload[title_field]
    assert not result.truncated
    for phrase in phrases:
        assert phrase in result.text
    assert re.search(r"<[^>]*>|&(?:#\d+|#x[0-9a-fA-F]+|\w+);", result.text) is None
