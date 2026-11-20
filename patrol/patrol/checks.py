"""The patrol's pure parts: the X-Patrol header, the host filter and the verdict.

No browser and no model here, so the unit tests cover them without either.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from urllib.parse import urldefrag, urlsplit

DEFAULT_PORTS = {"http": 80, "https": 443}


def patrol_header(secret: str, now: float) -> str:
    """`<unix seconds>.<hex HMAC-SHA256(secret, "<unix seconds>")>`, as the API checks it
    (`api/app/api/session.py`, valid for 300 seconds either side)."""
    timestamp = str(int(now))
    signature = hmac.digest(secret.encode(), timestamp.encode("ascii"), "sha256").hex()
    return f"{timestamp}.{signature}"


def _origin(url: str) -> tuple[str, str, int] | None:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return None
    return scheme, parts.hostname.lower(), port or DEFAULT_PORTS[scheme]


def base_origin(base_url: str) -> str:
    """`scheme://host[:port]` of the site under patrol; raises ValueError for anything else."""
    origin = _origin(base_url)
    if origin is None:
        raise ValueError(f"--base-url must be an http or https URL, not {base_url!r}")
    scheme, host, port = origin
    return f"{scheme}://{host}" if port == DEFAULT_PORTS[scheme] else f"{scheme}://{host}:{port}"


def same_origin(url: str, base_url: str) -> bool:
    """Whether the browser may load `url`: only the site under patrol (same scheme, host and
    port). Everything else, Cloudflare's Turnstile script included, is aborted."""
    origin = _origin(url)
    return origin is not None and origin == _origin(base_url)


def domain_pattern(base_url: str) -> str:
    """browser-use's `allowed_domains` entry for the site (`browser_use/domains.py`):
    `https://example.com` is that scheme and host on the default port, `localhost:4321` that
    host and port."""
    scheme, host, port = _origin(base_origin(base_url))  # type: ignore[misc]
    return f"{scheme}://{host}" if port == DEFAULT_PORTS[scheme] else f"{host}:{port}"


@dataclass
class Snapshot:
    """What one look at a page found (`SNAPSHOT_JS` in `run.py` builds it)."""

    url: str
    home: bool = False
    answer_chars: int = 0
    answer_done: bool = False
    answer_links: list[str] = field(default_factory=list)
    match_chars: int = 0
    match_done: bool = False
    errors: list[str] = field(default_factory=list)


@dataclass
class Observation:
    """Everything the patrol saw during one run, gathered by code, never from the agent."""

    base_url: str
    session_status: int | None = None
    session_error: str | None = None
    home_status: int | None = None
    home: bool = False
    answered: bool = False
    links: set[str] = field(default_factory=set)
    navigations: list[str] = field(default_factory=list)
    matched: bool = False
    api_status: dict[str, list[int]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    blocked_hosts: set[str] = field(default_factory=set)

    def add(self, snap: Snapshot) -> None:
        if not same_origin(snap.url, self.base_url):
            return
        self.home |= snap.home
        if snap.answer_chars > 0 and snap.answer_done:
            self.answered = True
            self.links.update(
                link for link in snap.answer_links if same_origin(link, self.base_url)
            )
        if snap.match_chars > 0 and snap.match_done:
            self.matched = True
        for error in snap.errors:
            if error and error not in self.errors:
                self.errors.append(error)

    def add_status(self, path: str, status: int) -> None:
        self.api_status.setdefault(path, []).append(status)

    def citation_opened(self) -> bool:
        """A main-frame navigation reached one of the answer's links (fragment ignored) that is
        not the home page itself."""
        home = urldefrag(self.base_url.rstrip("/") + "/").url
        targets = {urldefrag(link).url for link in self.links} - {home}
        return any(urldefrag(url).url in targets for url in self.navigations)


@dataclass
class Step:
    name: str
    ok: bool
    detail: str


def _api(obs: Observation, path: str) -> str:
    statuses = obs.api_status.get(path)
    return f"{path} returned {', '.join(map(str, statuses))}" if statuses else f"no {path} call"


def _errors(obs: Observation) -> str:
    return f"; page said: {' | '.join(obs.errors)}" if obs.errors else ""


def verdict(obs: Observation) -> list[Step]:
    """One line per patrol step, in the order a recruiter goes; green only when all pass."""
    session_ok = obs.session_status == 200 and obs.session_error is None
    return [
        Step(
            "session",
            session_ok,
            "patrol session issued"
            if session_ok
            else f"/api/session: {obs.session_error or f'HTTP {obs.session_status}'}",
        ),
        Step(
            "home",
            obs.home_status == 200 and obs.home,
            "home page shows Ask and Match"
            if obs.home_status == 200 and obs.home
            else f"home page HTTP {obs.home_status}, Ask and Match boxes "
            + ("shown" if obs.home else "missing"),
        ),
        Step(
            "ask",
            obs.answered,
            "an example question was answered to the end"
            if obs.answered
            else f"no finished answer ({_api(obs, '/api/ask')}){_errors(obs)}",
        ),
        Step(
            "citation",
            bool(obs.links),
            f"the answer links {len(obs.links)} project section(s)"
            if obs.links
            else "the answer has no link to a project page",
        ),
        Step(
            "open-citation",
            obs.citation_opened(),
            "a cited project page was opened"
            if obs.citation_opened()
            else "no cited project page was opened",
        ),
        Step(
            "match",
            obs.matched,
            "the job ad match result was shown"
            if obs.matched
            else f"no finished match result ({_api(obs, '/api/match')}){_errors(obs)}",
        ),
    ]
