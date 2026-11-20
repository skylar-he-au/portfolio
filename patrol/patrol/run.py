"""One patrol run: Playwright owns the browser and watches it; browser-use's agent drives it.

1. `/api/session` with X-Patrol through Playwright's `request` (the signature never reaches the
   browser or the agent); the returned JSON goes into sessionStorage `session` by an init script.
2. A local headless Chromium (persistent context = the browser's default context, so a tab the
   agent opens over CDP lands in it too) with a route that aborts every other host.
3. browser-use connects over `cdp_url`, with `allowed_domains` as a second guard.
4. While the agent runs, code samples the pages and records navigations and API statuses; the
   verdict (`checks.verdict`) uses only that, never what the agent says.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Page, Route, async_playwright
from playwright.async_api import Error as PlaywrightError

from patrol.checks import (
    Observation,
    Snapshot,
    Step,
    domain_pattern,
    patrol_header,
    same_origin,
    verdict,
)
from patrol.task import task_text

MODEL = "gemini-3.5-flash-lite"
MAX_TOKENS = 4096  # per model request, thinking included
MAX_STEPS = 25
AGENT_TIMEOUT_S = 180  # browser-use's agent loop (model requests and cells)
HARD_TIMEOUT_S = 240  # the loop plus browser-use's own start and delivery turn, as a backstop
SAMPLE_EVERY_S = 0.5
API_PATHS = ("/api/ask", "/api/match")

SNAPSHOT_JS = """() => {
  const ask = document.querySelector('#ask');
  const match = document.querySelector('#match');
  const chars = (nodes) => nodes.map((node) => node.innerText.trim()).join('').length;
  const askAnswers = ask ? [...ask.querySelectorAll('.ask-answer')] : [];
  const matchAnswers = match ? [...match.querySelectorAll('.ask-answer')] : [];
  return {
    url: location.href,
    home: !!(ask && match && ask.querySelector('textarea') && match.querySelector('textarea')),
    answer_chars: chars(askAnswers),
    answer_done: !!(ask && ask.querySelector('[aria-label="This answer is wrong"]')),
    answer_links: askAnswers.flatMap((a) => [...a.querySelectorAll('a[href]')].map((l) => l.href)),
    match_chars: chars(matchAnswers),
    match_done: !!(match && match.querySelector('[aria-label="This guide is wrong"]')),
    errors: [...document.querySelectorAll('.ask-error')].map((e) => e.innerText.trim()).slice(0, 5),
  };
}"""


@dataclass
class AgentRun:
    status: str = "not run"
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    cost_usd: float | None = None
    error: str | None = None


@dataclass
class Report:
    obs: Observation
    agent: AgentRun = field(default_factory=AgentRun)
    seconds: float = 0.0

    @property
    def steps(self) -> list[Step]:
        return verdict(self.obs)

    @property
    def passed(self) -> bool:
        return all(step.ok for step in self.steps)


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


async def _get_session(pw, base: str, secret: str, obs: Observation) -> str | None:
    """The patrol session JSON, as the API returned it; None (and obs says why) when refused."""
    request = await pw.request.new_context(base_url=base)
    try:
        response = await request.post(
            "/api/session",
            headers={"X-Patrol": patrol_header(secret, time.time())},
            data={"turnstile": ""},
            timeout=30_000,
        )
        obs.session_status = response.status
        if response.status != 200:
            with contextlib.suppress(PlaywrightError, ValueError):
                code = (await response.json()).get("code")
                obs.session_error = f"HTTP {response.status} {code}" if code else None
            return None
        text = await response.text()
        try:
            body = json.loads(text)
        except ValueError:
            body = None
        if not (
            isinstance(body, dict)
            and isinstance(body.get("token"), str)
            and isinstance(body.get("expires_at"), str)
        ):
            obs.session_error = "the response is not {token, expires_at}"
            return None
        return text
    except PlaywrightError as e:
        obs.session_error = f"request failed ({type(e).__name__})"
        return None
    finally:
        await request.dispose()


def _init_script(base: str, session_json: str) -> str:
    # Runs before the site's own scripts on every document; only the patrolled origin gets it.
    origin, value = json.dumps(base), json.dumps(session_json)
    return (
        f"(() => {{ if (location.origin !== {origin}) return;"
        f" try {{ sessionStorage.setItem('session', {value}); }} catch (e) {{}} }})();"
    )


async def _cdp_port(user_data_dir: Path) -> int:
    marker = user_data_dir / "DevToolsActivePort"
    for _ in range(100):
        with contextlib.suppress(OSError, ValueError, IndexError):
            return int(marker.read_text().splitlines()[0])
        await asyncio.sleep(0.1)
    raise RuntimeError("Chromium did not report its remote debugging port")


def _watch(context: BrowserContext, base: str, obs: Observation) -> None:
    def on_response(response) -> None:
        url = response.url
        path = urlsplit(url).path
        if same_origin(url, base) and path in API_PATHS:
            obs.add_status(path, response.status)

    def on_page(page: Page) -> None:
        page.on(
            "framenavigated",
            lambda frame: (
                obs.navigations.append(frame.url) if frame == frame.page.main_frame else None
            ),
        )

    context.on("response", on_response)
    context.on("page", on_page)
    for page in context.pages:
        on_page(page)


async def _sample(context: BrowserContext, base: str, obs: Observation) -> None:
    for page in list(context.pages):
        if page.is_closed() or not same_origin(page.url, base):
            continue
        try:
            data = await asyncio.wait_for(page.evaluate(SNAPSHOT_JS), 2)
        except (PlaywrightError, TimeoutError):
            continue  # mid-navigation; the next sample sees the new page
        obs.add(Snapshot(**data))


async def _sampler(context: BrowserContext, base: str, obs: Observation, stop: asyncio.Event):
    while not stop.is_set():
        await _sample(context, base, obs)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), SAMPLE_EVERY_S)


async def _run_agent(base: str, port: int, gemini_key: str, run: AgentRun) -> None:
    from browser_use import Agent, Browser
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    # ChatGoogle takes no model settings in 1.0.6 (llm/__init__.py: "pass a Pydantic AI Model built
    # with settings=..."), so the same Gemini model is built directly to carry max_tokens.
    llm = GoogleModel(
        MODEL, provider=GoogleProvider(api_key=gemini_key), settings={"max_tokens": MAX_TOKENS}
    )
    browser = Browser(
        cdp_url=f"http://127.0.0.1:{port}",
        allowed_domains=[domain_pattern(base)],
        keep_alive=True,  # Playwright owns the browser and looks at it after the run
    )
    # ultrafast: browser-use's accessibility-tree helpers (bu.state/click/type, browser_use/ax.py),
    # which click and type with real input events; Flash-Lite's own page-JS clicks were missed.
    agent = Agent(
        task_text(base),
        llm=llm,
        browser=browser,
        judge=False,
        reasoning="low",
        mode="ultrafast",
    )
    try:
        result = await asyncio.wait_for(
            agent.run(max_steps=MAX_STEPS, timeout=AGENT_TIMEOUT_S), HARD_TIMEOUT_S
        )
        run.status, run.steps = result.status, result.steps
        usage = dict(result.usage)
        run.input_tokens, run.output_tokens = usage.get("input", 0), usage.get("output", 0)
        run.requests, run.cost_usd = usage.get("requests", 0), result.cost_usd
        if result.status != "completed":
            run.error = result.status
    finally:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(browser.stop(), 30)


async def run_patrol(base: str, secret: str, gemini_key: str) -> Report:
    started = time.monotonic()
    obs = Observation(base_url=base)
    report = Report(obs)
    async with async_playwright() as pw:
        session_json = await _get_session(pw, base, secret, obs)
        if session_json is None:
            report.seconds = time.monotonic() - started
            return report

        user_data_dir = Path(tempfile.mkdtemp(prefix="portfolio-patrol-"))
        context = await pw.chromium.launch_persistent_context(
            str(user_data_dir),
            headless=True,
            args=["--remote-debugging-port=0", "--remote-debugging-address=127.0.0.1"],
        )
        try:

            async def route(route: Route) -> None:
                if same_origin(route.request.url, base):
                    await route.continue_()
                else:
                    obs.blocked_hosts.add(urlsplit(route.request.url).hostname or "?")
                    await route.abort()

            await context.route("**/*", route)
            await context.add_init_script(_init_script(base, session_json))
            _watch(context, base, obs)

            page = context.pages[0] if context.pages else await context.new_page()
            response = await page.goto(base + "/", wait_until="load", timeout=30_000)
            obs.home_status = response.status if response else None
            with contextlib.suppress(PlaywrightError):
                await page.wait_for_selector("#match textarea", timeout=15_000)
            await _sample(context, base, obs)
            if not (obs.home_status == 200 and obs.home):
                return report

            port = await _cdp_port(user_data_dir)
            stop = asyncio.Event()
            sampler = asyncio.create_task(_sampler(context, base, obs, stop))
            try:
                await _run_agent(base, port, gemini_key, report.agent)
            except TimeoutError:
                report.agent.status, report.agent.error = "timeout", "hard timeout"
            except Exception as e:  # noqa: BLE001 - a red patrol with the reason, not a traceback
                report.agent.status = "error"
                report.agent.error = redact(
                    f"{type(e).__name__}: {e}"[:300], [secret, gemini_key, session_json]
                )
            finally:
                stop.set()
                await sampler
            await _sample(context, base, obs)
        finally:
            report.seconds = time.monotonic() - started
            with contextlib.suppress(PlaywrightError):
                await context.close()
            shutil.rmtree(user_data_dir, ignore_errors=True)
    return report


def format_report(report: Report) -> str:
    lines = [f"patrol {report.obs.base_url}"]
    steps = report.steps
    # The run stops after a failed session or home page: what comes later was never tried.
    stopped = next((i for i, s in enumerate(steps[:2]) if not s.ok), None)
    for i, s in enumerate(steps):
        if stopped is not None and i > stopped:
            lines.append(f"  --    {s.name:<14} not reached")
        else:
            lines.append(f"  {'ok  ' if s.ok else 'FAIL'}  {s.name:<14} {s.detail}")
    a = report.agent
    cost = f"US${a.cost_usd:.4f}" if a.cost_usd is not None else "cost unknown"
    lines.append(
        f"agent: {a.status}, {a.steps} steps, {a.requests} model requests, "
        f"{a.input_tokens} input / {a.output_tokens} output tokens, {cost}"
        + (f" ({a.error})" if a.error else "")
    )
    if report.obs.blocked_hosts:
        lines.append(f"blocked hosts: {', '.join(sorted(report.obs.blocked_hosts))}")
    failed = next((s for s in report.steps if not s.ok), None)
    lines.append(
        f"result: GREEN in {report.seconds:.0f} s"
        if failed is None
        else f"result: RED at step '{failed.name}' in {report.seconds:.0f} s"
    )
    return "\n".join(lines)
