import { Fragment, useEffect, useId, useLayoutEffect, useRef, useState } from 'react';
import type { KeyboardEvent, ReactNode, SubmitEvent } from 'react';
import { answerSegments, citationHref, safeSiteUrl } from '../lib/answer';
import type { AnswerSegment, Citation, InlineSegment, QuoteFragments } from '../lib/answer';
import { fragmentDirective, sectionElements, sectionHeadingIn, textBlocks, usableQuotes } from '../lib/quote-text';
import { ASK_MAX_CHARS, BAD_LINK_MESSAGE, detectInput } from '../lib/detect';
import type { MatchItem, MatchNotes, MatchSection, PageTitles } from '../lib/match';
import { useAsk } from '../hooks/useAsk';
import type { Turn } from '../hooks/useAsk';
import { useMatch } from '../hooks/useMatch';
import HowItAnswered from './HowItAnswered';

// Styles live in src/styles/assistant.css; colors only come from tokens.css.

export interface AssistantProps {
  /** "home": the home page's first screen (two columns from 960px). "panel": the project-page side panel (9p). */
  variant?: 'home' | 'panel';
  examples: string[];
  /** Project page path -> front matter title, so sources can name their page. */
  pageTitles?: PageTitles;
  /** Astro named slot: the home page's kicker, headline and intro. */
  intro?: ReactNode;
  /** Astro named slot: the example answer shown on the home page before anyone asks. */
  example?: ReactNode;
}

type Mode = 'ask' | 'match' | 'bad-link';

const REQUIREMENT_MAX_CHARS = 120;
const OFFLINE_MESSAGE = 'The assistant is offline right now — please try again later.';

function codePoints(text: string): number {
  return [...text].length;
}

function pagePath(url: string): string {
  return url.split('#')[0]!;
}

/** "RoleRadar · How RoleRadar matches jobs"; pages without a known title show the heading only. */
function sourceLabel(url: string, heading: string, pageTitles: PageTitles): string {
  const page = pagePath(url);
  const title = Object.prototype.hasOwnProperty.call(pageTitles, page) ? pageTitles[page] : undefined;
  return title ? `${title} · ${heading}` : heading;
}

/** "RoleRadar: Architecture" for the job-match cards. */
function sectionLine(section: MatchSection, pageTitles: PageTitles): string {
  const page = pagePath(section.url);
  const title = Object.prototype.hasOwnProperty.call(pageTitles, page) ? pageTitles[page] : undefined;
  return title ? `${title}: ${section.title}` : section.title;
}

/** Host and path of a pasted job link, shown in small text above its result. */
function linkLabel(link: string): string | null {
  try {
    const url = new URL(link);
    return `${url.host}${url.pathname}`;
  } catch {
    return null;
  }
}

function truncate(text: string, max: number): string {
  const points = [...text];
  return points.length > max ? `${points.slice(0, max - 1).join('')}…` : text;
}

/**
 * Display numbers for one answer's sources: 1..N in the order they first appear in its text,
 * then any safe source the text never cites. The backend's numbers are retrieval ranks
 * (kept in "How it answered"); readers expect a single source to be "1", not "6".
 */
function sourceNumbers(text: string, citations: Citation[]): Map<number, number> {
  const safe = new Set(citations.filter(({ url }) => safeSiteUrl(url) !== null).map(({ n }) => n));
  const numbers = new Map<number, number>();
  for (const match of text.matchAll(/\[(\d+)\]/g)) {
    const n = Number(match[1]);
    if (safe.has(n) && !numbers.has(n)) numbers.set(n, numbers.size + 1);
  }
  for (const { n } of citations) {
    if (safe.has(n) && !numbers.has(n)) numbers.set(n, numbers.size + 1);
  }
  return numbers;
}

const NO_FRAGMENTS: QuoteFragments = new Map();
// Pages a source points to, parsed once, to check that a quote appears only once there.
const sourcePages = new Map<string, Promise<Document | null>>();

function sourcePage(path: string): Promise<Document | null> {
  const page = (value: string) => value.replace(/\/+$/, '') || '/';
  if (page(path) === page(window.location.pathname)) return Promise.resolve(document);
  let promise = sourcePages.get(page(path));
  if (!promise) {
    promise = Promise.resolve()
      .then(() => fetch(path, { credentials: 'same-origin', headers: { Accept: 'text/html' } }))
      .then((response) => (response.ok && (response.headers.get('content-type') ?? '').includes('text/html')
        ? response.text() : null))
      .then((html) => (html === null ? null : new DOMParser().parseFromString(html, 'text/html')))
      .catch(() => null);
    sourcePages.set(page(path), promise);
  }
  return promise;
}

/** The Text Fragment for each source whose quotes appear only once on its page (SPEC "出处标到句子"). */
async function quoteFragments(citations: Citation[]): Promise<QuoteFragments> {
  const result = new Map<number, string>();
  for (const citation of citations) {
    const quotes = usableQuotes(citation.quotes);
    const href = safeSiteUrl(citation.url);
    if (quotes.length === 0 || !href) continue;
    const [path, hash = ''] = href.split('#');
    const doc = await sourcePage(path!);
    const heading = doc ? sectionHeadingIn(doc, hash) : null;
    if (!doc || !heading) continue;
    const directive = fragmentDirective(textBlocks([doc.body]), textBlocks(sectionElements(heading)), quotes);
    if (directive) result.set(citation.n, directive);
  }
  return result;
}

function useQuoteFragments(citations: Citation[], ready: boolean): QuoteFragments {
  const [fragments, setFragments] = useState<QuoteFragments>(NO_FRAGMENTS);
  const withQuotes = ready ? citations.filter(({ quotes }) => usableQuotes(quotes).length > 0) : [];
  const key = JSON.stringify(withQuotes.map(({ n, url, quotes }) => [n, url, quotes]));
  const latest = useRef(withQuotes);
  latest.current = withQuotes;
  useEffect(() => {
    if (latest.current.length === 0) {
      setFragments(NO_FRAGMENTS);
      return undefined;
    }
    let live = true;
    void quoteFragments(latest.current).then((result) => {
      if (live) setFragments(result);
    });
    return () => {
      live = false;
    };
  }, [key]);
  return fragments;
}

/** The quotes a source link carries for highlight.ts, which marks them on the page. */
function quotesAttribute(quotes: string[]): string | undefined {
  return quotes.length > 0 ? JSON.stringify(quotes) : undefined;
}

/** A source box is named like its Sources line ("Source 1: RoleRadar · In 30 seconds"), so a
 * screen reader can tell two pages' same-named sections apart (10i). */
function renderInline(segment: InlineSegment, key: number, numbers: Map<number, number>, pageTitles: PageTitles) {
  if (segment.type !== 'link') return segment.text;
  const rank = Number(segment.text.replace(/[[\]]/g, ''));
  const n = numbers.get(rank) ?? rank;
  return (
    <a key={key} className="ask-cite" href={segment.href} data-quotes={quotesAttribute(segment.quotes)} title={segment.title} aria-label={`Source ${n}: ${sourceLabel(segment.href, segment.title, pageTitles)}`}>
      {n}
    </a>
  );
}

// Punctuation that belongs to the sentence a source box ends: kept on the box's line.
const AFTER_CITE = /^[.,;:!?)\]…。，、；：！？）」』”’]+/u;

/**
 * Inline segments with each run of source boxes and the punctuation right after it in one
 * nowrap span: a box is an atomic inline, so the browser would otherwise put a line break
 * between it and the full stop (10v).
 */
function renderInlines(segments: InlineSegment[], numbers: Map<number, number>, pageTitles: PageTitles): ReactNode[] {
  const out: ReactNode[] = [];
  let index = 0;
  while (index < segments.length) {
    const segment = segments[index]!;
    if (segment.type !== 'link') {
      out.push(segment.text);
      index += 1;
      continue;
    }
    const start = index;
    const links: ReactNode[] = [];
    while (segments[index]?.type === 'link') {
      links.push(renderInline(segments[index]!, index, numbers, pageTitles));
      index += 1;
    }
    const next = segments[index];
    const punctuation = next?.type === 'text' ? AFTER_CITE.exec(next.text)?.[0] : undefined;
    if (!punctuation) {
      out.push(...links);
      continue;
    }
    out.push(<span key={`run-${start}`} className="ask-cite-run">{links}{punctuation}</span>);
    const rest = (next as { text: string }).text.slice(punctuation.length);
    if (rest) out.push(rest);
    index += 1;
  }
  return out;
}

/** Consecutive inline segments as one array, so a source box and the full stop after it
 * are seen together; marked (unsupported) spans stay as they are. */
function groupSegments(segments: AnswerSegment[]): (InlineSegment[] | { type: 'unsupported'; children: InlineSegment[] })[] {
  const groups: (InlineSegment[] | { type: 'unsupported'; children: InlineSegment[] })[] = [];
  for (const segment of segments) {
    if (segment.type === 'unsupported') groups.push(segment);
    else {
      const last = groups[groups.length - 1];
      if (Array.isArray(last)) last.push(segment);
      else groups.push([segment]);
    }
  }
  return groups;
}

function Steps({ steps }: { steps: { name: string; state: 'active' | 'done' }[] }) {
  if (steps.length === 0) return null;
  return (
    <ol className="ask-steps" aria-label="Progress">
      {steps.map((step) => (
        <li key={step.name} className="ask-step" data-state={step.state}>
          {step.state === 'done' && <span aria-hidden="true">✓ </span>}
          {step.name}
        </li>
      ))}
    </ol>
  );
}

function Shimmer() {
  return (
    <div className="ask-shimmer" aria-hidden="true">
      <span />
      <span />
    </div>
  );
}

function AiLabel({ children }: { children: string }) {
  return (
    <p className="ask-answer-label">
      <span className="ask-dot" aria-hidden="true" />
      {children}
    </p>
  );
}

function Sources({ citations, numbers, pageTitles, fragments }: {
  citations: Citation[];
  numbers: Map<number, number>;
  pageTitles: PageTitles;
  fragments: QuoteFragments;
}) {
  const safe = citations
    .filter(({ n, url }) => safeSiteUrl(url) !== null && numbers.has(n))
    .sort((a, b) => numbers.get(a.n)! - numbers.get(b.n)!);
  if (safe.length === 0) return null;
  return (
    <div className="ask-sources">
      <p className="ask-sources-title">Sources</p>
      <ol>
        {safe.map((citation) => (
          <li key={citation.n}>
            <span className="ask-source-n" aria-hidden="true">{numbers.get(citation.n)}</span>
            <a href={citationHref(citation, fragments) ?? citation.url} data-quotes={quotesAttribute(usableQuotes(citation.quotes))}>
              {sourceLabel(citation.url, citation.heading, pageTitles)}
            </a>
          </li>
        ))}
      </ol>
    </div>
  );
}

function AnswerTurn({ turn, pageTitles, onFeedback }: {
  turn: Turn;
  pageTitles: PageTitles;
  onFeedback: (turn: Turn) => void;
}) {
  const generating = !turn.finished && !turn.error;
  // A private question shows nothing until its check is done (SPEC); the steps carry the wait.
  const showText = !turn.private || turn.checkResult !== undefined || turn.finished;
  const fragments = useQuoteFragments(turn.citations, turn.finished && !turn.error);
  const segments = showText ? answerSegments(turn.text, turn.citations, turn.unsupported, fragments) : [];
  const numbers = sourceNumbers(turn.text, turn.citations);
  return (
    <li className="ask-turn" data-restored={turn.restored ? '' : undefined}>
      <p className="ask-question"><span className="assistant-sr">You asked: </span>{turn.question}</p>
      <div className="ask-answer" aria-label="Answer">
        <AiLabel>AI answer</AiLabel>
        {segments.length > 0 && (
          <p className="ask-answer-text">
            {groupSegments(segments).map((group, index) => (
              Array.isArray(group)
                ? <Fragment key={index}>{renderInlines(group, numbers, pageTitles)}</Fragment>
                : <mark key={index} className="ask-unsupported" title="Not found in the project pages">
                  {renderInlines(group.children, numbers, pageTitles)}
                </mark>
            ))}
            {turn.incomplete && turn.text && <span className="ask-incomplete"> (incomplete)</span>}
          </p>
        )}
        {generating && <Shimmer />}
        {generating && <Steps steps={turn.steps} />}
        {turn.error && <p className="ask-error">{turn.error}</p>}
        {turn.finished && !turn.error && <Sources citations={turn.citations} numbers={numbers} pageTitles={pageTitles} fragments={fragments} />}
        {turn.finished && !turn.error && (turn.checkResult?.passed === true || turn.turnId) && (
          <div className="ask-answer-foot">
            {turn.checkResult?.passed === true
              ? <p className="ask-checked"><span aria-hidden="true">✓ </span>Checked by a second model</p>
              : <span />}
            {turn.turnId && (
              turn.feedback === 'sent'
                ? <span className="ask-thanks" role="status">Thanks — noted</span>
                : <button type="button" className="ask-feedback" aria-label="This answer is wrong" disabled={turn.feedback === 'sending'} onClick={() => onFeedback(turn)}>👎</button>
            )}
          </div>
        )}
        <HowItAnswered trace={turn.trace} checkResult={turn.checkResult} />
      </div>
    </li>
  );
}

function HowItMatched({ notes, pageTitles }: { notes: MatchNotes; pageTitles: PageTitles }) {
  return (
    <details className="ask-panel">
      <summary>How it matched</summary>
      <div className="ask-panel-block">
        {notes.requirements.length > 0 && (
          <ul>
            {notes.requirements.map((requirement, index) => {
              const finding = notes.findings.find((item) => item.requirement_id === requirement.id);
              const status = finding?.status === 'found' ? 'found'
                : finding?.status === 'not_found' ? 'not found'
                  : finding?.status === 'unchecked' ? "couldn't be checked" : 'pending';
              return (
                <li key={index}>
                  {requirement.id} · {requirement.text} · {status}
                  {finding && finding.sections.length > 0 && (
                    <ul>
                      {finding.sections.map((section, sectionIndex) => {
                        const href = safeSiteUrl(section.url);
                        const label = sourceLabel(section.url, section.title, pageTitles);
                        return <li key={sectionIndex}>{href ? <a href={href}>{label}</a> : label}</li>;
                      })}
                    </ul>
                  )}
                  {finding?.provider !== undefined && <p>{finding.provider ?? 'no model'}</p>}
                  {finding?.ms !== undefined && <p>{finding.ms} ms</p>}
                </li>
              );
            })}
          </ul>
        )}
        {notes.truncated && <p>The job ad was cut to 20,000 characters</p>}
      </div>
    </details>
  );
}

function MatchItemCard({ item, pageTitles }: { item: MatchItem; pageTitles: PageTitles }) {
  return (
    <li className="match-item">
      <p className="match-requirement">{truncate(item.requirement, REQUIREMENT_MAX_CHARS)}</p>
      {item.text && <p className="match-guide">{item.text}</p>}
      <ul className="match-sections">
        {item.sections.map((section, index) => {
          const href = safeSiteUrl(section.url);
          const label = sectionLine(section, pageTitles);
          return (
            <li key={index}>
              <span aria-hidden="true">→ </span>
              {href ? <a href={href}>{label}</a> : label}
            </li>
          );
        })}
      </ul>
    </li>
  );
}

export default function Assistant({ variant = 'home', examples, pageTitles = {}, intro, example }: AssistantProps) {
  const inputId = useId();
  const hintId = useId();
  const offlineId = useId();
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const [input, setInput] = useState('');
  const [override, setOverride] = useState<'ask' | 'match' | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [queued, setQueued] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState('Checking database…');
  const [announcement, setAnnouncement] = useState('');
  // Where the job match sits among the questions (the hook keeps only the latest match).
  const [matchAt, setMatchAt] = useState<number | null>(null);
  const [matchLink, setMatchLink] = useState<string | null>(null);

  // The island hydrates when it scrolls into view (client:visible), so a visitor can type into
  // the server-rendered box first. React keeps that text at hydration but would wipe it on the
  // next render; take it into state before any other update commits.
  useLayoutEffect(() => {
    const early = inputRef.current?.value;
    if (early) setInput(early);
  }, []);

  const askState = useAsk({ persist: true });
  const { turns, readiness, forgotten, prepare, ask, recordFeedback } = askState;
  const matchState = useMatch();
  const busy = askState.inProgress || matchState.inProgress;

  // The dev-only database line (/api/health exists only when app_env is dev).
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const controller = new AbortController();
    const { signal } = controller;
    async function checkHealth() {
      try {
        const response = await fetch('/api/health', { signal });
        if (signal.aborted) return;
        const body: unknown = await response.json();
        if (signal.aborted) return;
        let message = 'db: down';
        if (typeof body === 'object' && body !== null) {
          if (response.status === 200 && 'pgvector' in body && typeof body.pgvector === 'string') {
            message = `db: ok · pgvector ${body.pgvector}`;
          } else if (response.status === 503 && 'db' in body && body.db === 'ok') {
            message = 'db: ok · pgvector missing';
          }
        }
        setHealthStatus(message);
      } catch {
        if (!signal.aborted) setHealthStatus('db: down');
      }
    }
    void checkHealth();
    return () => controller.abort();
  }, []);

  // The textarea grows with its text; CSS caps it at about eight lines.
  useLayoutEffect(() => {
    const field = inputRef.current;
    if (!field) return;
    field.style.blockSize = 'auto';
    const border = field.offsetHeight - field.clientHeight;
    if (field.scrollHeight > 0) field.style.blockSize = `${field.scrollHeight + border}px`;
  }, [input]);

  // Panel only: the answers scroll inside the panel and follow new text while the reader is
  // at the bottom; scrolling up to read an earlier answer stops the follow.
  const answersRef = useRef<HTMLDivElement>(null);
  const followAnswers = useRef(true);
  function onAnswersScroll() {
    const box = answersRef.current;
    if (box) followAnswers.current = box.scrollHeight - box.scrollTop - box.clientHeight < 48;
  }
  const answerCount = useRef(0);
  useLayoutEffect(() => {
    const box = answersRef.current;
    if (variant !== 'panel' || !box) return;
    // A new question or job match always brings the reader back to the bottom.
    const count = turns.length + (matchAt === null ? 0 : 1);
    if (count !== answerCount.current) followAnswers.current = true;
    answerCount.current = count;
    if (followAnswers.current) box.scrollTop = box.scrollHeight;
  });

  // Home only: a new question or job match is scrolled into view, unless its start is already
  // in the upper part of the screen (the first answer on a desktop sits next to the box).
  const turnsRef = useRef<HTMLOListElement>(null);
  const scrollTo = useRef<'turn' | 'match' | null>(null);
  useEffect(() => {
    const target = scrollTo.current;
    const list = turnsRef.current;
    if (variant !== 'home' || target === null || !list) return;
    const item = target === 'match'
      ? list.querySelector<HTMLElement>(':scope > .match-turn')
      : list.querySelector<HTMLElement>(':scope > .ask-turn:last-child');
    if (!item) return;
    scrollTo.current = null;
    const top = item.getBoundingClientRect().top;
    if (top >= 0 && top <= window.innerHeight * 0.6) return;
    const reduce = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (typeof item.scrollIntoView === 'function') item.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
  });

  function started(target: 'turn' | 'match') {
    scrollTo.current = target;
    setInput('');
    setOverride(null);
  }

  // A question sent before /api/ping answered goes out once the backend is ready.
  useEffect(() => {
    if (queued === null) return;
    if (readiness === 'ready') {
      setQueued(null);
      if (ask(queued).started) started('turn');
    } else if (readiness === 'offline') {
      setQueued(null);
    }
  }, [queued, readiness]);

  // One status line for screen readers: started, finished, failed. Never the streaming text.
  const previous = useRef({ ask: false, match: false });
  useEffect(() => {
    const was = previous.current;
    if (askState.inProgress && !was.ask) setAnnouncement('Answering your question…');
    if (!askState.inProgress && was.ask) {
      const last = turns[turns.length - 1];
      setAnnouncement(last?.error ?? 'Answer ready.');
    }
    if (matchState.inProgress && !was.match) setAnnouncement('Matching the job ad…');
    if (!matchState.inProgress && was.match) setAnnouncement(matchState.error ?? 'Job match ready.');
    previous.current = { ask: askState.inProgress, match: matchState.inProgress };
  }, [askState.inProgress, matchState.inProgress]);

  const trimmed = input.trim();
  const length = codePoints(trimmed);
  const detected = detectInput(input);
  let mode: Mode = detected.kind;
  if (trimmed && override === 'ask' && length <= ASK_MAX_CHARS) mode = 'ask';
  else if (trimmed && override === 'match') mode = 'match';

  function send(text: string, chosen: Mode) {
    const value = text.trim();
    if (!value || busy || queued !== null) return;
    if (chosen === 'bad-link') {
      setNotice(BAD_LINK_MESSAGE);
      setAnnouncement(BAD_LINK_MESSAGE);
      return;
    }
    setNotice(null);
    if (chosen === 'match') {
      const kind = detectInput(value);
      setMatchAt(turns.length);
      setMatchLink(kind.kind === 'match' && kind.source === 'link' ? linkLabel(value) : null);
      if (matchState.match(value).started) started('match');
      return;
    }
    // Offline: prepare pings again and the question waits for it, like before the first
    // ping; if the backend is still unreachable the offline sentence comes back.
    prepare();
    if (readiness === 'ready') {
      if (ask(value).started) started('turn');
    } else {
      setQueued(value);
    }
  }

  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    send(input, mode);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      send(input, mode);
    }
  }

  function tryExample(question: string) {
    prepare();
    setInput(question);
    setOverride(null);
    send(question, 'ask');
  }

  const matchShown = matchAt !== null && (matchState.inProgress || matchState.result || matchState.error || matchState.notes);
  const asked = turns.length > 0 || matchShown;
  const insertAt = Math.min(matchAt ?? turns.length, turns.length);
  const items = matchState.result?.items.filter((item) => item.sections.length > 0) ?? [];

  const matchBlock = matchShown ? (
    <li className="match-turn" key="match">
      <div className="match-result" aria-label="Job match">
        <AiLabel>Job match</AiLabel>
        {matchLink && <p className="match-source">{matchLink}</p>}
        {matchState.inProgress && <Shimmer />}
        {matchState.inProgress && <Steps steps={matchState.steps} />}
        {matchState.result && (
          <>
            {items.length > 0 && (
              <>
                <h3 className="match-title">Where my pages cover this role</h3>
                <ul className="match-items">
                  {items.map((item, index) => <MatchItemCard key={index} item={item} pageTitles={pageTitles} />)}
                </ul>
              </>
            )}
            {matchState.result.not_covered.length > 0 && (
              <div className="match-gaps">
                <p>Not covered on the pages:</p>
                <ul aria-label="Not covered on the pages">
                  {matchState.result.not_covered.map((requirement, index) => <li key={index}>{requirement}</li>)}
                </ul>
              </div>
            )}
            {matchState.result.unchecked > 0 && (
              <p className="match-note">
                {matchState.result.unchecked} {matchState.result.unchecked === 1 ? 'requirement' : 'requirements'} couldn't be checked
              </p>
            )}
            <p className="match-note">Showing up to 12 requirements from the ad.</p>
          </>
        )}
        {matchState.error && <p className="ask-error">{matchState.error}</p>}
        {matchState.done && matchState.result?.match_id && !matchState.error && (
          <div className="ask-answer-foot">
            <p className="match-followup">Follow-up questions don't include this job ad.</p>
            {matchState.feedback === 'sent'
              ? <span className="ask-thanks" role="status">Thanks — noted</span>
              : <button type="button" className="ask-feedback" aria-label="This guide is wrong" disabled={matchState.feedback === 'sending'} onClick={() => { void matchState.recordFeedback(); }}>👎</button>}
          </div>
        )}
        {matchState.notes && <HowItMatched notes={matchState.notes} pageTitles={pageTitles} />}
      </div>
    </li>
  ) : null;

  const conversation = (
    <>
      {forgotten && turns.length > 0 && (
        <p className="ask-forgotten">Earlier questions are no longer in the assistant's memory.</p>
      )}
      <ol className="ask-turns" aria-label="Conversation" ref={turnsRef}>
        {turns.slice(0, insertAt).map((turn) => (
          <AnswerTurn key={turn.id} turn={turn} pageTitles={pageTitles} onFeedback={(item) => { void recordFeedback(item); }} />
        ))}
        {matchBlock}
        {turns.slice(insertAt).map((turn) => (
          <AnswerTurn key={turn.id} turn={turn} pageTitles={pageTitles} onFeedback={(item) => { void recordFeedback(item); }} />
        ))}
      </ol>
    </>
  );

  const switchLink = trimmed && !busy ? (
    mode === 'ask'
      ? <button type="button" className="ask-switch" onClick={() => { setOverride('match'); setNotice(null); }}>Match this as a job ad</button>
      : length <= ASK_MAX_CHARS
        ? <button type="button" className="ask-switch" onClick={() => { setOverride('ask'); setNotice(null); }}>Ask this as a question instead</button>
        : null
  ) : null;

  const waiting = queued !== null && readiness === 'pending';
  const offline = readiness === 'offline' && mode === 'ask';
  const describedBy = [notice ? hintId : null, offline ? offlineId : null].filter(Boolean).join(' ') || undefined;

  // The offline sentence is spoken through the one status line, like the bad-link hint.
  useEffect(() => {
    if (offline) setAnnouncement(OFFLINE_MESSAGE);
  }, [offline]);

  const form = (
    <form className="ask-form" onSubmit={submit}>
      <label htmlFor={inputId} className="assistant-sr">Ask a question, or paste a job ad</label>
      <div className="ask-box">
        <textarea
          id={inputId}
          ref={inputRef}
          className="ask-input"
          rows={2}
          placeholder="Ask a question, or paste a job ad"
          value={input}
          aria-describedby={describedBy}
          onChange={(event) => {
            setInput(event.target.value);
            setNotice(null);
            if (!event.target.value.trim()) setOverride(null);
          }}
          onFocus={prepare}
          onKeyDown={onKeyDown}
        />
        <button type="submit" className="ask-send">{mode === 'ask' ? 'Ask' : 'Match job'}</button>
      </div>
      {(switchLink || notice || waiting || offline) && (
        <div className="ask-below">
          {notice && <p id={hintId} className="ask-hint">{notice}</p>}
          {waiting && <p className="ask-hint">Getting ready…</p>}
          {offline && <p id={offlineId} className="ask-hint">{OFFLINE_MESSAGE}</p>}
          {switchLink}
        </div>
      )}
    </form>
  );

  const examplesGroup = examples.length > 0 && (
    <div role="group" aria-label="Example questions" className="ask-examples">
      <span className="ask-examples-label" aria-hidden="true">Try:</span>
      <ul>
        {examples.map((question) => (
          <li key={question}>
            <button type="button" className="ask-example" onClick={() => tryExample(question)}>{question}</button>
          </li>
        ))}
      </ul>
    </div>
  );

  const notes = (
    <>
      <p className="ask-note">
        Your question or job ad is processed by AI providers. <a href="/privacy">See Privacy.</a>
      </p>
      {import.meta.env.DEV && <p className="ask-dev">{healthStatus}</p>}
    </>
  );

  const status = <p className="assistant-sr" role="status">{announcement}</p>;

  if (variant === 'panel') {
    // Project page (9p): one instance is the right column from 960px and the bottom sheet
    // below that ([slug].astro, project-page.css). Answers scroll above the box, like a chat;
    // this page's examples show until the first question.
    return (
      <section id="ask" aria-label="Ask about my projects" className="assistant assistant-panel">
        {asked && (
          <div className="assistant-answers" ref={answersRef} onScroll={onAnswersScroll}>{conversation}</div>
        )}
        {form}
        {!asked && examplesGroup}
        {notes}
        {status}
      </section>
    );
  }

  return (
    <section id="ask" aria-label="Ask about my projects" className="assistant assistant-home">
      <div className="assistant-start">
        {intro}
        {form}
        {examplesGroup}
        {notes}
      </div>
      <div className="assistant-answers">
        {asked ? conversation : example}
      </div>
      {status}
    </section>
  );
}
