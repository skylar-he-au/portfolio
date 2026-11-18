import { useEffect, useId, useRef, useState } from 'react';
import type { SubmitEvent } from 'react';
import { safeSiteUrl } from '../lib/answer';
import {
  MATCH_DROPPED_MESSAGE, MATCH_MAX_CHARS, MATCH_MESSAGES, matchErrorMessage, matchHttpMessage,
  matchInputLength, runMatch, sendMatchFeedback,
} from '../lib/match';
import type { MatchEvent, MatchNotes, MatchResult, MatchSection } from '../lib/match';
import { acquireWidget, HttpError, prepareWidget, VerificationError } from '../lib/session';

function Sections({ sections }: { sections: MatchSection[] }) {
  if (sections.length === 0) return null;
  return (
    <ul>
      {sections.map((section, index) => {
        const href = safeSiteUrl(section.url);
        return <li key={index}>{href ? <a href={href}>{section.title}</a> : section.title}</li>;
      })}
    </ul>
  );
}

function HowItMatched({ notes }: { notes: MatchNotes }) {
  return (
    <details className="ask-panel">
      <summary>How it matched</summary>
      <div className="ask-panel-block ask-answer">
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
                  {finding && <Sections sections={finding.sections} />}
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

function stageMessage(data: Extract<MatchEvent, { event: 'stage' }>['data']): string {
  switch (data.stage) {
    case 'fetching': return 'Fetching the job ad…';
    case 'reading': return 'Reading the job ad…';
    case 'matching': return data.count === undefined ? 'Matching requirements to my project pages…'
      : `Matching ${data.count} requirements to my project pages…`;
    case 'writing': return 'Writing the guide…';
    case 'checking': return 'Checking the guide…';
  }
}

export default function Match() {
  const inputId = useId();
  const [input, setInput] = useState('');
  const [inProgress, setInProgress] = useState(false);
  const [progress, setProgress] = useState('');
  const [notes, setNotes] = useState<MatchNotes>();
  const [result, setResult] = useState<MatchResult>();
  const [error, setError] = useState<string>();
  const [done, setDone] = useState(false);
  const [feedback, setFeedback] = useState<'idle' | 'sending' | 'sent'>('idle');
  const mounted = useRef(true);
  const requestRef = useRef<AbortController | null>(null);
  const feedbackRef = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    const releaseWidget = acquireWidget();
    return () => {
      mounted.current = false;
      requestRef.current?.abort();
      feedbackRef.current?.abort();
      releaseWidget();
    };
  }, []);

  async function recordFeedback() {
    if (!mounted.current || !done || error || !result?.match_id || feedback !== 'idle' || feedbackRef.current) return;
    const controller = new AbortController();
    feedbackRef.current = controller;
    setFeedback('sending');
    const sent = await sendMatchFeedback(result.match_id, controller.signal);
    if (!mounted.current || controller.signal.aborted) return;
    feedbackRef.current = null;
    setFeedback(sent ? 'sent' : 'idle');
  }

  async function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || requestRef.current) return;
    feedbackRef.current?.abort();
    feedbackRef.current = null;
    setResult(undefined);
    setNotes(undefined);
    setError(undefined);
    setProgress('');
    setDone(false);
    setFeedback('idle');
    if (matchInputLength(trimmed) > MATCH_MAX_CHARS) {
      setError(MATCH_MESSAGES.too_long);
      return;
    }

    const controller = new AbortController();
    requestRef.current = controller;
    setInProgress(true);
    try {
      await runMatch(trimmed, {
        signal: controller.signal,
        onEvent(event) {
          if (!mounted.current || controller.signal.aborted) return;
          switch (event.event) {
            case 'stage': setProgress(stageMessage(event.data)); break;
            case 'notes': setNotes(event.data); break;
            case 'result': setResult(event.data); break;
            case 'done': setDone(true); break;
            case 'error': setError(matchErrorMessage(event.data.code, event.data.message)); break;
          }
        },
      });
    } catch (error) {
      if (!mounted.current || controller.signal.aborted) return;
      setError(error instanceof VerificationError ? MATCH_MESSAGES.unavailable
        : error instanceof HttpError ? matchHttpMessage(error) : MATCH_DROPPED_MESSAGE);
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
      if (mounted.current && !controller.signal.aborted) {
        setInProgress(false);
        setProgress('');
      }
    }
  }

  const items = result?.items.filter((item) => item.sections.length > 0) ?? [];
  return (
    <div className="ask" style={{ minInlineSize: 'var(--space-0)' }}>
      <form className="ask-form" onSubmit={submit}>
        <label htmlFor={inputId} className="ask-label">Paste a job ad</label>
        <textarea
          id={inputId}
          className="ask-control ask-input"
          rows={6}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onFocus={prepareWidget}
        />
        <button type="submit" disabled={inProgress || !input.trim()} className="ask-control ask-send">Match</button>
      </form>
      <p className="ask-note">
        Your job ad is processed by AI providers. <a href="/privacy">See Privacy.</a>
      </p>
      {progress && <p className="ask-status" role="status">{progress}</p>}
      {result && (
        <div className="ask-answer" aria-live="polite">
          {items.length > 0 && <>
            <p>For this role, look at:</p>
            <ul>
              {items.map((item, index) => (
                <li key={index}>{item.text}<Sections sections={item.sections} /></li>
              ))}
            </ul>
          </>}
          {result.not_covered.length > 0 && <p>Not covered on the pages: {result.not_covered.join(', ')}</p>}
          {result.unchecked > 0 && <p>{result.unchecked} {result.unchecked === 1 ? 'requirement' : 'requirements'} couldn't be checked</p>}
          {result.top8 && <p>Top 8 requirements of the job ad shown.</p>}
        </div>
      )}
      {done && result?.match_id && !error && (
        feedback === 'sent'
          ? <span className="ask-checked" role="status">Thanks — noted</span>
          : <button type="button" className="ask-control" aria-label="This guide is wrong" disabled={feedback === 'sending'} onClick={() => { void recordFeedback(); }}>👎</button>
      )}
      {error && <p className="ask-error" role="alert">{error}</p>}
      {notes && <HowItMatched notes={notes} />}
    </div>
  );
}
