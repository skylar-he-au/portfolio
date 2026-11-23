import { useId, useState } from 'react';
import type { SubmitEvent } from 'react';
import { safeSiteUrl } from '../lib/answer';
import { sectionLabel } from '../lib/match';
import type { MatchEvent, MatchNotes, MatchSection, PageTitles } from '../lib/match';
import { useMatch } from '../hooks/useMatch';

function Sections({ sections, pageTitles }: { sections: MatchSection[]; pageTitles: PageTitles }) {
  if (sections.length === 0) return null;
  return (
    <ul>
      {sections.map((section, index) => {
        const href = safeSiteUrl(section.url);
        const label = sectionLabel(section, pageTitles);
        return <li key={index}>{href ? <a href={href}>{label}</a> : label}</li>;
      })}
    </ul>
  );
}

function HowItMatched({ notes, pageTitles }: { notes: MatchNotes; pageTitles: PageTitles }) {
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
                  {finding && <Sections sections={finding.sections} pageTitles={pageTitles} />}
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
      : `Matching ${data.count} ${data.count === 1 ? 'requirement' : 'requirements'} to my project pages…`;
    case 'checking': return 'Checking the sections…';
    case 'writing': return 'Writing the guide…';
  }
}

/** `pageTitles`: project page path -> title, from the pages' front matter (see `pages/index.astro`). */
export default function Match({ pageTitles = {} }: { pageTitles?: PageTitles }) {
  const inputId = useId();
  const [input, setInput] = useState('');
  const { inProgress, stage, notes, result, error, done, feedback, prepare, match, recordFeedback } = useMatch();

  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    match(input);
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
          onFocus={prepare}
        />
        <button type="submit" disabled={inProgress || !input.trim()} className="ask-control ask-send">Match</button>
      </form>
      <p className="ask-note">
        Your job ad is processed by AI providers. <a href="/privacy">See Privacy.</a>
      </p>
      {inProgress && stage && <p className="ask-status" role="status">{stageMessage(stage)}</p>}
      {result && (
        <div className="ask-answer" aria-live="polite">
          {items.length > 0 && <>
            <p>Where my pages cover this role:</p>
            <ul>
              {items.map((item, index) => (
                <li key={index}>{item.text}<Sections sections={item.sections} pageTitles={pageTitles} /></li>
              ))}
            </ul>
          </>}
          {result.not_covered.length > 0 && <>
            <p>Not covered on the pages:</p>
            <ul aria-label="Not covered on the pages">
              {result.not_covered.map((requirement, index) => <li key={index}>{requirement}</li>)}
            </ul>
          </>}
          {result.unchecked > 0 && <p>{result.unchecked} {result.unchecked === 1 ? 'requirement' : 'requirements'} couldn't be checked</p>}
          {result.top8 && <p>Top 12 requirements of the job ad shown.</p>}
        </div>
      )}
      {done && result?.match_id && !error && (
        feedback === 'sent'
          ? <span className="ask-checked" role="status">Thanks — noted</span>
          : <button type="button" className="ask-control" style={{ justifySelf: 'start' }} aria-label="This guide is wrong" disabled={feedback === 'sending'} onClick={() => { void recordFeedback(); }}>👎</button>
      )}
      {error && <p className="ask-error" role="alert">{error}</p>}
      {notes && <HowItMatched notes={notes} pageTitles={pageTitles} />}
    </div>
  );
}
