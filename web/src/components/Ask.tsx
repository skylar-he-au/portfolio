import { Fragment, useEffect, useId, useState } from 'react';
import type { SubmitEvent } from 'react';
import { answerSegments, unreferencedCitations } from '../lib/answer';
import type { InlineSegment } from '../lib/answer';
import { useAsk } from '../hooks/useAsk';
import HowItAnswered from './HowItAnswered';

function renderInline(segment: InlineSegment, key: number) {
  return segment.type === 'link'
    ? <a key={key} href={segment.href} title={segment.title}>{segment.text}</a>
    : segment.text;
}

export default function Ask({ examples }: { examples: string[] }) {
  const inputId = useId();
  const [question, setQuestion] = useState('');
  const [healthStatus, setHealthStatus] = useState('Checking database…');
  const { turns, readiness, inProgress, prepare, ask, recordFeedback } = useAsk();

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    if (import.meta.env.DEV) {
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
    }

    return () => controller.abort();
  }, []);

  function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    const { started } = ask(trimmed);
    if (started) setQuestion('');
  }

  // Styles live in src/styles/global.css (.ask-*); the inline token keeps grid children shrinkable.
  return (
    <div className="ask" style={{ minInlineSize: 'var(--space-0)' }}>
      <ol className="ask-turns" aria-label="Conversation">
        {turns.map((turn) => {
          const see = unreferencedCitations(turn.text, turn.citations);
          return (
            <li key={turn.id} className="ask-turn">
              <p className="ask-question"><span className="ask-label">You asked: </span>{turn.question}</p>
              <div className="ask-answer" aria-label="Answer" aria-live="polite" aria-atomic="true">
                {answerSegments(turn.text, turn.citations, turn.unsupported).map((segment, index) => (
                  segment.type === 'unsupported'
                    ? <mark key={index} className="ask-unsupported" title="Not found in the project pages">
                      {segment.children.map(renderInline)}
                    </mark>
                    : renderInline(segment, index)
                ))}
                {turn.checking && <span className="ask-checking">checking…</span>}
                {turn.checkResult?.passed === true && <span className="ask-checked">checked</span>}
                {turn.incomplete && turn.text && <span className="ask-incomplete">(incomplete)</span>}
                {see.length > 0 && (
                  <p className="ask-see">
                    See: {see.map((citation, index) => (
                      <Fragment key={citation.n}>
                        {index > 0 && ', '}<a href={citation.url}>{citation.heading}</a>
                      </Fragment>
                    ))}
                  </p>
                )}
              </div>
              {turn.finished && turn.turnId && !turn.error && (
                turn.feedback === 'sent'
                  ? <span className="ask-checked" role="status">Thanks — noted</span>
                  : <button type="button" className="ask-control" aria-label="This answer is wrong" disabled={turn.feedback === 'sending'} onClick={() => { void recordFeedback(turn); }}>👎</button>
              )}
              {turn.error && <p className="ask-error" role="alert">{turn.error}</p>}
              <HowItAnswered trace={turn.trace} checkResult={turn.checkResult} />
            </li>
          );
        })}
      </ol>
      <form className="ask-form" onSubmit={submit}>
        <label htmlFor={inputId} className="ask-label">Ask about my projects</label>
        <textarea
          id={inputId}
          className="ask-control ask-input"
          maxLength={500}
          rows={4}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onFocus={prepare}
        />
        {(readiness === 'pending' || readiness === 'offline') && (
          <p role="status" className="ask-status">
            {readiness === 'pending' ? 'getting ready…' : 'Assistant is offline'}
          </p>
        )}
        <button type="submit" disabled={readiness !== 'ready' || inProgress || !question.trim()} className="ask-control ask-send">
          Send
        </button>
      </form>
      <div role="group" aria-label="Example questions" className="ask-examples">
        <span className="ask-label">Example questions</span>
        <div className="ask-example-list">
          {examples.map((example) => (
            <button key={example} type="button" onClick={() => { prepare(); setQuestion(example); }} className="ask-control ask-example">
              {example}
            </button>
          ))}
        </div>
      </div>
      <p className="ask-note">
        Your question is processed by AI providers. <a href="/privacy">See Privacy.</a>
      </p>
      {import.meta.env.DEV && <p className="ask-note">{healthStatus}</p>}
    </div>
  );
}
