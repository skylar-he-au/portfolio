import { Fragment, useEffect, useId, useRef, useState } from 'react';
import type { SubmitEvent } from 'react';
import { answerSegments, unreferencedCitations } from '../lib/answer';
import type { Citation, InlineSegment } from '../lib/answer';
import { parseSse } from '../lib/sse';
import HowItAnswered from './HowItAnswered';
import type { AnswerTrace } from './HowItAnswered';

interface Turn {
  id: number;
  question: string;
  text: string;
  citations: Citation[];
  unsupported: string[];
  checking: boolean;
  incomplete: boolean;
  error?: string;
  trace?: AnswerTrace;
}

const droppedMessage = 'The connection dropped — please try again.';
const genericMessage = 'Something went wrong — please try again.';

function errorMessage(code: unknown, message: unknown): string {
  switch (code) {
    case 'unavailable': return 'The assistant is unavailable right now — please try again later or email me.';
    case 'timeout': return 'That took too long — please try again.';
    case 'busy': return 'Please wait for the current answer to finish.';
    default: return typeof message === 'string' && message.trim() ? message : genericMessage;
  }
}

function renderInline(segment: InlineSegment, key: number) {
  return segment.type === 'link'
    ? <a key={key} href={segment.href} title={segment.title}>{segment.text}</a>
    : segment.text;
}

export default function Ask({ examples }: { examples: string[] }) {
  const inputId = useId();
  const [question, setQuestion] = useState('');
  const [readiness, setReadiness] = useState<'pending' | 'ready' | 'offline'>('pending');
  const [healthStatus, setHealthStatus] = useState('Checking database…');
  const [conversationId, setConversationId] = useState<string>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [inProgress, setInProgress] = useState(false);
  const requestRef = useRef<AbortController | null>(null);
  const nextId = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    async function checkPing() {
      try {
        const response = await fetch('/api/ping', { signal });
        if (!signal.aborted) setReadiness(response.status === 200 ? 'ready' : 'offline');
      } catch {
        if (!signal.aborted) setReadiness('offline');
      }
    }
    void checkPing();

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

    return () => {
      controller.abort();
      requestRef.current?.abort();
    };
  }, []);

  async function submit(event: SubmitEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || readiness !== 'ready' || requestRef.current) return;

    const controller = new AbortController();
    requestRef.current = controller;
    const id = nextId.current++;
    setInProgress(true);
    setQuestion('');
    setTurns((previous) => [...previous, {
      id, question: trimmed, text: '', citations: [], unsupported: [], checking: true, incomplete: false,
    }]);

    function update(change: (turn: Turn) => Turn) {
      if (!controller.signal.aborted) {
        setTurns((previous) => previous.map((turn) => turn.id === id ? change(turn) : turn));
      }
    }

    let terminal = false;
    let privateRoute = false;
    let receivedMeta = false;
    try {
      const response = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: trimmed, conversation_id: conversationId }),
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      if (!response.ok) {
        terminal = true;
        update((turn) => ({
          ...turn,
          error: response.status === 422 ? 'Please ask a question of up to 500 characters.' : genericMessage,
        }));
        return;
      }
      if (!response.body) throw new Error('Missing response stream');

      for await (const event of parseSse(response.body)) {
        if (controller.signal.aborted) return;
        const data = typeof event.data === 'object' && event.data !== null
          ? event.data as Record<string, unknown> : {};
        switch (event.event) {
          case 'meta':
            if (!receivedMeta && typeof data.conversation_id === 'string') {
              setConversationId(data.conversation_id);
              receivedMeta = true;
            }
            break;
          case 'route':
            privateRoute = data.private === true;
            if (privateRoute) update((turn) => ({ ...turn, text: '', citations: [] }));
            break;
          case 'token':
            if (!privateRoute && typeof data.text === 'string') {
              const text = data.text;
              update((turn) => ({ ...turn, text: turn.text + text }));
            }
            break;
          case 'reset':
            update((turn) => ({ ...turn, text: '', citations: [], unsupported: [], checking: true }));
            break;
          case 'answer':
            if (typeof data.text === 'string') {
              const text = data.text;
              const citations = Array.isArray(data.citations) ? data.citations as Citation[] : [];
              update((turn) => ({ ...turn, text, citations }));
            }
            break;
          case 'check': {
            const unsupported = Array.isArray(data.unsupported)
              ? data.unsupported.filter((span): span is string => typeof span === 'string') : [];
            update((turn) => ({ ...turn, checking: false, unsupported }));
            break;
          }
          case 'trace': {
            const trace = data as unknown as AnswerTrace;
            update((turn) => ({ ...turn, trace }));
            break;
          }
          case 'done':
            terminal = true;
            break;
          case 'error':
            terminal = true;
            update((turn) => ({ ...turn, error: errorMessage(data.code, data.message) }));
            break;
        }
        if (terminal) break;
      }
      if (!terminal) update((turn) => ({ ...turn, error: droppedMessage, incomplete: true }));
    } catch {
      if (!terminal) update((turn) => ({ ...turn, error: droppedMessage, incomplete: true }));
    } finally {
      update((turn) => ({ ...turn, checking: false }));
      if (requestRef.current === controller) requestRef.current = null;
      if (!controller.signal.aborted) setInProgress(false);
    }
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
              {turn.error && <p className="ask-error" role="alert">{turn.error}</p>}
              <HowItAnswered trace={turn.trace} />
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
        />
        {readiness !== 'ready' && (
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
            <button key={example} type="button" onClick={() => setQuestion(example)} className="ask-control ask-example">
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
