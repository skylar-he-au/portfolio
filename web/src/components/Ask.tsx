import { Fragment, useEffect, useId, useRef, useState } from 'react';
import type { SubmitEvent } from 'react';
import { answerSegments, unreferencedCitations } from '../lib/answer';
import type { Citation, InlineSegment } from '../lib/answer';
import { sendFeedback } from '../lib/feedback';
import { parseSse } from '../lib/sse';
import { clearToken, getToken, isSession, setToken } from '../lib/session';
import { renderWidget } from '../lib/turnstile';
import type { WidgetController } from '../lib/turnstile';
import HowItAnswered from './HowItAnswered';
import type { AnswerTrace, CheckResult } from './HowItAnswered';

interface Turn {
  id: number;
  turnId?: string;
  finished: boolean;
  feedback: 'idle' | 'sending' | 'sent';
  question: string;
  text: string;
  citations: Citation[];
  unsupported: string[];
  checking: boolean;
  checkResult?: CheckResult;
  incomplete: boolean;
  error?: string;
  trace?: AnswerTrace;
}

const droppedMessage = 'The connection dropped — please try again.';
const genericMessage = 'Something went wrong — please try again.';
const unavailableMessage = 'The assistant is unavailable right now — please try again later or email me.';
let warnedMissingSitekey = false;

class VerificationError extends Error {}

class HttpError extends Error {
  constructor(readonly status: number, readonly code: unknown) {
    let message = genericMessage;
    switch (code) {
      case 'auth': message = 'Please try again.'; break;
      case 'slow_down': message = 'Too many requests — please wait a minute and try again.'; break;
      case 'limit_day': message = "Today's AI limit is reached — please email me"; break;
      case 'paused_month': message = 'The AI is paused until next month — please email me'; break;
      case 'unavailable': message = unavailableMessage; break;
    }
    super(status === 422 ? 'Please ask a question of up to 500 characters.' : message);
  }
}

async function responseError(response: Response): Promise<HttpError> {
  let code: unknown;
  try {
    const body: unknown = await response.json();
    if (typeof body === 'object' && body !== null && 'code' in body) code = body.code;
  } catch {
    // Non-JSON rejections use the generic message.
  }
  return new HttpError(response.status, code);
}

function errorMessage(code: unknown, message: unknown): string {
  switch (code) {
    case 'unavailable': return unavailableMessage;
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
  const widgetContainer = useRef<HTMLDivElement | null>(null);
  const widgetRef = useRef<WidgetController | null>(null);
  const widgetExecuted = useRef(false);
  const mounted = useRef(true);
  const nextId = useRef(0);

  useEffect(() => {
    mounted.current = true;
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
      mounted.current = false;
      controller.abort();
      requestRef.current?.abort();
      widgetRef.current?.remove();
      widgetRef.current = null;
      widgetExecuted.current = false;
    };
  }, []);

  function getWidget(): WidgetController {
    const sitekey = import.meta.env.PUBLIC_TURNSTILE_SITEKEY;
    if (!sitekey) {
      if (!warnedMissingSitekey) {
        console.warn('PUBLIC_TURNSTILE_SITEKEY is missing; the assistant is unavailable.');
        warnedMissingSitekey = true;
      }
      throw new VerificationError(unavailableMessage);
    }
    if (!mounted.current || !widgetContainer.current) throw new VerificationError(unavailableMessage);
    if (!widgetRef.current) widgetRef.current = renderWidget(widgetContainer.current, sitekey);
    return widgetRef.current;
  }

  function prepareWidget() {
    if (import.meta.env.PUBLIC_TURNSTILE_SITEKEY) getWidget();
  }

  async function recordFeedback(turn: Turn) {
    if (!mounted.current || !turn.turnId || turn.feedback !== 'idle') return;
    setTurns((previous) => previous.map((item) => item.id === turn.id ? { ...item, feedback: 'sending' } : item));
    const sent = await sendFeedback(turn.turnId);
    if (!mounted.current) return;
    setTurns((previous) => previous.map((item) => item.id === turn.id ? { ...item, feedback: sent ? 'sent' : 'idle' } : item));
  }

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
      finished: false, feedback: 'idle',
    }]);

    function update(change: (turn: Turn) => Turn) {
      if (!controller.signal.aborted) {
        setTurns((previous) => previous.map((turn) => turn.id === id ? change(turn) : turn));
      }
    }

    let terminal = false;
    let privateRoute = false;
    let receivedMeta = false;
    let receivedTurnMeta = false;
    let verificationRetries = 0;

    async function createSession(reset = false): Promise<string> {
      let turnstile: string;
      try {
        const widget = getWidget();
        if (reset || widgetExecuted.current) widget.reset();
        while (true) {
          try {
            widgetExecuted.current = true;
            turnstile = await widget.execute();
            break;
          } catch (error) {
            if (controller.signal.aborted || verificationRetries >= 1) throw error;
            verificationRetries++;
            widget.reset();
          }
        }
      } catch {
        throw new VerificationError(unavailableMessage);
      }
      controller.signal.throwIfAborted();
      const response = await fetch('/api/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ turnstile }),
        signal: controller.signal,
      });
      controller.signal.throwIfAborted();
      if (!response.ok) throw await responseError(response);
      const session: unknown = await response.json();
      controller.signal.throwIfAborted();
      if (!isSession(session)) throw new HttpError(response.status, undefined);
      setToken(session);
      return session.token;
    }

    try {
      if (!import.meta.env.PUBLIC_TURNSTILE_SITEKEY) getWidget();
      let token = getToken() ?? await createSession();
      let response: Response;
      for (let attempt = 0; ; attempt++) {
        controller.signal.throwIfAborted();
        response = await fetch('/api/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
          body: JSON.stringify({ question: trimmed, conversation_id: conversationId }),
          signal: controller.signal,
        });
        controller.signal.throwIfAborted();
        if (response.ok) break;
        const error = await responseError(response);
        controller.signal.throwIfAborted();
        if (error.status === 401 && error.code === 'auth') {
          clearToken();
          if (attempt === 0) {
            token = await createSession(true);
            continue;
          }
        }
        throw error;
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
            if (!receivedTurnMeta) {
              const turnId = data.turn_id;
              if (typeof turnId === 'string') update((turn) => ({ ...turn, turnId }));
              receivedTurnMeta = true;
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
            update((turn) => ({ ...turn, text: '', citations: [], unsupported: [], checking: true, checkResult: undefined }));
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
            const checkResult = {
              passed: data.passed === true,
              reason: typeof data.reason === 'string' ? data.reason : null,
            };
            update((turn) => ({ ...turn, checking: false, unsupported, checkResult }));
            break;
          }
          case 'trace': {
            const trace = data as unknown as AnswerTrace;
            update((turn) => ({ ...turn, trace }));
            break;
          }
          case 'done':
            terminal = true;
            update((turn) => ({ ...turn, finished: true }));
            break;
          case 'error':
            terminal = true;
            update((turn) => ({ ...turn, error: errorMessage(data.code, data.message) }));
            break;
        }
        if (terminal) break;
      }
      if (!terminal) update((turn) => ({ ...turn, error: droppedMessage, incomplete: true }));
    } catch (error) {
      if (error instanceof HttpError || error instanceof VerificationError) {
        update((turn) => ({ ...turn, error: error.message }));
      } else if (!terminal) {
        update((turn) => ({ ...turn, error: droppedMessage, incomplete: true }));
      }
    } finally {
      update((turn) => ({ ...turn, checking: false }));
      if (requestRef.current === controller) requestRef.current = null;
      if (!controller.signal.aborted) setInProgress(false);
    }
  }

  // Styles live in src/styles/global.css (.ask-*); the inline token keeps grid children shrinkable.
  return (
    <div className="ask" style={{ minInlineSize: 'var(--space-0)' }}>
      <div ref={widgetContainer} hidden />
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
          onFocus={prepareWidget}
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
            <button key={example} type="button" onClick={() => { prepareWidget(); setQuestion(example); }} className="ask-control ask-example">
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
