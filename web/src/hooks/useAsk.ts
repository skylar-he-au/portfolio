import { useEffect, useRef, useState } from 'react';
import type { Citation } from '../lib/answer';
import { adoptConversationId, loadConversation, saveConversation } from '../lib/conversation';
import type { StoredTurn } from '../lib/conversation';
import { sendFeedback } from '../lib/feedback';
import { parseSse } from '../lib/sse';
import { acquireWidget, clearToken, createSession, getToken, HttpError, prepareWidget, responseError, VerificationError } from '../lib/session';
import type { AnswerTrace, CheckResult } from '../components/HowItAnswered';

export type AskStepName = 'Rewrite' | 'Search' | 'Answer' | 'Check';
export interface AskStep { name: AskStepName; state: 'active' | 'done' }
export interface UseAskOptions { persist?: boolean }

export interface Turn {
  steps: AskStep[];
  private: boolean;
  restored?: boolean;
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

function errorMessage(code: unknown, message: unknown): string {
  switch (code) {
    case 'unavailable': return unavailableMessage;
    case 'timeout': return 'That took too long — please try again.';
    case 'busy': return 'Please wait for the current answer to finish.';
    default: return typeof message === 'string' && message.trim() ? message : genericMessage;
  }
}

function canStore(turn: Turn): boolean {
  return turn.finished && !turn.error && !turn.incomplete && (!turn.private || turn.checkResult !== undefined);
}

export function useAsk(options?: UseAskOptions): {
  turns: Turn[];
  readiness: 'idle' | 'pending' | 'ready' | 'offline';
  inProgress: boolean;
  forgotten: boolean;
  prepare: () => void;
  ask: (question: string) => { started: boolean; finished: Promise<void> };
  recordFeedback: (turn: Turn) => Promise<void>;
} {
  const persist = options?.persist ?? false;
  const [readiness, setReadiness] = useState<'idle' | 'pending' | 'ready' | 'offline'>('idle');
  const [turns, setTurnState] = useState<Turn[]>([]);
  const [inProgress, setInProgress] = useState(false);
  const [forgotten, setForgotten] = useState(false);
  const turnsRef = useRef<Turn[]>([]);
  const conversationId = useRef<string | undefined>(undefined);
  const forgottenRef = useRef(false);
  const requestRef = useRef<AbortController | null>(null);
  const pingRef = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const nextId = useRef(0);
  const restored = useRef(false);

  // Keep stream updates synchronous so saving at done includes every preceding event.
  function setTurns(change: (previous: Turn[]) => Turn[]) {
    turnsRef.current = change(turnsRef.current);
    setTurnState(turnsRef.current);
  }

  function persistTurns() {
    if (!persist || !conversationId.current) return;
    const saved: StoredTurn[] = turnsRef.current
      .filter(canStore)
      .map((turn) => ({
        turnId: turn.turnId, question: turn.question, text: turn.text, citations: turn.citations,
        unsupported: turn.unsupported, checkResult: turn.checkResult, trace: turn.trace,
        feedback: turn.feedback === 'sent' ? 'sent' : 'idle', private: turn.private,
      }));
    saveConversation({ v: 1, id: conversationId.current, turns: saved, ...(forgottenRef.current ? { forgotten: true } : {}) });
  }

  useEffect(() => {
    mounted.current = true;
    const releaseWidget = acquireWidget();
    return () => {
      mounted.current = false;
      pingRef.current?.abort();
      pingRef.current = null;
      requestRef.current?.abort();
      releaseWidget();
    };
  }, []);

  useEffect(() => {
    if (!persist || restored.current) return;
    restored.current = true;
    const stored = loadConversation();
    if (!stored) return;
    conversationId.current = stored.id;
    forgottenRef.current = stored.forgotten === true;
    setForgotten(forgottenRef.current);
    setTurns(() => stored.turns.map((turn) => ({
      ...turn, id: nextId.current++, finished: true, checking: false, incomplete: false,
      restored: true, steps: [], private: turn.private === true,
    })));
  }, [persist]);

  function prepare() {
    prepareWidget();
    if (!mounted.current || pingRef.current) return;
    const controller = new AbortController();
    pingRef.current = controller;
    const { signal } = controller;
    setReadiness('pending');
    async function checkPing() {
      try {
        const response = await fetch('/api/ping', { signal });
        if (!signal.aborted) setReadiness(response.status === 200 ? 'ready' : 'offline');
      } catch {
        if (!signal.aborted) setReadiness('offline');
      }
    }
    void checkPing();
  }

  async function recordFeedback(turn: Turn) {
    if (!mounted.current || !turn.turnId || turn.feedback !== 'idle') return;
    setTurns((previous) => previous.map((item) => item.id === turn.id ? { ...item, feedback: 'sending' } : item));
    const sent = await sendFeedback(turn.turnId);
    if (!mounted.current) return;
    setTurns((previous) => previous.map((item) => item.id === turn.id ? { ...item, feedback: sent ? 'sent' : 'idle' } : item));
    if (sent) persistTurns();
  }

  function ask(question: string): { started: boolean; finished: Promise<void> } {
    const trimmed = question.trim();
    if (!trimmed || readiness !== 'ready' || requestRef.current) return { started: false, finished: Promise.resolve() };

    const controller = new AbortController();
    requestRef.current = controller;
    const id = nextId.current++;
    setInProgress(true);
    setTurns((previous) => [...previous, {
      id, question: trimmed, text: '', citations: [], unsupported: [], checking: true, incomplete: false,
      finished: false, feedback: 'idle', steps: [], private: false,
    }]);

    function update(change: (turn: Turn) => Turn) {
      if (!controller.signal.aborted) {
        setTurns((previous) => previous.map((turn) => turn.id === id ? change(turn) : turn));
      }
    }

    return { started: true, finished: run() };

    async function run() {
      let terminal = false;
      let privateRoute = false;
      let receivedMeta = false;
      let receivedTurnMeta = false;

      try {
        let token = (import.meta.env.PUBLIC_TURNSTILE_SITEKEY && getToken()) || await createSession(controller.signal);
        let response: Response;
        for (let attempt = 0; ; attempt++) {
          controller.signal.throwIfAborted();
          response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
            body: JSON.stringify({ question: trimmed, conversation_id: conversationId.current }),
            signal: controller.signal,
          });
          controller.signal.throwIfAborted();
          if (response.ok) break;
          const error = await responseError(response);
          controller.signal.throwIfAborted();
          if (error.status === 401 && error.code === 'auth') {
            clearToken();
            if (attempt === 0) {
              token = await createSession(controller.signal, true);
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
                conversationId.current = data.conversation_id;
                if (persist) {
                  const adopted = adoptConversationId(loadConversation(), data.conversation_id);
                  if (adopted.replaced) {
                    forgottenRef.current = true;
                    setForgotten(true);
                    saveConversation(adopted.conversation);
                  }
                }
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
              if (privateRoute) update((turn) => ({ ...turn, private: true, text: '', citations: [] }));
              break;
            case 'status': {
              const name: AskStepName | undefined = data.node === 'rewrite' ? 'Rewrite'
                : data.node === 'retrieve' ? 'Search' : data.node === 'answer' ? 'Answer'
                  : data.node === 'check' ? 'Check' : undefined;
              if (name && (data.state === 'start' || data.state === 'end')) {
                const state = data.state === 'start' ? 'active' : 'done';
                update((turn) => ({ ...turn, steps: turn.steps.some((step) => step.name === name)
                  ? turn.steps.map((step) => step.name === name ? { name, state } : step)
                  : [...turn.steps, { name, state }] }));
              }
              break;
            }
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
            case 'done': {
              terminal = true;
              update((turn) => ({ ...turn, finished: true, steps: turn.steps.map((step) => ({ ...step, state: 'done' })) }));
              const completed = turnsRef.current.find((turn) => turn.id === id);
              if (completed && canStore(completed)) persistTurns();
              break;
            }
            case 'error':
              terminal = true;
              update((turn) => ({ ...turn, error: errorMessage(data.code, data.message) }));
              break;
          }
          if (terminal) break;
        }
        if (!terminal) update((turn) => ({ ...turn, error: droppedMessage, incomplete: true }));
      } catch (error) {
        if (error instanceof VerificationError) {
          update((turn) => ({ ...turn, error: unavailableMessage }));
        } else if (error instanceof HttpError) {
          const message = error.status === 422 ? 'Please ask a question of up to 500 characters.'
            : error.code === 'unavailable' ? unavailableMessage : error.message;
          update((turn) => ({ ...turn, error: message }));
        } else if (!terminal) {
          update((turn) => ({ ...turn, error: droppedMessage, incomplete: true }));
        }
      } finally {
        update((turn) => ({ ...turn, checking: false }));
        if (requestRef.current === controller) requestRef.current = null;
        if (!controller.signal.aborted) setInProgress(false);
      }
    }
  }

  return { turns, readiness, inProgress, forgotten, prepare, ask, recordFeedback };
}
