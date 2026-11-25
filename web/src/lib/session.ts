import { renderWidget } from './turnstile';
import type { WidgetController } from './turnstile';

export interface Session {
  token: string;
  expires_at: string;
}

let memory: Session | null = null;
let restored = false;

export class VerificationError extends Error {}

export class HttpError extends Error {
  constructor(readonly status: number, readonly code: unknown) {
    let message = 'Something went wrong — please try again.';
    switch (code) {
      case 'auth': message = 'Please try again.'; break;
      case 'slow_down': message = 'Too many requests — please wait a minute and try again.'; break;
      case 'limit_day': message = "Today's AI limit is reached — please email me"; break;
      case 'paused_month': message = 'The AI is paused until next month — please email me'; break;
    }
    super(message);
  }
}

export async function responseError(response: Response): Promise<HttpError> {
  let code: unknown;
  try {
    const body: unknown = await response.json();
    if (typeof body === 'object' && body !== null && 'code' in body) code = body.code;
  } catch {
    // Non-JSON rejections use the generic message.
  }
  return new HttpError(response.status, code);
}

interface RetryBudget { retries: number }
interface SessionFlight {
  controller: AbortController;
  promise: Promise<string>;
  waiters: Set<RetryBudget>;
  retries: number;
  settled: boolean;
}

let references = 0;
let container: HTMLDivElement | null = null;
let widget: WidgetController | null = null;
let widgetExecuted = false;
let warnedMissingSitekey = false;
let flight: SessionFlight | null = null;
const retryBudgets = new WeakMap<AbortSignal, RetryBudget>();

export function acquireWidget(): () => void {
  references++;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    if (--references !== 0) return;
    flight?.controller.abort();
    flight = null;
    widget?.remove();
    widget = null;
    widgetExecuted = false;
    container?.remove();
    container = null;
  };
}

function getWidget(): WidgetController {
  const sitekey = import.meta.env.PUBLIC_TURNSTILE_SITEKEY;
  if (!sitekey) {
    if (!warnedMissingSitekey) {
      console.warn('PUBLIC_TURNSTILE_SITEKEY is missing; the assistant is unavailable.');
      warnedMissingSitekey = true;
    }
    throw new VerificationError();
  }
  if (references === 0) throw new VerificationError();
  // api.js did not load (offline at the first focus): start again with a fresh widget, which
  // loads the script again, so asking works once the connection is back.
  if (widget?.failed) {
    widget.remove();
    widget = null;
    widgetExecuted = false;
    container?.remove();
    container = null;
  }
  if (!widget) {
    container = document.createElement('div');
    container.hidden = true;
    document.body.appendChild(container);
    widget = renderWidget(container, sitekey);
  }
  return widget;
}

export function prepareWidget(): void {
  if (import.meta.env.PUBLIC_TURNSTILE_SITEKEY) getWidget();
}

async function requestSession(current: SessionFlight, reset: boolean): Promise<string> {
  const { signal } = current.controller;
  let turnstile: string;
  try {
    const controller = getWidget();
    if (reset || widgetExecuted) controller.reset();
    while (true) {
      try {
        signal.throwIfAborted();
        widgetExecuted = true;
        turnstile = await controller.execute();
        break;
      } catch (error) {
        if (signal.aborted || [...current.waiters].some((budget) => budget.retries >= 1)) throw error;
        current.retries++;
        for (const budget of current.waiters) budget.retries++;
        controller.reset();
      }
    }
  } catch {
    signal.throwIfAborted();
    throw new VerificationError();
  }
  signal.throwIfAborted();
  const response = await fetch('/api/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ turnstile }),
    signal,
  });
  signal.throwIfAborted();
  if (!response.ok) throw await responseError(response);
  const session: unknown = await response.json();
  signal.throwIfAborted();
  if (!isSession(session)) throw new HttpError(response.status, undefined);
  setToken(session);
  return session.token;
}

export function createSession(signal: AbortSignal, reset = false): Promise<string> {
  if (signal.aborted) return Promise.reject(signal.reason);
  let budget = retryBudgets.get(signal);
  if (!budget) {
    budget = { retries: 0 };
    retryBudgets.set(signal, budget);
  }
  if (!flight) {
    const current: SessionFlight = {
      controller: new AbortController(), promise: Promise.resolve(''), waiters: new Set([budget]), retries: 0, settled: false,
    };
    flight = current;
    current.promise = requestSession(current, reset).finally(() => {
      current.settled = true;
      if (flight === current) flight = null;
    });
  }
  const current = flight;
  budget.retries = Math.max(budget.retries, current.retries);
  current.waiters.add(budget);
  const callerBudget = budget;

  // Each box can cancel independently; cancel the shared work only after its last caller leaves.
  return new Promise<string>((resolve, reject) => {
    function finish() {
      signal.removeEventListener('abort', abort);
      current.waiters.delete(callerBudget);
    }
    function abort() {
      finish();
      if (!current.settled && current.waiters.size === 0) {
        current.controller.abort();
        if (flight === current) {
          flight = null;
          widget?.reset();
        }
      }
      reject(signal.reason);
    }
    signal.addEventListener('abort', abort, { once: true });
    void current.promise.then((token) => {
      finish();
      if (signal.aborted) reject(signal.reason);
      else resolve(token);
    }, (error: unknown) => { finish(); reject(error); });
  });
}

export function isSession(value: unknown): value is Session {
  return typeof value === 'object' && value !== null
    && 'token' in value && typeof value.token === 'string' && value.token.length > 0
    && 'expires_at' in value && typeof value.expires_at === 'string'
    && Number.isFinite(Date.parse(value.expires_at));
}

export function getToken(): string | null {
  if (!restored) {
    restored = true;
    try {
      const value: unknown = JSON.parse(sessionStorage.getItem('session') ?? 'null');
      if (isSession(value)) memory = value;
    } catch {
      // Storage may be blocked; memory remains usable.
    }
  }
  return memory && Date.parse(memory.expires_at) >= Date.now() + 5 * 60_000 ? memory.token : null;
}

export function setToken(session: Session): void {
  memory = session;
  restored = true;
  try {
    sessionStorage.setItem('session', JSON.stringify(session));
  } catch {
    // Keep the in-memory token when storage is blocked.
  }
}

export function clearToken(): void {
  memory = null;
  restored = true;
  try {
    sessionStorage.removeItem('session');
  } catch {
    // Clearing memory is sufficient when storage is blocked.
  }
}
