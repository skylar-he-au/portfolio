import { clearToken, createSession, getToken, HttpError, responseError, VerificationError } from './session';
import { parseSse } from './sse';
import type { SseEvent } from './sse';

export const MATCH_MAX_CHARS = 20_000;
export const MATCH_DROPPED_MESSAGE = 'The connection dropped — please try again.';
export const MATCH_MESSAGES = {
  fetch_failed: "Couldn't read that link — please paste the job ad text.",
  fetch_limit: 'Too many links in a row — please paste the job ad text.',
  not_a_job: "This doesn't look like a job ad — please paste the job description.",
  too_long: "That's longer than 20,000 characters — please paste the main part.",
  unavailable: "The job match isn't available right now — please try again later.",
  timeout: 'That took too long — please try again.',
  limit_day: "Today's AI limit is reached — please email me",
  paused_month: 'The AI is paused until next month — please email me',
  internal: 'Something went wrong — please try again.',
} as const;

export interface MatchSection { url: string; title: string }
export interface MatchRequirement { id: string; text: string }
export interface MatchFinding {
  requirement_id: string;
  status?: 'found' | 'not_found' | 'unchecked';
  sections: MatchSection[];
  provider?: string | null;
  ms?: number;
}
export interface MatchNotes {
  requirements: MatchRequirement[];
  findings: MatchFinding[];
  truncated: boolean;
}
export interface MatchItem {
  requirement_id: string;
  requirement: string;
  text: string;
  sections: MatchSection[];
}
export interface MatchResult {
  match_id?: string;
  items: MatchItem[];
  not_covered: string[];
  unchecked: number;
  top8: boolean;
  language?: string;
}
type Stage = 'fetching' | 'reading' | 'matching' | 'writing' | 'checking';
export type MatchEvent =
  | { event: 'stage'; data: { stage: Stage; count?: number } }
  | { event: 'notes'; data: MatchNotes }
  | { event: 'result'; data: MatchResult }
  | { event: 'error'; data: { code?: string; message?: string } }
  | { event: 'done'; data: Record<string, never> };

export function matchInputLength(text: string): number {
  return [...text].length;
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function string(value: unknown): string | undefined {
  return typeof value === 'string' ? value.slice(0, 500) : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function list<T>(value: unknown, limit: number, parse: (item: unknown) => T | undefined): T[] {
  if (!Array.isArray(value)) return [];
  return value.slice(0, limit).flatMap((item: unknown) => {
    const parsed = parse(item);
    return parsed === undefined ? [] : [parsed];
  });
}

function sections(value: unknown): MatchSection[] {
  return list(value, 8, (item) => {
    const data = record(item);
    const url = string(data.url);
    const title = string(data.title);
    return url !== undefined && title !== undefined ? { url, title } : undefined;
  });
}

export function parseMatchEvent(event: SseEvent): MatchEvent | null {
  const data = record(event.data);
  switch (event.event) {
    case 'stage': {
      const stage = data.stage;
      if (stage !== 'fetching' && stage !== 'reading' && stage !== 'matching' && stage !== 'writing' && stage !== 'checking') return null;
      const count = number(data.count);
      return { event: 'stage', data: { stage, ...(count === undefined ? {} : { count }) } };
    }
    case 'notes': return { event: 'notes', data: {
      requirements: list(data.requirements, 16, (item) => {
        const value = record(item);
        const id = string(value.id);
        const text = string(value.text);
        return id !== undefined && text !== undefined ? { id, text } : undefined;
      }),
      findings: list(data.findings, 16, (item): MatchFinding | undefined => {
        const value = record(item);
        const requirement_id = string(value.requirement_id);
        if (requirement_id === undefined) return;
        const finding: MatchFinding = { requirement_id, sections: sections(value.sections) };
        if (value.status === 'found' || value.status === 'not_found' || value.status === 'unchecked') finding.status = value.status;
        if (value.provider === null) finding.provider = null;
        else if (typeof value.provider === 'string') finding.provider = string(value.provider);
        const ms = number(value.ms);
        if (ms !== undefined && ms >= 0) finding.ms = ms;
        return finding;
      }),
      truncated: data.truncated === true,
    } };
    case 'result': {
      const result: MatchResult = {
        items: list(data.items, 16, (item) => {
          const value = record(item);
          const requirement_id = string(value.requirement_id);
          const requirement = string(value.requirement);
          const text = string(value.text);
          return requirement_id !== undefined && requirement !== undefined && text !== undefined
            ? { requirement_id, requirement, text, sections: sections(value.sections) } : undefined;
        }),
        not_covered: list(data.not_covered, 16, (item) => string(item)),
        unchecked: Math.min(Number.MAX_SAFE_INTEGER, Math.max(0, Math.floor(number(data.unchecked) ?? 0))),
        top8: data.top8 === true,
      };
      if (typeof data.match_id === 'string') result.match_id = string(data.match_id);
      if (typeof data.language === 'string') result.language = string(data.language);
      return { event: 'result', data: result };
    }
    case 'error': return { event: 'error', data: { code: string(data.code), message: string(data.message) } };
    case 'done': return { event: 'done', data: {} };
    default: return null;
  }
}

export function matchErrorMessage(code?: string, message?: string): string {
  if (message?.trim()) return message;
  return code && Object.prototype.hasOwnProperty.call(MATCH_MESSAGES, code)
    ? MATCH_MESSAGES[code as keyof typeof MATCH_MESSAGES] : MATCH_MESSAGES.internal;
}

export function matchHttpMessage(error: HttpError): string {
  if (error.status === 422) return MATCH_MESSAGES.too_long;
  if (error.code === 'unavailable') return MATCH_MESSAGES.unavailable;
  return error.message;
}

export async function runMatch(input: string, { signal, onEvent }: {
  signal: AbortSignal;
  onEvent: (event: MatchEvent) => void;
}): Promise<void> {
  async function sessionToken(reset = false): Promise<string> {
    try {
      return await createSession(signal, reset);
    } catch (error) {
      signal.throwIfAborted();
      if (error instanceof HttpError) {
        if (error.status >= 200 && error.status < 300) throw new VerificationError();
        throw error;
      }
      throw new VerificationError();
    }
  }

  signal.throwIfAborted();
  let token = (import.meta.env.PUBLIC_TURNSTILE_SITEKEY && getToken()) || await sessionToken();
  let response: Response;
  for (let attempt = 0; ; attempt++) {
    signal.throwIfAborted();
    response = await fetch('/api/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ input }),
      signal,
    });
    signal.throwIfAborted();
    if (response.ok) break;
    const error = await responseError(response);
    signal.throwIfAborted();
    if (error.status === 401 && error.code === 'auth') {
      clearToken();
      if (attempt === 0) {
        token = await sessionToken(true);
        continue;
      }
    }
    throw error;
  }
  if (!response.body) throw new Error(MATCH_DROPPED_MESSAGE);
  for await (const raw of parseSse(response.body)) {
    signal.throwIfAborted();
    const event = parseMatchEvent(raw);
    if (!event) continue;
    onEvent(event);
    if (event.event === 'done' || event.event === 'error') return;
  }
  throw new Error(MATCH_DROPPED_MESSAGE);
}

export async function sendMatchFeedback(matchId: string, signal?: AbortSignal): Promise<boolean> {
  try {
    const token = getToken();
    if (token === null) return false;
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ match_id: matchId }),
      signal,
    });
    return response.ok;
  } catch {
    return false;
  }
}
