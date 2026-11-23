import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MATCH_MESSAGES } from '../../src/lib/match';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';

let useMatch: typeof import('../../src/hooks/useMatch').useMatch;

function response(status = 200, body: unknown = {}, stream: ReadableStream<Uint8Array> | null = null) {
  return { status, ok: status >= 200 && status < 300, body: stream, json: async (): Promise<unknown> => body };
}

function controlledStream() {
  let controller!: ReadableStreamDefaultController<Uint8Array>;
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({ start(value) { controller = value; } });
  return {
    body,
    emit(event: string, data: unknown) {
      controller.enqueue(encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`));
    },
    close() { controller.close(); },
  };
}

function mockEndpoints(stream: ReadableStream<Uint8Array>) {
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    if (input === '/api/match') return Promise.resolve(response(200, {}, stream));
    if (input === '/api/feedback') return Promise.resolve(response());
    if (input === '/api/session') return Promise.resolve(response(200, {
      token: 'token', expires_at: new Date(Date.now() + 3_600_000).toISOString(),
    }));
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
  mockTurnstile();
  useMatch = (await import('../../src/hooks/useMatch')).useMatch;
});
afterEach(() => {
  cleanup();
  for (const script of turnstileScripts()) script.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('useMatch', () => {
  it.each([false, true])('tracks stage steps, with a fetching step only for links: %s', async (link) => {
    const stream = controlledStream();
    const fetchMock = mockEndpoints(stream.body);
    const storage = vi.spyOn(Storage.prototype, 'setItem');
    const { result } = renderHook(() => useMatch());
    act(() => result.current.prepare());
    expect(fetchMock).not.toHaveBeenCalled();
    let request!: ReturnType<typeof result.current.match>;
    act(() => {
      request = result.current.match(link ? 'https://boards.greenhouse.io/acme/jobs/123' : 'Build services');
      expect(result.current.match('Concurrent').started).toBe(false);
    });
    expect(request.started).toBe(true);
    const stages = link ? ['fetching', 'reading', 'matching', 'checking', 'writing'] : ['reading', 'matching', 'checking', 'writing'];
    const names = link ? ['Fetch job', 'Read job', 'Match', 'Check', 'Write'] : ['Read job', 'Match', 'Check', 'Write'];
    for (const [index, stage] of stages.entries()) {
      await act(async () => { stream.emit('stage', { stage, count: 2 }); });
      await waitFor(() => expect(result.current.stage).toEqual({ stage, count: 2 }));
      expect(result.current.steps).toEqual(names.slice(0, index + 1).map((name, step) => ({ name, state: step === index ? 'active' : 'done' })));
    }
    await act(async () => {
      stream.emit('result', { match_id: 'match-1', items: [], not_covered: [], unchecked: 0, top8: false });
      stream.emit('done', {}); stream.close(); await request.finished;
    });
    expect(result.current.steps).toEqual(names.map((name) => ({ name, state: 'done' })));
    expect(result.current.stage).toBeUndefined();
    expect(result.current.inProgress).toBe(false);
    expect(result.current.done).toBe(true);
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/ping')).toBe(false);
    await act(async () => { await result.current.recordFeedback(); });
    expect(result.current.feedback).toBe('sent');
    expect(storage.mock.calls.filter(([key]) => key === 'conversation')).toHaveLength(0);
  });

  it('rejects over 20,000 code points after resetting prior state', async () => {
    const stream = controlledStream();
    stream.emit('stage', { stage: 'reading' }); stream.emit('done', {}); stream.close();
    const fetchMock = mockEndpoints(stream.body);
    const { result } = renderHook(() => useMatch());
    expect(result.current.match('  ').started).toBe(false);
    await act(async () => { await result.current.match('Short').finished; });
    let request!: ReturnType<typeof result.current.match>;
    act(() => { request = result.current.match('😀'.repeat(20_001)); });
    expect(request.started).toBe(false);
    await request.finished;
    expect(result.current.error).toBe(MATCH_MESSAGES.too_long);
    expect(result.current.steps).toEqual([]);
    expect(result.current.stage).toBeUndefined();
    expect(result.current.done).toBe(false);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/match')).toHaveLength(1);
  });

  it('preserves steps on errors and resets them for the next request', async () => {
    const first = controlledStream();
    const second = controlledStream();
    const fetchMock = mockEndpoints(first.body);
    const { result } = renderHook(() => useMatch());
    let finished!: Promise<void>;
    act(() => { finished = result.current.match('First').finished; });
    await act(async () => {
      first.emit('stage', { stage: 'reading' }); first.emit('error', { code: 'not_a_job' }); first.close(); await finished;
    });
    expect(result.current.steps).toEqual([{ name: 'Read job', state: 'active' }]);
    expect(result.current.error).toBe(MATCH_MESSAGES.not_a_job);
    expect(result.current.stage).toBeUndefined();
    fetchMock.mockImplementation(async () => response(200, {}, second.body));
    act(() => { finished = result.current.match('Second').finished; });
    expect(result.current.steps).toEqual([]);
    expect(result.current.error).toBeUndefined();
    await act(async () => { second.emit('done', {}); second.close(); await finished; });
  });
});
