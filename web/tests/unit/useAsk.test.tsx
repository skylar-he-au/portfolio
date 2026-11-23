import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';
import type { StoredConversation } from '../../src/lib/conversation';

let useAsk: typeof import('../../src/hooks/useAsk').useAsk;

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

function eventResponse(events: [string, unknown][]) {
  const stream = controlledStream();
  for (const [name, data] of events) stream.emit(name, data);
  stream.close();
  return response(200, {}, stream.body);
}

type Endpoint = () => Promise<ReturnType<typeof response>>;
function mockEndpoints(ask: Endpoint = async () => eventResponse([
  ['meta', { conversation_id: 'conversation-1', turn_id: 'turn-1' }],
  ['answer', { text: 'Answer', citations: [] }], ['done', {}],
])) {
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    if (input === '/api/ping' || input === '/api/feedback') return Promise.resolve(response());
    if (input === '/api/ask') return ask();
    if (input === '/api/session') return Promise.resolve(response(200, {
      token: 'token', expires_at: new Date(Date.now() + 3_600_000).toISOString(),
    }));
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function ready(persist = false) {
  const hook = renderHook(() => useAsk({ persist }));
  act(() => hook.result.current.prepare());
  await waitFor(() => expect(hook.result.current.readiness).toBe('ready'));
  return hook;
}

function stored(): StoredConversation | null {
  return JSON.parse(sessionStorage.getItem('conversation') ?? 'null') as StoredConversation | null;
}

const previous: StoredConversation = {
  v: 1, id: 'old', turns: [{
    turnId: 'old-turn', question: 'Earlier?', text: 'Earlier answer', citations: [], unsupported: [], feedback: 'idle',
  }],
};

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
  mockTurnstile();
  useAsk = (await import('../../src/hooks/useAsk')).useAsk;
});
afterEach(() => {
  cleanup();
  for (const script of turnstileScripts()) script.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('useAsk', () => {
  it('pings only on the first prepare', async () => {
    const fetchMock = mockEndpoints();
    const { result } = renderHook(() => useAsk());
    expect(result.current.readiness).toBe('idle');
    expect(fetchMock).not.toHaveBeenCalled();
    act(() => result.current.prepare());
    expect(result.current.readiness).toBe('pending');
    await waitFor(() => expect(result.current.readiness).toBe('ready'));
    act(() => result.current.prepare());
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ping')).toHaveLength(1);
    expect(fetchMock).toHaveBeenCalledWith('/api/ping', expect.objectContaining({ signal: expect.any(AbortSignal) }));
  });

  it('maps statuses in first-seen order and ignores recall and unknown nodes', async () => {
    mockEndpoints(async () => eventResponse([
      ['meta', { conversation_id: 'conversation-1', turn_id: 'turn-1' }],
      ['status', { node: 'recall', state: 'start' }], ['status', { node: 'recall', state: 'end' }],
      ['status', { node: 'unknown', state: 'start' }],
      ['status', { node: 'rewrite', state: 'start' }], ['status', { node: 'rewrite', state: 'end' }],
      ['status', { node: 'retrieve', state: 'start' }], ['status', { node: 'retrieve', state: 'end' }],
      ['status', { node: 'answer', state: 'start' }], ['token', { text: 'Draft' }],
      ['status', { node: 'answer', state: 'end' }],
      ['status', { node: 'check', state: 'start' }], ['status', { node: 'check', state: 'end' }],
      ['answer', { text: 'Answer', citations: [] }], ['check', { passed: true, unsupported: [] }], ['done', {}],
    ]));
    const { result } = await ready();
    await act(async () => { await result.current.ask('Question?').finished; });
    expect(result.current.turns[0]?.steps).toEqual(['Rewrite', 'Search', 'Answer', 'Check'].map((name) => ({ name, state: 'done' })));
    expect(result.current.turns[0]).toMatchObject({ text: 'Answer', finished: true, checkResult: { passed: true } });
  });

  it('keeps active steps across reset, reactivates repeated steps, and finishes them on done', async () => {
    const stream = controlledStream();
    mockEndpoints(async () => response(200, {}, stream.body));
    const { result } = await ready();
    let finished!: Promise<void>;
    act(() => { finished = result.current.ask('Question?').finished; });
    await act(async () => { stream.emit('status', { node: 'answer', state: 'start' }); });
    await waitFor(() => expect(result.current.turns[0]?.steps).toEqual([{ name: 'Answer', state: 'active' }]));
    await act(async () => {
      stream.emit('status', { node: 'answer', state: 'end' });
      stream.emit('reset', {});
    });
    await waitFor(() => expect(result.current.turns[0]?.steps[0]?.state).toBe('done'));
    await act(async () => { stream.emit('status', { node: 'answer', state: 'start' }); });
    await waitFor(() => expect(result.current.turns[0]?.steps).toEqual([{ name: 'Answer', state: 'active' }]));
    await act(async () => { stream.emit('done', {}); stream.close(); await finished; });
    expect(result.current.turns[0]?.steps[0]?.state).toBe('done');
  });

  it('guards empty, unready and concurrent requests synchronously', async () => {
    const stream = controlledStream();
    mockEndpoints(async () => response(200, {}, stream.body));
    const { result } = renderHook(() => useAsk());
    expect(result.current.ask('Question?').started).toBe(false);
    act(() => result.current.prepare());
    await waitFor(() => expect(result.current.readiness).toBe('ready'));
    expect(result.current.ask('  ').started).toBe(false);
    let first!: ReturnType<typeof result.current.ask>;
    act(() => {
      first = result.current.ask('  Question?  ');
      expect(result.current.ask('Again?').started).toBe(false);
    });
    expect(first.started).toBe(true);
    expect(result.current.turns[0]?.question).toBe('Question?');
    await act(async () => { stream.emit('done', {}); stream.close(); await first.finished; });
  });

  it('saves finished turns, restores them and persists successful feedback', async () => {
    const fetchMock = mockEndpoints();
    const first = await ready(true);
    await act(async () => { await first.result.current.ask('Question?').finished; });
    expect(stored()?.turns).toHaveLength(1);
    expect(stored()?.turns[0]).toMatchObject({ question: 'Question?', feedback: 'idle' });
    first.unmount();
    const second = renderHook(() => useAsk({ persist: true }));
    expect(second.result.current.turns[0]).toMatchObject({
      restored: true, finished: true, checking: false, incomplete: false, feedback: 'idle', steps: [], private: false,
    });
    await act(async () => { await second.result.current.recordFeedback(second.result.current.turns[0]!); });
    expect(fetchMock).toHaveBeenCalledWith('/api/feedback', expect.objectContaining({ body: JSON.stringify({ turn_id: 'turn-1' }) }));
    expect(stored()?.turns[0]?.feedback).toBe('sent');
    second.unmount();
    const third = renderHook(() => useAsk({ persist: true }));
    expect(third.result.current.turns[0]?.feedback).toBe('sent');
  });

  it.each(['error', 'generating', 'private', 'incomplete'])('does not save an %s turn or discard saved turns', async (kind) => {
    sessionStorage.setItem('conversation', JSON.stringify(previous));
    const stream = controlledStream();
    mockEndpoints(async () => response(200, {}, stream.body));
    const { result } = await ready(true);
    let finished!: Promise<void>;
    act(() => { finished = result.current.ask('Question?').finished; });
    await act(async () => {
      stream.emit('meta', { conversation_id: 'old', turn_id: 'new' });
      stream.emit('answer', { text: 'Answer', citations: [] });
      if (kind === 'private') stream.emit('route', { private: true });
    });
    await waitFor(() => expect(result.current.turns[1]?.turnId).toBe('new'));
    expect(stored()?.turns).toEqual(previous.turns);
    await act(async () => {
      if (kind === 'error') stream.emit('error', { code: 'busy' });
      else if (kind !== 'incomplete') stream.emit('done', {});
      stream.close();
      await finished;
    });
    if (kind !== 'generating') expect(stored()?.turns).toEqual(previous.turns);
    if (kind === 'error') expect(result.current.turns[1]?.error).toBe('Please wait for the current answer to finish.');
    if (kind === 'incomplete') expect(result.current.turns[1]?.incomplete).toBe(true);
    if (kind === 'private') expect(result.current.turns[1]?.private).toBe(true);
  });

  it('saves a checked private answer and flags private routes', async () => {
    mockEndpoints(async () => eventResponse([
      ['meta', { conversation_id: 'private' }], ['route', { private: true }],
      ['answer', { text: 'Checked private answer', citations: [] }],
      ['check', { passed: false, reason: 'inferred', unsupported: [] }], ['done', {}],
    ]));
    const { result } = await ready(true);
    await act(async () => { await result.current.ask('Private?').finished; });
    expect(result.current.turns[0]?.private).toBe(true);
    expect(stored()?.turns[0]).toMatchObject({ private: true, checkResult: { passed: false } });
  });

  it('adopts a new id at meta, keeps old turns and sends the current id next time', async () => {
    sessionStorage.setItem('conversation', JSON.stringify(previous));
    const fetchMock = mockEndpoints(async () => eventResponse([
      ['meta', { conversation_id: 'new' }], ['error', { code: 'busy' }],
    ]));
    const { result } = await ready(true);
    await act(async () => { await result.current.ask('Question?').finished; });
    expect(result.current.forgotten).toBe(true);
    expect(stored()).toEqual({ ...previous, id: 'new', forgotten: true });
    await act(async () => { await result.current.ask('Again?').finished; });
    const requests = fetchMock.mock.calls.filter(([url]) => url === '/api/ask');
    expect(JSON.parse(String(requests[0]?.[1]?.body)).conversation_id).toBe('old');
    expect(JSON.parse(String(requests[1]?.[1]?.body)).conversation_id).toBe('new');
    const restored = renderHook(() => useAsk({ persist: true }));
    expect(restored.result.current.forgotten).toBe(true);
  });

  it('never accesses the conversation key when persistence is off', async () => {
    sessionStorage.setItem('conversation', JSON.stringify(previous));
    const get = vi.spyOn(Storage.prototype, 'getItem');
    const set = vi.spyOn(Storage.prototype, 'setItem');
    mockEndpoints();
    const { result } = await ready();
    expect(result.current.turns).toEqual([]);
    await act(async () => { await result.current.ask('Question?').finished; });
    await act(async () => { await result.current.recordFeedback(result.current.turns[0]!); });
    expect(get).not.toHaveBeenCalledWith('conversation');
    expect(set.mock.calls.filter(([key]) => key === 'conversation')).toHaveLength(0);
  });
});
