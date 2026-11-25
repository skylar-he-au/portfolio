import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';

let session: typeof import('../../src/lib/session');
const valid = () => ({ token: 'stored-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() });

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  session = await import('../../src/lib/session');
});

afterEach(() => { vi.restoreAllMocks(); });

describe('session tokens', () => {
  it('mirrors the token in storage and restores it after a module reload', async () => {
    expect(session.getToken()).toBeNull();
    const token = valid();
    session.setToken(token);
    expect(session.getToken()).toBe(token.token);
    expect(JSON.parse(sessionStorage.getItem('session')!)).toEqual(token);
    vi.resetModules();
    expect((await import('../../src/lib/session')).getToken()).toBe(token.token);
  });

  it.each([299_000, -1])('rejects a token expiring in %i ms', (remaining) => {
    session.setToken({ token: 'expiring', expires_at: new Date(Date.now() + remaining).toISOString() });
    expect(session.getToken()).toBeNull();
  });

  it.each(['bad json', '{}', 'null', '{"token":"x","expires_at":"invalid"}', '{"token":42,"expires_at":"2099-01-01"}'])('ignores malformed storage %s', (value) => {
    sessionStorage.setItem('session', value);
    expect(session.getToken()).toBeNull();
  });

  it('clears both memory and storage', () => {
    session.setToken(valid());
    session.clearToken();
    expect(session.getToken()).toBeNull();
    expect(sessionStorage.getItem('session')).toBeNull();
  });

  it('supports memory only, including clearing, when every storage operation throws', () => {
    for (const method of ['getItem', 'setItem', 'removeItem'] as const) {
      vi.spyOn(Storage.prototype, method).mockImplementation(() => { throw new Error('Blocked'); });
    }
    expect(session.getToken()).toBeNull();
    session.setToken(valid());
    expect(session.getToken()).toBe('stored-token');
    session.clearToken();
    expect(session.getToken()).toBeNull();
  });
});

describe('shared session creation', () => {
  let widget: ReturnType<typeof mockTurnstile>;
  let releases: (() => void)[];

  beforeEach(() => {
    vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
    widget = mockTurnstile();
    releases = [session.acquireWidget(), session.acquireWidget()];
  });

  afterEach(() => {
    for (const release of releases) release();
    for (const script of turnstileScripts()) script.remove();
    vi.unstubAllGlobals();
    vi.unstubAllEnvs();
  });

  it('deduplicates verification and the pending session fetch, isolating caller cancellation', async () => {
    let resolveResponse!: (value: Response) => void;
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => new Promise<Response>((resolve) => { resolveResponse = resolve; }));
    vi.stubGlobal('fetch', fetchMock);
    const first = new AbortController();
    const second = new AbortController();
    const firstResult = expect(session.createSession(first.signal)).rejects.toMatchObject({ name: 'AbortError' });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    const other = session.createSession(second.signal);
    first.abort();
    await firstResult;
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(false);
    resolveResponse(new Response(JSON.stringify(valid()), { status: 200 }));
    await expect(other).resolves.toBe('stored-token');
    expect(widget.api.execute).toHaveBeenCalledOnce();
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(session.getToken()).toBe('stored-token');
  });

  it('aborts the session fetch after the last waiter cancels, ignoring late JSON', async () => {
    let resolveResponse!: (value: Response) => void;
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => new Promise<Response>((resolve) => { resolveResponse = resolve; }));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    const rejected = expect(session.createSession(controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    controller.abort();
    await rejected;
    expect(fetchMock.mock.calls[0][1]?.signal?.aborted).toBe(true);
    resolveResponse(new Response(JSON.stringify(valid()), { status: 200 }));
    await Promise.resolve();
    await Promise.resolve();
    expect(session.getToken()).toBeNull();
  });

  it('replaces a widget whose script failed to load, so a session works once back online', async () => {
    widget.holdLoad();
    session.prepareWidget();
    turnstileScripts()[0]!.dispatchEvent(new Event('error'));
    const signal = new AbortController().signal;
    await expect(session.createSession(signal)).rejects.toBeInstanceOf(session.VerificationError);
    expect(widget.api.render).not.toHaveBeenCalled();
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(valid()), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const created = session.createSession(signal);
    await vi.waitFor(() => expect(turnstileScripts()).toHaveLength(1));
    widget.load();
    await expect(created).resolves.toBe('stored-token');
    expect(widget.api.render).toHaveBeenCalledOnce();
    expect(document.querySelectorAll('body > div[hidden]')).toHaveLength(1);
  });

  it('releases idempotently and creates a fresh hidden widget after the last release', async () => {
    session.prepareWidget();
    await vi.waitFor(() => expect(widget.api.render).toHaveBeenCalledOnce());
    const container = widget.api.render.mock.calls[0][0];
    releases[0]();
    releases[0]();
    expect(widget.api.remove).not.toHaveBeenCalled();
    releases[1]();
    expect(widget.api.remove).toHaveBeenCalledOnce();
    expect(container).not.toBeInTheDocument();
    releases.push(session.acquireWidget());
    session.prepareWidget();
    await vi.waitFor(() => expect(widget.api.render).toHaveBeenCalledTimes(2));
    expect(widget.api.render.mock.calls[1][0]).not.toBe(container);
    expect(turnstileScripts()).toHaveLength(1);
  });
});
