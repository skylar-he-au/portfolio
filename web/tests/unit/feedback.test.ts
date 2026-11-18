import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let sendFeedback: typeof import('../../src/lib/feedback').sendFeedback;
let session: typeof import('../../src/lib/session');

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  session = await import('../../src/lib/session');
  ({ sendFeedback } = await import('../../src/lib/feedback'));
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('sendFeedback', () => {
  it('returns false without a request when there is no token', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(session.getToken()).toBeNull();
    await expect(sendFeedback('turn-1')).resolves.toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([200, 201, 204, 299])('posts authenticated feedback and returns true for HTTP %i', async (status) => {
    session.setToken({ token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() });
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status }));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    await expect(sendFeedback('turn-1', controller.signal)).resolves.toBe(true);
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer session-token' },
      body: JSON.stringify({ turn_id: 'turn-1' }),
      signal: controller.signal,
    });
  });

  it.each([300, 401, 429, 500, 503])('returns false without retrying or clearing the token for HTTP %i', async (status) => {
    session.setToken({ token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() });
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status }));
    vi.stubGlobal('fetch', fetchMock);
    await expect(sendFeedback('turn-1')).resolves.toBe(false);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(session.getToken()).toBe('session-token');
  });

  it.each([new Error('Network failure'), new DOMException('Aborted', 'AbortError')])('returns false when fetch throws %s', async (error) => {
    session.setToken({ token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() });
    const fetchMock = vi.fn().mockRejectedValue(error);
    vi.stubGlobal('fetch', fetchMock);
    await expect(sendFeedback('turn-1')).resolves.toBe(false);
    expect(fetchMock).toHaveBeenCalledOnce();
  });
});
