import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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
