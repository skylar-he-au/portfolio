export interface Session {
  token: string;
  expires_at: string;
}

let memory: Session | null = null;
let restored = false;

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
