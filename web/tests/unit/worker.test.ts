// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import worker from '../../worker/index';

const env = { ORIGIN_URL: 'https://origin.example/', ORIGIN_SECRET: 'origin-secret' };
const upstreamFetch = vi.fn<typeof fetch>();

beforeEach(() => {
  upstreamFetch.mockReset();
  upstreamFetch.mockResolvedValue(new Response('ok'));
  vi.stubGlobal('fetch', upstreamFetch);
});
afterEach(() => { vi.unstubAllGlobals(); });

function forwarded() {
  const [target, options] = upstreamFetch.mock.calls[0]!;
  return { target, options: options!, headers: new Headers(options?.headers) };
}

async function expectUnavailable(response: Response) {
  expect(response.status).toBe(503);
  expect(await response.text()).toBe('{"error":"unavailable"}');
  expect(response.headers.get('Content-Type')).toBe('application/json');
  expect(response.headers.get('Cache-Control')).toBe('no-store');
  expect(response.headers.get('X-Content-Type-Options')).toBe('nosniff');
  expect([...response.headers.values()].join(' ')).not.toContain(env.ORIGIN_SECRET);
}

async function withinTimeout<T>(promise: Promise<T>): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error('Worker buffered the stream')), 500);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

describe('API Worker', () => {
  it('replaces forged trust headers with the Cloudflare IP and origin secret', async () => {
    const request = new Request('https://portfolio.example/api/session', {
      headers: {
        'x-client-ip': '192.0.2.9',
        'x-origin-secret': 'forged-secret',
        'cf-connecting-ip': '203.0.113.7',
        'x-forwarded-for': '192.0.2.10',
        'x-real-ip': '192.0.2.11',
      },
    });
    const response = await worker.fetch(request, env);
    const { headers } = forwarded();
    expect(headers.get('X-Client-IP')).toBe('203.0.113.7');
    expect(headers.get('X-Origin-Secret')).toBe(env.ORIGIN_SECRET);
    for (const name of ['CF-Connecting-IP', 'X-Forwarded-For', 'X-Real-IP']) {
      expect(headers.has(name)).toBe(false);
    }
    expect(headers.has('Host')).toBe(false);
    expect(request.headers.get('X-Client-IP')).toBe('192.0.2.9');
    expect(request.headers.get('CF-Connecting-IP')).toBe('203.0.113.7');
    expect(await response.text()).toBe('ok');
    expect([...response.headers.values()].join(' ')).not.toMatch(/203\.0\.113\.7|origin-secret/);
  });

  it.each([undefined, ''])('omits forged trust headers when IP and secret are absent (%j)', async (secret) => {
    await worker.fetch(new Request('https://portfolio.example/api/session', {
      headers: { 'X-Client-IP': 'forged-ip', 'X-Origin-Secret': 'forged-secret' },
    }), { ORIGIN_URL: env.ORIGIN_URL, ORIGIN_SECRET: secret });
    const { headers } = forwarded();
    expect(headers.has('X-Client-IP')).toBe(false);
    expect(headers.has('X-Origin-Secret')).toBe(false);
  });

  it('forwards patrol, auth, content type, method, path, raw query and the request stream', async () => {
    const body = '{"turnstile":"token"}';
    const request = new Request('https://portfolio.example/api/session?x=a%2Fb&x=two+words&empty=', {
      method: 'POST',
      headers: {
        'X-Patrol': 'signed-patrol',
        Authorization: 'Bearer session-token',
        'Content-Type': 'application/json',
      },
      body,
    });
    await worker.fetch(request, { ...env, ORIGIN_URL: 'https://origin.example///' });
    const { target, options, headers } = forwarded();
    expect(target).toBe('https://origin.example/api/session?x=a%2Fb&x=two+words&empty=');
    expect(options.method).toBe('POST');
    expect(options.body).toBe(request.body);
    expect(await new Response(options.body).text()).toBe(body);
    expect(headers.get('X-Patrol')).toBe('signed-patrol');
    expect(headers.get('Authorization')).toBe('Bearer session-token');
    expect(headers.get('Content-Type')).toBe('application/json');
    expect(options).toMatchObject({ redirect: 'manual', duplex: 'half' });
    expect(options).not.toHaveProperty('cache');
    expect(options).not.toHaveProperty('cf');
  });

  it.each(['GET', 'HEAD'])('forwards %s without a body', async (method) => {
    await worker.fetch(new Request('https://portfolio.example/api/status', { method }), env);
    expect(forwarded().options.method).toBe(method);
    expect(forwarded().options.body).toBeNull();
  });

  it('returns the upstream stream and first chunk before the second chunk is released', async () => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        controller.enqueue(encoder.encode('data: first\n\n'));
        await gate;
        controller.enqueue(encoder.encode('data: second\n\n'));
        controller.close();
      },
    });
    const upstream = new Response(stream, { headers: { 'Content-Type': 'text/event-stream' } });
    upstreamFetch.mockResolvedValue(upstream);
    try {
      const response = await withinTimeout(worker.fetch(new Request('https://portfolio.example/api/ask'), env));
      expect(response.body).toBe(upstream.body);
      expect(response.headers.get('Content-Type')).toBe('text/event-stream');
      const reader = response.body!.getReader();
      expect(await withinTimeout(reader.read())).toEqual({ done: false, value: encoder.encode('data: first\n\n') });
      release();
      expect(await withinTimeout(reader.read())).toEqual({ done: false, value: encoder.encode('data: second\n\n') });
      expect(await withinTimeout(reader.read())).toEqual({ done: true, value: undefined });
      reader.releaseLock();
    } finally {
      release();
    }
  });

  it.each([502, 503, 504])('replaces upstream %i with unavailable and cancels its body', async (status) => {
    const cancel = vi.fn();
    const body = new ReadableStream({ cancel });
    upstreamFetch.mockResolvedValue(new Response(body, { status, headers: { 'X-Upstream': 'discard' } }));
    const response = await worker.fetch(new Request('https://portfolio.example/api/ask'), env);
    await expectUnavailable(response);
    expect(cancel).toHaveBeenCalledOnce();
    expect(response.headers.has('X-Upstream')).toBe(false);
  });

  it('returns unavailable when fetch rejects', async () => {
    upstreamFetch.mockRejectedValue(new Error('network failure'));
    await expectUnavailable(await worker.fetch(new Request('https://portfolio.example/api/ask'), env));
  });

  it.each([undefined, '', 'not a URL', 'ftp://origin.example', 'https://'])('rejects invalid ORIGIN_URL %j', async (origin) => {
    const invalidEnv = { ORIGIN_URL: origin } as Parameters<typeof worker.fetch>[1];
    await expectUnavailable(await worker.fetch(new Request('https://portfolio.example/api/ask'), invalidEnv));
    expect(upstreamFetch).not.toHaveBeenCalled();
  });

  it.each([200, 302, 401, 422, 429, 500, 501])('passes through status %i, body and headers with nosniff', async (status) => {
    const upstream = new Response('data: answer\n\n', {
      status,
      statusText: 'Upstream status',
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Upstream': 'kept',
        Location: '/elsewhere',
        'X-Content-Type-Options': 'incorrect',
      },
    });
    upstreamFetch.mockResolvedValue(upstream);
    const response = await worker.fetch(new Request('https://portfolio.example/api/ask'), env);
    expect(response.status).toBe(status);
    expect(response.statusText).toBe('Upstream status');
    expect(response.body).toBe(upstream.body);
    expect(await response.text()).toBe('data: answer\n\n');
    expect(response.headers.get('Content-Type')).toBe('text/event-stream');
    expect(response.headers.get('Cache-Control')).toBe('no-cache');
    expect(response.headers.get('X-Upstream')).toBe('kept');
    expect(response.headers.get('Location')).toBe('/elsewhere');
    expect(response.headers.get('X-Content-Type-Options')).toBe('nosniff');
    expect(upstream.headers.get('X-Content-Type-Options')).toBe('incorrect');
  });
});
