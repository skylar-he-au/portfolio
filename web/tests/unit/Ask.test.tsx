import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';
import HowItAnswered from '../../src/components/HowItAnswered';
import type { AnswerTrace } from '../../src/components/HowItAnswered';

let Ask: typeof import('../../src/components/Ask').default;
let widget: ReturnType<typeof mockTurnstile>;

const examples = ['First question?', 'Second question?', 'Third question?'];
const unavailable = 'The assistant is unavailable right now — please try again later or email me.';

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
    fail() { controller.error(new Error('Network failure')); },
  };
}

function eventResponse(events: [string, unknown][]) {
  const stream = controlledStream();
  for (const [name, data] of events) stream.emit(name, data);
  stream.close();
  return response(200, {}, stream.body);
}

type Endpoint = () => Promise<ReturnType<typeof response>>;

function mockEndpoints({
  ping = async () => response(),
  health = async () => response(200, { db: 'ok', pgvector: '0.8.0' }),
  ask = async () => eventResponse([['done', {}]]),
  session = async () => response(200, { token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() }),
}: { ping?: Endpoint; health?: Endpoint; ask?: Endpoint; session?: Endpoint } = {}) {
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    if (input === '/api/ping') return ping();
    if (input === '/api/health') return health();
    if (input === '/api/ask') return ask();
    if (input === '/api/session') return session();
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function renderReady(ask?: Endpoint) {
  const fetchMock = mockEndpoints({ ask });
  const rendered = render(<Ask examples={examples} />);
  await waitFor(() => expect(screen.queryByText('getting ready…')).not.toBeInTheDocument());
  return { ...rendered, fetchMock };
}

function draft(text: string) {
  fireEvent.change(screen.getByRole('textbox', { name: 'Ask about my projects' }), { target: { value: text } });
}

async function send(text = 'My question?') {
  draft(text);
  const button = screen.getByRole('button', { name: 'Send' });
  await waitFor(() => expect(button).toBeEnabled());
  await act(async () => { expect(fireEvent.submit(button.closest('form')!)).toBe(false); });
}

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  vi.stubEnv('DEV', true);
  vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
  widget = mockTurnstile();
  Ask = (await import('../../src/components/Ask')).default;
});

afterEach(() => {
  cleanup();
  for (const script of turnstileScripts()) script.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('Ask sessions and Turnstile', () => {
  it('mounts without third-party requests or scripts, and keeps ping unauthenticated', async () => {
    const { fetchMock } = await renderReady();
    expect(turnstileScripts()).toHaveLength(0);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual(['/api/ping', '/api/health']);
    expect(fetchMock.mock.calls[0][1]).not.toHaveProperty('headers');
    expect(widget.api.render).not.toHaveBeenCalled();
  });

  it('loads one script and renders one hidden widget on repeated focus', async () => {
    await renderReady();
    fireEvent.focus(screen.getByRole('textbox'));
    fireEvent.focus(screen.getByRole('textbox'));
    await waitFor(() => expect(widget.api.render).toHaveBeenCalledOnce());
    expect(turnstileScripts()).toHaveLength(1);
    const [container, options] = widget.api.render.mock.calls[0];
    expect(container).not.toBeVisible();
    expect(options).toMatchObject({ sitekey: 'test-sitekey', execution: 'execute', action: 'ask', appearance: 'execute' });
    expect(widget.api.execute).not.toHaveBeenCalled();
  });

  it('loads the widget when an example is clicked', async () => {
    await renderReady();
    fireEvent.click(screen.getByRole('button', { name: examples[0] }));
    await waitFor(() => expect(widget.api.render).toHaveBeenCalledOnce());
    expect(turnstileScripts()).toHaveLength(1);
  });

  it('gets a session before asking, authorizes both questions, and reuses the token', async () => {
    const { fetchMock } = await renderReady();
    await send('First');
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    await send('Second');
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    const calls = fetchMock.mock.calls.filter(([url]) => url === '/api/session' || url === '/api/ask');
    expect(calls.map(([url]) => url)).toEqual(['/api/session', '/api/ask', '/api/ask']);
    expect(calls[0][1]).toMatchObject({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ turnstile: 'turnstile-token' }) });
    for (const call of calls.slice(1)) expect(call[1]?.headers).toMatchObject({ Authorization: 'Bearer session-token' });
    expect(widget.api.execute).toHaveBeenCalledOnce();
  });

  it.each([false, true])('refreshes after auth once; repeated auth = %s', async (repeat) => {
    sessionStorage.setItem('session', JSON.stringify({ token: 'old-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() }));
    const remove = vi.spyOn(Storage.prototype, 'removeItem');
    const ask = vi.fn<Endpoint>().mockResolvedValueOnce(response(401, { code: 'auth' }))
      .mockImplementation(async () => repeat ? response(401, { code: 'auth' }) : eventResponse([['answer', { text: 'Refreshed', citations: [] }], ['done', {}]]));
    const { fetchMock } = await renderReady(ask);
    await send('Same question');
    if (repeat) expect(await screen.findByRole('alert')).toHaveTextContent('Please try again.');
    else await screen.findByText('Refreshed');
    expect(ask).toHaveBeenCalledTimes(2);
    expect(remove).toHaveBeenCalledWith('session');
    expect(widget.api.reset).toHaveBeenCalledOnce();
    expect(widget.api.execute).toHaveBeenCalledOnce();
    expect(widget.api.reset.mock.invocationCallOrder[0]).toBeLessThan(widget.api.execute.mock.invocationCallOrder[0]);
    const calls = fetchMock.mock.calls.filter(([url]) => url === '/api/ask');
    expect(calls[0][1]?.headers).toMatchObject({ Authorization: 'Bearer old-token' });
    expect(calls[1][1]?.headers).toMatchObject({ Authorization: 'Bearer session-token' });
    expect(calls.map(([, init]) => JSON.parse(String(init?.body)))).toEqual([{ question: 'Same question' }, { question: 'Same question' }]);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
    if (repeat) expect((await import('../../src/lib/session')).getToken()).toBeNull();
  });

  it('refreshes a session created for the first question after auth rejection', async () => {
    const ask = vi.fn<Endpoint>().mockResolvedValueOnce(response(401, { code: 'auth' }))
      .mockResolvedValueOnce(eventResponse([['answer', { text: 'After refresh', citations: [] }], ['done', {}]]));
    const { fetchMock } = await renderReady(ask);
    await send();
    await screen.findByText('After refresh');
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session' || url === '/api/ask').map(([url]) => url))
      .toEqual(['/api/session', '/api/ask', '/api/session', '/api/ask']);
    expect(widget.api.execute).toHaveBeenCalledTimes(2);
    expect(widget.api.reset).toHaveBeenCalledOnce();
  });

  describe.each(['/api/ask', '/api/session'])('%s rejections', (endpoint) => {
    it.each([
      ['limit_day', 429, "Today's AI limit is reached — please email me"],
      ['paused_month', 503, 'The AI is paused until next month — please email me'],
      ['slow_down', 429, 'Too many requests — please wait a minute and try again.'],
      ['forbidden', 403, 'Something went wrong — please try again.'],
      ['unavailable', 503, unavailable],
      ['auth', 401, 'Please try again.'],
      ['unknown', 500, 'Something went wrong — please try again.'],
    ] as const)('maps %s HTTP %i to its fixed message', async (code, status, message) => {
      const reject = async () => response(status, { code, message: 'Untrusted server text' });
      const fetchMock = mockEndpoints(endpoint === '/api/ask' ? { ask: reject } : { session: reject });
      render(<Ask examples={examples} />);
      await send();
      expect(await screen.findByRole('alert')).toHaveTextContent(message);
      expect(screen.queryByText('Untrusted server text')).not.toBeInTheDocument();
      expect(screen.getByText('You asked:').parentElement).toHaveTextContent('My question?');
      if (endpoint === '/api/session') {
        expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
        expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ask')).toHaveLength(0);
      }
    });

    it('handles unparseable JSON with the generic message', async () => {
      const reject = async () => ({ ...response(503), json: async () => { throw new SyntaxError('Bad JSON'); } });
      mockEndpoints(endpoint === '/api/ask' ? { ask: reject } : { session: reject });
      render(<Ask examples={examples} />);
      await send();
      expect(await screen.findByRole('alert')).toHaveTextContent('Something went wrong — please try again.');
    });
  });

  it('stops if the refresh session itself replies auth', async () => {
    sessionStorage.setItem('session', JSON.stringify({ token: 'old', expires_at: new Date(Date.now() + 3_600_000).toISOString() }));
    const fetchMock = mockEndpoints({ ask: async () => response(401, { code: 'auth' }), session: async () => response(401, { code: 'auth' }) });
    render(<Ask examples={examples} />);
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent('Please try again.');
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ask')).toHaveLength(1);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
    expect(widget.api.execute).toHaveBeenCalledOnce();
  });

  it('asks and reuses memory when sessionStorage get/set throw', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('Blocked'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Blocked'); });
    const { fetchMock } = await renderReady();
    await send();
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    await send('Again');
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ask')).toHaveLength(2);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
  });

  it.each([undefined, ''])('handles missing sitekey %j without loading a script, and warns once', async (sitekey) => {
    vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', sitekey);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const { fetchMock } = await renderReady();
    fireEvent.focus(screen.getByRole('textbox'));
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(unavailable);
    await send('Again');
    await waitFor(() => expect(screen.getAllByRole('alert')).toHaveLength(2));
    expect(warn).toHaveBeenCalledOnce();
    expect(turnstileScripts()).toHaveLength(0);
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/session' || url === '/api/ask')).toBe(false);
  });

  it('replaces a session expiring in less than five minutes', async () => {
    sessionStorage.setItem('session', JSON.stringify({ token: 'expiring', expires_at: new Date(Date.now() + 299_000).toISOString() }));
    const { fetchMock } = await renderReady();
    await send();
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
    expect(fetchMock.mock.calls.find(([url]) => url === '/api/ask')?.[1]?.headers).toMatchObject({ Authorization: 'Bearer session-token' });
  });

  it('resets the used widget when a later question needs a new session', async () => {
    const { fetchMock } = await renderReady();
    await send();
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    const session = await import('../../src/lib/session');
    session.setToken({ token: 'expiring', expires_at: new Date(Date.now() + 299_000).toISOString() });
    await send('Next question');
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    expect(widget.api.execute).toHaveBeenCalledTimes(2);
    expect(widget.api.reset).toHaveBeenCalledOnce();
    expect(widget.api.reset.mock.invocationCallOrder[0]).toBeLessThan(widget.api.execute.mock.invocationCallOrder[1]);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(2);
  });

  it('waits for a loading widget before sending a session or question', async () => {
    widget.api.ready.mockImplementation(() => {});
    const { fetchMock } = await renderReady();
    fireEvent.focus(screen.getByRole('textbox'));
    await send();
    expect(widget.api.execute).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/session' || url === '/api/ask')).toBe(false);
    await act(async () => { widget.api.ready.mock.calls[0][0](); });
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === '/api/ask')).toBe(true));
    expect(widget.api.render).toHaveBeenCalledOnce();
  });

  it('keeps the Turnstile failure retry budget across an auth refresh', async () => {
    widget.api.execute.mockImplementationOnce(() => { widget.options()['error-callback'](); })
      .mockImplementationOnce(() => { widget.options().callback('proof'); })
      .mockImplementationOnce(() => { widget.options()['error-callback'](); });
    const { fetchMock } = await renderReady(async () => response(401, { code: 'auth' }));
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(unavailable);
    expect(widget.api.execute).toHaveBeenCalledTimes(3);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ask')).toHaveLength(1);
  });

  it('aborts a pending session request and ignores its late JSON on unmount', async () => {
    let resolveJson!: (body: unknown) => void;
    const session = async () => ({ ...response(), json: () => new Promise<unknown>((resolve) => { resolveJson = resolve; }) });
    const fetchMock = mockEndpoints({ session });
    const { unmount } = render(<Ask examples={examples} />);
    await send();
    const signal = fetchMock.mock.calls.find(([url]) => url === '/api/session')?.[1]?.signal;
    expect(signal?.aborted).toBe(false);
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { resolveJson({ token: 'late', expires_at: new Date(Date.now() + 3_600_000).toISOString() }); });
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/ask')).toBe(false);
    expect(sessionStorage.getItem('session')).toBeNull();
  });

  it.each([false, true])('retries a Turnstile failure once; repeated failure = %s', async (repeat) => {
    widget.api.execute.mockImplementationOnce(() => { widget.options()['error-callback'](); });
    if (repeat) widget.api.execute.mockImplementationOnce(() => { widget.options()['error-callback'](); });
    const { fetchMock } = await renderReady();
    await send();
    if (repeat) expect(await screen.findByRole('alert')).toHaveTextContent(unavailable);
    else await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
    expect(widget.api.execute).toHaveBeenCalledTimes(2);
    expect(widget.api.reset).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ask')).toHaveLength(repeat ? 0 : 1);
  });

  it('removes the widget on unmount and ignores a pending execution', async () => {
    widget.api.execute.mockImplementation(() => {});
    const { unmount, fetchMock } = await renderReady();
    await send();
    await waitFor(() => expect(widget.api.execute).toHaveBeenCalledOnce());
    unmount();
    expect(widget.api.remove).toHaveBeenCalledWith('widget-1');
    await act(async () => { widget.options().callback('late-token'); });
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/session' || url === '/api/ask')).toBe(false);
  });

  it('does not render a widget when readiness finishes after unmount', async () => {
    widget.api.ready.mockImplementation(() => {});
    const { unmount } = await renderReady();
    fireEvent.focus(screen.getByRole('textbox'));
    await waitFor(() => expect(widget.api.ready).toHaveBeenCalled());
    const ready = widget.api.ready.mock.calls[0][0];
    unmount();
    await act(async () => { ready(); });
    expect(widget.api.render).not.toHaveBeenCalled();
  });
});


describe('Ask readiness and retained behaviour', () => {
  it('keeps Send disabled until ping returns 200, with a labelled 500-character textarea', async () => {
    let resolvePing!: (value: ReturnType<typeof response>) => void;
    mockEndpoints({ ping: () => new Promise((resolve) => { resolvePing = resolve; }) });
    render(<Ask examples={examples} />);
    const textarea = screen.getByRole('textbox', { name: 'Ask about my projects' });
    expect(textarea).toHaveAttribute('maxlength', '500');
    expect(screen.getByRole('status')).toHaveTextContent('getting ready…');
    draft('A question');
    const sendButton = screen.getByRole('button', { name: 'Send' });
    expect(sendButton).toHaveAttribute('type', 'submit');
    expect(sendButton).toBeDisabled();
    await act(async () => { resolvePing(response()); });
    await waitFor(() => expect(sendButton).toBeEnabled());
    draft('  \n  ');
    expect(sendButton).toBeDisabled();
  });

  it.each([201, 422, 500])('keeps Send disabled when ping returns %i', async (status) => {
    mockEndpoints({ ping: async () => response(status) });
    render(<Ask examples={examples} />);
    draft('A question');
    expect(await screen.findByText('Assistant is offline')).toHaveAttribute('role', 'status');
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  });

  it('shows offline when ping rejects', async () => {
    mockEndpoints({ ping: async () => { throw new Error('Offline'); } });
    render(<Ask examples={examples} />);
    expect(await screen.findByText('Assistant is offline')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
  });

  it('fills the textarea from each of three example buttons without sending', async () => {
    const { fetchMock } = await renderReady();
    const group = screen.getByRole('group', { name: 'Example questions' });
    const buttons = within(group).getAllByRole('button');
    expect(buttons.map((button) => button.textContent)).toEqual(examples);
    for (const button of buttons) {
      expect(button).toHaveAttribute('type', 'button');
      fireEvent.click(button);
      expect(screen.getByRole('textbox')).toHaveValue(button.textContent);
    }
    expect(fetchMock.mock.calls.some(([input]) => input === '/api/ask')).toBe(false);
  });

  it('ignores empty submissions and includes the exact privacy sentence and link', async () => {
    const { fetchMock } = await renderReady();
    draft('   ');
    fireEvent.submit(screen.getByRole('button', { name: 'Send' }).closest('form')!);
    expect(fetchMock.mock.calls.some(([input]) => input === '/api/ask')).toBe(false);
    const link = screen.getByRole('link', { name: 'See Privacy.' });
    expect(link).toHaveAttribute('href', '/privacy');
    expect(link.parentElement).toHaveTextContent(/^Your question is processed by AI providers\. See Privacy\.$/);
  });

  it.each([
    [200, { db: 'ok', pgvector: '0.8.0' }, 'db: ok · pgvector 0.8.0'],
    [503, { db: 'ok', pgvector: null }, 'db: ok · pgvector missing'],
    [503, { db: 'down', pgvector: null }, 'db: down'],
    [200, { db: 'ok' }, 'db: down'],
    [200, { pgvector: null }, 'db: down'],
    [200, { pgvector: 8 }, 'db: down'],
    [503, {}, 'db: down'],
    [503, null, 'db: down'],
    [500, { db: 'ok', pgvector: '0.8.0' }, 'db: down'],
  ])('handles health %i with body %j as %s', async (status, body, message) => {
    mockEndpoints({ health: async () => response(status, body) });
    render(<Ask examples={examples} />);
    expect(await screen.findByText(message)).toBeInTheDocument();
  });

  it.each(['fetch', 'json'])('shows db: down when health %s rejects', async (failure) => {
    mockEndpoints({ health: async () => {
      if (failure === 'fetch') throw new Error('Network failure');
      return { ...response(503), json: async () => { throw new SyntaxError('Invalid JSON'); } };
    } });
    render(<Ask examples={examples} />);
    expect(await screen.findByText('db: down')).toBeInTheDocument();
  });

  it('omits the health request and dev line in production', async () => {
    vi.stubEnv('DEV', false);
    const { fetchMock } = await renderReady();
    expect(fetchMock).toHaveBeenCalledWith('/api/ping', expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(fetchMock.mock.calls.some(([input]) => input === '/api/health')).toBe(false);
    expect(screen.queryByText(/^db:/)).not.toBeInTheDocument();
  });

  it('aborts initial requests on unmount and ignores late responses', async () => {
    const requests: { signal: AbortSignal; resolve: (value: ReturnType<typeof response>) => void }[] = [];
    vi.stubGlobal('fetch', vi.fn((_input: RequestInfo | URL, init?: RequestInit) => (
      new Promise<ReturnType<typeof response>>((resolve) => {
        requests.push({ signal: init!.signal!, resolve });
      })
    )));
    const { unmount, container } = render(<Ask examples={examples} />);
    expect(requests).toHaveLength(2);
    expect(requests.every(({ signal }) => !signal.aborted)).toBe(true);
    unmount();
    expect(requests.every(({ signal }) => signal.aborted)).toBe(true);
    const json = vi.fn(async () => ({ pgvector: '0.8.0' }));
    await act(async () => {
      for (const { resolve } of requests) resolve({ ...response(), json });
    });
    expect(json).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it('ignores health JSON that finishes after unmount', async () => {
    let resolveJson!: (body: unknown) => void;
    const json = vi.fn(() => new Promise<unknown>((resolve) => { resolveJson = resolve; }));
    mockEndpoints({ health: async () => ({ ...response(), json }) });
    const { unmount, container } = render(<Ask examples={examples} />);
    await waitFor(() => expect(json).toHaveBeenCalledOnce());
    unmount();
    await act(async () => { resolveJson({ pgvector: '0.8.0' }); });
    expect(container).toBeEmptyDOMElement();
  });
});

describe('Ask streaming', () => {
  it('shows the About citation in a See line for the fixed private answer', async () => {
    await renderReady(async () => eventResponse([
      ['route', { private: true, off_topic: false, language: 'en' }],
      ['answer', {
        text: 'Please see the About page or email me.',
        citations: [{ n: 1, url: '/about', heading: 'About Skylar' }],
      }],
      ['done', {}],
    ]));
    await send();
    const link = await screen.findByRole('link', { name: 'About Skylar' });
    expect(link).toHaveAttribute('href', '/about');
    expect(link.closest('.ask-see')).toHaveTextContent(/^See: About Skylar$/);
    expect(screen.getByLabelText('Answer')).toHaveTextContent('Please see the About page or email me.');
  });

  it('does not repeat a citation already linked by its marker in a See line', async () => {
    await renderReady(async () => eventResponse([
      ['answer', {
        text: 'Read about RoleRadar [1].',
        citations: [{ n: 1, url: '/projects/roleradar', heading: 'RoleRadar' }],
      }],
      ['done', {}],
    ]));
    await send();
    expect(await screen.findByRole('link', { name: '[1]' })).toHaveAttribute('href', '/projects/roleradar');
    expect(screen.queryByText(/^See:/)).not.toBeInTheDocument();
    expect(document.querySelector('.ask-see')).toBeNull();
  });

  it('skips an unsafe uncited citation without showing a See line', async () => {
    await renderReady(async () => eventResponse([
      ['answer', {
        text: 'An answer without markers.',
        citations: [{ n: 1, url: '//example.com', heading: 'Unsafe heading' }],
      }],
      ['done', {}],
    ]));
    await send();
    await screen.findByText('An answer without markers.');
    expect(screen.queryByRole('link', { name: 'Unsafe heading' })).not.toBeInTheDocument();
    expect(screen.queryByText(/^See:/)).not.toBeInTheDocument();
    expect(document.querySelector('.ask-see')).toBeNull();
  });

  it('lists only the uncited citation when an answer also has an inline citation', async () => {
    await renderReady(async () => eventResponse([
      ['answer', {
        text: 'Read about RoleRadar [1].',
        citations: [
          { n: 1, url: '/projects/roleradar', heading: 'RoleRadar' },
          { n: 2, url: '/about', heading: 'About Skylar' },
        ],
      }],
      ['done', {}],
    ]));
    await send();
    const link = await screen.findByRole('link', { name: 'About Skylar' });
    expect(link).toHaveAttribute('href', '/about');
    const see = link.closest<HTMLElement>('.ask-see')!;
    expect(see).toHaveTextContent(/^See: About Skylar$/);
    expect(within(see).getAllByRole('link')).toHaveLength(1);
    expect(screen.getByRole('link', { name: '[1]' })).toHaveAttribute('href', '/projects/roleradar');
  });

  it('shows tokens, then links final citations and marks unsupported text after checking', async () => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send('  Explain RoleRadar?  ');
    expect(screen.getByRole('textbox')).toHaveValue('');
    draft('Next question');
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
    await act(async () => {
      stream.emit('route', { private: false, off_topic: false, language: 'en' });
      stream.emit('status', { node: 'answer', state: 'start' });
      stream.emit('token', { text: 'Streaming ' });
    });
    expect(screen.getByLabelText('Answer')).toHaveTextContent('Streaming');
    expect(screen.getByText('checking…')).toBeInTheDocument();
    await act(async () => { stream.emit('token', { text: 'answer.' }); });
    expect(screen.getByLabelText('Answer')).toHaveTextContent('Streaming answer.');
    await act(async () => {
      stream.emit('answer', {
        text: 'Supported [1]. Unsupported sentence. <script>alert(1)</script> [2] [9]',
        citations: [
          { n: 1, url: '/projects/roleradar#x', heading: 'Architecture' },
          { n: 2, url: 'javascript:alert(1)', heading: 'Unsafe' },
        ],
      });
    });
    expect(screen.getByText('checking…')).toBeInTheDocument();
    const link = screen.getByRole('link', { name: '[1]' });
    expect(link).toHaveAttribute('href', '/projects/roleradar#x');
    expect(link).toHaveAttribute('title', 'Architecture');
    expect(link).not.toHaveAttribute('target');
    expect(screen.queryByRole('link', { name: '[2]' })).not.toBeInTheDocument();
    expect(screen.queryByRole('link', { name: '[9]' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Answer')).toHaveTextContent('<script>alert(1)</script>');
    expect(screen.getByLabelText('Answer').querySelector('script')).toBeNull();
    await act(async () => {
      stream.emit('check', { passed: false, unsupported: ['Unsupported sentence.'], retried: true });
    });
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    const mark = screen.getByText('Unsupported sentence.');
    expect(mark.tagName).toBe('MARK');
    expect(mark).toHaveClass('ask-unsupported');
    expect(mark).toHaveAttribute('title', 'Not found in the project pages');
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled();
    await act(async () => { stream.emit('done', {}); });
    await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled());
  });

  it('clears an earlier answer, citations and check marks on reset and streams the retry', async () => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    await act(async () => {
      stream.emit('answer', { text: 'Old claim [1]', citations: [{ n: 1, url: '/old', heading: 'Old' }] });
      stream.emit('check', { passed: false, unsupported: ['Old claim'], retried: false });
    });
    expect(screen.getByText('Old claim')).toHaveClass('ask-unsupported');
    await act(async () => { stream.emit('reset', { reason: 'retry' }); });
    expect(screen.getByLabelText('Answer')).toHaveTextContent(/^checking…$/);
    expect(screen.queryByRole('link', { name: '[1]' })).not.toBeInTheDocument();
    expect(document.querySelector('.ask-unsupported')).toBeNull();
    await act(async () => { stream.emit('token', { text: 'New answer' }); });
    expect(screen.getByLabelText('Answer')).toHaveTextContent('New answer');
    await act(async () => { stream.emit('done', {}); });
    await waitFor(() => expect(screen.queryByText('checking…')).not.toBeInTheDocument());
  });

  it('hides private route tokens and shows only checking until the approved answer', async () => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    await act(async () => {
      stream.emit('route', { private: true, off_topic: false, language: 'en' });
      stream.emit('token', { text: 'Secret draft' });
    });
    expect(screen.getByLabelText('Answer')).toHaveTextContent(/^checking…$/);
    expect(screen.queryByText('Secret draft')).not.toBeInTheDocument();
    await act(async () => { stream.emit('check', { passed: true, unsupported: [], retried: false }); });
    await act(async () => { stream.emit('answer', { text: 'Approved answer', citations: [] }); });
    expect(screen.getByLabelText('Answer')).toHaveTextContent('Approved answer');
    await act(async () => { stream.emit('done', {}); });
  });

  it('removes checking on done for an off-topic answer with no check event', async () => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    draft('Next question');
    await act(async () => {
      stream.emit('route', { private: false, off_topic: true, language: 'en' });
      stream.emit('answer', { text: 'Please ask about my projects.', citations: [] });
    });
    expect(screen.getByText('checking…')).toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); });
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    await waitFor(() => expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled());
  });

  it.each(['end', 'reader rejection'])('keeps partial text incomplete when the stream drops via %s', async (ending) => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    draft('Retry question');
    await act(async () => { stream.emit('token', { text: 'Partial answer' }); });
    expect(screen.getByLabelText('Answer')).toHaveTextContent('Partial answer');
    await act(async () => { if (ending === 'end') stream.close(); else stream.fail(); });
    expect(await screen.findByRole('alert')).toHaveTextContent('The connection dropped — please try again.');
    expect(screen.getByLabelText('Answer')).toHaveTextContent('Partial answer');
    expect(screen.getByText('(incomplete)')).toBeInTheDocument();
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
  });

  it('shows a dropped connection and allows retry when fetch rejects', async () => {
    let rejectAsk!: (reason: Error) => void;
    await renderReady(() => new Promise((_resolve, reject) => { rejectAsk = reject; }));
    await send();
    draft('Retry question');
    await act(async () => { rejectAsk(new Error('Fetch failed')); });
    expect(await screen.findByRole('alert')).toHaveTextContent('The connection dropped — please try again.');
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
  });

  it('treats an empty successful stream as dropped', async () => {
    await renderReady(async () => eventResponse([]));
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent('The connection dropped — please try again.');
  });

  it('keeps conversation IDs in state, posts trimmed questions, and preserves turns oldest first', async () => {
    const storage = vi.spyOn(Storage.prototype, 'setItem');
    const ask = vi.fn<Endpoint>()
      .mockResolvedValueOnce(eventResponse([
        ['meta', { conversation_id: 'session-123', turn_id: 'turn-1' }],
        ['meta', { conversation_id: 'ignore-second-meta', turn_id: 'turn-1' }],
        ['answer', { text: 'First answer', citations: [] }], ['done', {}],
      ]))
      .mockResolvedValueOnce(eventResponse([
        ['answer', { text: 'Second answer', citations: [] }], ['done', {}],
      ]));
    const { fetchMock } = await renderReady(ask);
    await send('  First question  ');
    await screen.findByText('First answer');
    await send('Second question');
    await screen.findByText('Second answer');
    const calls = fetchMock.mock.calls.filter(([input]) => input === '/api/ask');
    expect(calls).toHaveLength(2);
    expect(calls[0][1]).toMatchObject({ method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer session-token' }, signal: expect.any(AbortSignal) });
    expect(JSON.parse(String(calls[0][1]?.body))).toEqual({ question: 'First question' });
    expect(JSON.parse(String(calls[1][1]?.body))).toEqual({ question: 'Second question', conversation_id: 'session-123' });
    expect(storage).toHaveBeenCalledOnce();
    expect(storage).toHaveBeenCalledWith('session', expect.any(String));
    expect(JSON.parse(String(storage.mock.calls[0][1]))).toEqual({ token: 'session-token', expires_at: expect.any(String) });
    const turns = within(screen.getByRole('list', { name: 'Conversation' })).getAllByRole('listitem');
    expect(turns).toHaveLength(2);
    expect(turns[0]).toHaveTextContent('You asked: First question');
    expect(turns[0]).toHaveTextContent('First answer');
    expect(turns[1]).toHaveTextContent('You asked: Second question');
    expect(turns[1]).toHaveTextContent('Second answer');
  });

  it('prevents duplicate submissions while streaming and aborts an active ask on unmount', async () => {
    let rejectAsk!: (reason: Error) => void;
    const { fetchMock, unmount, container } = await renderReady(() => new Promise((_resolve, reject) => { rejectAsk = reject; }));
    await send();
    draft('Second question');
    fireEvent.submit(screen.getByRole('button', { name: 'Send' }).closest('form')!);
    const calls = fetchMock.mock.calls.filter(([input]) => input === '/api/ask');
    expect(calls).toHaveLength(1);
    const signal = calls[0][1]?.signal;
    expect(signal?.aborted).toBe(false);
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { rejectAsk(new DOMException('Aborted', 'AbortError')); });
    expect(container).toBeEmptyDOMElement();
  });

  it('stores the trace on its turn for the How it answered panel', async () => {
    await renderReady(async () => eventResponse([
      ['trace', { nodes: [{ name: 'answer', provider: null, model: null, ms: 123 }], sections: [], memory: [] }],
      ['done', {}],
    ]));
    await send();
    expect(await screen.findByText('How it answered')).toBeInTheDocument();
  });
});

describe('Ask errors', () => {
  it.each([
    [422, 'Please ask a question of up to 500 characters.'],
    [503, 'Something went wrong — please try again.'],
  ])('handles HTTP %i and enables Send again', async (status, message) => {
    await renderReady(async () => response(status));
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(message);
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    draft('Next question');
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
  });

  it.each([
    ['unavailable', 'Ignored', 'The assistant is unavailable right now — please try again later or email me.'],
    ['timeout', 'Ignored', 'That took too long — please try again.'],
    ['busy', 'Ignored', 'Please wait for the current answer to finish.'],
    ['internal', 'A custom English sentence.', 'A custom English sentence.'],
    ['unknown', 'Another English sentence.', 'Another English sentence.'],
    ['internal', undefined, 'Something went wrong — please try again.'],
    ['unknown', '', 'Something went wrong — please try again.'],
  ])('handles error code %s with message %j and enables Send again', async (code, message, expected) => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    draft('Next question');
    await act(async () => { stream.emit('error', { code, message }); });
    expect(await screen.findByRole('alert')).toHaveTextContent(expected);
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Send' })).toBeEnabled();
    expect(screen.queryByText('(incomplete)')).not.toBeInTheDocument();
  });
});

describe('Ask check results', () => {
  it('shows checked only after a passed check and clears it on reset until another check', async () => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    expect(screen.getByText('checking…')).toBeInTheDocument();
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
    await act(async () => {
      stream.emit('answer', { text: 'Supported answer.', citations: [] });
      stream.emit('check', { passed: true, unsupported: [], retried: false, reason: null });
    });
    expect(screen.getByText('checked')).toHaveClass('ask-checked');
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    await act(async () => { stream.emit('reset', { reason: 'fallback' }); });
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
    expect(screen.getByText('checking…')).toBeInTheDocument();
    await act(async () => { stream.emit('answer', { text: 'New answer.', citations: [] }); });
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
    await act(async () => { stream.emit('check', { passed: true, reason: null }); });
    expect(screen.getByText('checked')).toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); });
    expect(screen.getByText('checked')).toBeInTheDocument();
  });

  it.each(['off-topic', 'not found', 'error'])('never shows checked for %s without a check event', async (kind) => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
    await act(async () => {
      if (kind === 'error') {
        stream.emit('error', { code: 'internal' });
      } else {
        stream.emit('route', { private: false, off_topic: kind === 'off-topic', language: 'en' });
        stream.emit('answer', {
          text: kind === 'off-topic'
            ? 'Please ask about my projects.' : "I couldn't find that in the project pages — please email me.",
          citations: [],
        });
        stream.emit('done', {});
      }
    });
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
  });

  it('shows the unsupported reason in How it answered and clears it on reset', async () => {
    const stream = controlledStream();
    await renderReady(async () => response(200, {}, stream.body));
    await send();
    await act(async () => {
      stream.emit('answer', { text: 'Unsupported answer.', citations: [] });
      stream.emit('check', { passed: false, unsupported: [], retried: true, reason: 'unsupported' });
      stream.emit('trace', { nodes: [], sections: [], memory: [] });
    });
    fireEvent.click(screen.getByText('How it answered'));
    const panel = screen.getByText('How it answered').closest('details')!;
    const heading = within(panel).getByRole('heading', { name: 'Check' });
    expect(heading.parentElement).toHaveClass('ask-panel-block');
    expect(within(panel).getByText("Some sentences weren't found in the project pages.").tagName).toBe('P');
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
    expect(screen.queryByText('checking…')).not.toBeInTheDocument();
    await act(async () => { stream.emit('reset', { reason: 'retry' }); });
    expect(within(panel).queryByRole('heading', { name: 'Check' })).not.toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); });
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
  });

  it.each([undefined, null, 42, 'unknown', 'toString'])('handles an old or unknown check reason %j without a reason block', async (reason) => {
    await renderReady(async () => eventResponse([
      ['answer', { text: 'An answer.', citations: [] }],
      ['check', { passed: false, unsupported: [], retried: false, reason }],
      ['trace', { nodes: [], sections: [], memory: [] }],
      ['done', {}],
    ]));
    await send();
    fireEvent.click(await screen.findByText('How it answered'));
    expect(screen.queryByRole('heading', { name: 'Check' })).not.toBeInTheDocument();
    expect(screen.queryByText('checked')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('HowItAnswered', () => {
  it.each([
    ['not_in_about', "The About page doesn't answer this."],
    ['inferred', 'The draft said more than the About page does.'],
  ])('explains a failed check with reason %s', (reason, sentence) => {
    render(<HowItAnswered
      trace={{ nodes: [], sections: [], memory: [] }}
      checkResult={{ passed: false, reason }}
    />);
    fireEvent.click(screen.getByText('How it answered'));
    const heading = screen.getByRole('heading', { name: 'Check' });
    expect(heading.parentElement).toHaveClass('ask-panel-block');
    expect(screen.getByText(sentence).tagName).toBe('P');
  });

  it('omits the reason block for a passed check', () => {
    render(<HowItAnswered
      trace={{ nodes: [], sections: [], memory: [] }}
      checkResult={{ passed: true, reason: 'unsupported' }}
    />);
    fireEvent.click(screen.getByText('How it answered'));
    expect(screen.queryByRole('heading', { name: 'Check' })).not.toBeInTheDocument();
  });

  it('renders nothing without a trace even when a failed check exists', () => {
    const { container } = render(<HowItAnswered checkResult={{ passed: false, reason: 'unsupported' }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing without a trace', () => {
    const { container } = render(<HowItAnswered />);
    expect(container).toBeEmptyDOMElement();
  });

  it('starts collapsed and lists every retry step, safe passages and recalled memory', () => {
    const trace: AnswerTrace = {
      nodes: [
        { name: 'recall', provider: null, model: null, ms: 1 },
        ...['retrieve', 'answer', 'check', 'retrieve', 'answer', 'check'].map((name) => (
          { name, provider: 'Provider', model: 'Model', ms: 123 }
        )),
      ],
      sections: [
        { url: '/projects/roleradar#x', heading: 'RoleRadar', score: 0.5 },
        { url: '//example.com', heading: 'Unsafe heading', score: 0.126 },
      ],
      memory: [{ turn_id: 'old', question: 'Earlier question?', score: 0.8 }],
    };
    const { container } = render(<HowItAnswered trace={trace} />);
    const details = container.querySelector('details')!;
    expect(details).not.toHaveAttribute('open');
    fireEvent.click(screen.getByText('How it answered'));
    expect(details).toHaveAttribute('open');
    const lists = within(details).getAllByRole('list');
    const steps = within(lists[0]).getAllByRole('listitem');
    expect(steps.map((step) => step.textContent)).toEqual([
      'recall · no model · 1 ms',
      ...['retrieve', 'answer', 'check', 'retrieve', 'answer', 'check'].map((name) => `${name} · Provider · Model · 123 ms`),
    ]);
    expect(screen.getByRole('link', { name: 'RoleRadar' })).toHaveAttribute('href', '/projects/roleradar#x');
    expect(screen.queryByRole('link', { name: 'Unsafe heading' })).not.toBeInTheDocument();
    expect(lists[1]).toHaveTextContent('RoleRadar · 0.500');
    expect(lists[1]).toHaveTextContent('Unsafe heading · 0.126');
    expect(lists[2]).toHaveTextContent('Earlier question? · 0.80');
  });

  it('shows three decimals for a small passage score and two for memory', () => {
    render(<HowItAnswered trace={{
      nodes: [],
      sections: [{ url: '/about', heading: 'About Skylar', score: 0.0164 }],
      memory: [{ turn_id: 'old', question: 'Earlier question?', score: 0.8 }],
    }} />);
    fireEvent.click(screen.getByText('How it answered'));
    expect(screen.getByRole('link', { name: 'About Skylar' }).closest('li')).toHaveTextContent(/^About Skylar · 0.016$/);
    expect(screen.getByText('Earlier question? · 0.80')).toBeInTheDocument();
  });

  it('shows None for empty passages and memory', () => {
    render(<HowItAnswered trace={{ nodes: [], sections: [], memory: [] }} />);
    fireEvent.click(screen.getByText('How it answered'));
    expect(screen.getAllByText('None')).toHaveLength(2);
  });
});

it('uses no literal colors and only existing CSS variables in both component sources and chat styles', () => {
  const tokens = readFileSync(join(process.cwd(), 'src/styles/tokens.css'), 'utf8');
  const defined = new Set(Array.from(tokens.matchAll(/^\s*(--[\w-]+)\s*:/gm), (match) => match[1]));
  const sources = [
    readFileSync(join(process.cwd(), 'src/components/Ask.tsx'), 'utf8'),
    readFileSync(join(process.cwd(), 'src/components/HowItAnswered.tsx'), 'utf8'),
    readFileSync(join(process.cwd(), 'src/styles/global.css'), 'utf8').split('/* Ask chat')[1],
  ];
  for (const source of sources) {
    expect(source).not.toMatch(/#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\w-])|\b(?:rgba?|hsl)\s*\(/gi);
    const referenced = Array.from(source.matchAll(/var\(\s*(--[\w-]+)/g), (match) => match[1]);
    expect(referenced.filter((name) => !defined.has(name))).toEqual([]);
  }
  expect(sources[0]).toContain('var(--space-0)');
});
