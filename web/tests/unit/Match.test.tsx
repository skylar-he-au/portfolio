import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';

let Match: typeof import('../../src/components/Match').default;
let Ask: typeof import('../../src/components/Ask').default;
let widget: ReturnType<typeof mockTurnstile>;

const fallbacks = [
  ['fetch_failed', "Couldn't read that link — please paste the job ad text."],
  ['fetch_limit', 'Too many links in a row — please paste the job ad text.'],
  ['not_a_job', "This doesn't look like a job ad — please paste the job description."],
  ['too_long', "That's longer than 20,000 characters — please paste the main part."],
  ['unavailable', "The job match isn't available right now — please try again later."],
  ['timeout', 'That took too long — please try again.'],
  ['limit_day', "Today's AI limit is reached — please email me"],
  ['paused_month', 'The AI is paused until next month — please email me'],
  ['internal', 'Something went wrong — please try again.'],
] as const;
const dropped = 'The connection dropped — please try again.';
const section = { url: '/projects/roleradar#architecture', title: 'Architecture' };
const notes = {
  requirements: [{ id: 'r1', text: 'Build services' }, { id: 'r2', text: 'Operate services' }],
  findings: [{ requirement_id: 'r1', status: 'found', sections: [section], provider: null, ms: 12 }],
  truncated: true,
};
const result = {
  match_id: 'match-1',
  items: [{ requirement_id: 'r1', requirement: 'Original requirement', text: 'See how I built services.', sections: [section] }],
  not_covered: ['Kubernetes', 'Java'], unchecked: 1, top8: true, language: 'en',
};

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
function mockEndpoints(match: Endpoint = async () => eventResponse([['done', {}]])) {
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    if (input === '/api/match') return match();
    if (input === '/api/ask') return Promise.resolve(eventResponse([['done', {}]]));
    if (input === '/api/session') return Promise.resolve(response(200, {
      token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString(),
    }));
    if (input === '/api/ping' || input === '/api/health' || input === '/api/feedback') return Promise.resolve(response());
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

async function send(text = '  Build and operate services  ') {
  fireEvent.change(screen.getByRole('textbox', { name: 'Paste a job ad' }), { target: { value: text } });
  await act(async () => { fireEvent.submit(screen.getByRole('button', { name: 'Match' }).closest('form')!); });
}

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  vi.stubEnv('DEV', false);
  vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
  widget = mockTurnstile();
  Match = (await import('../../src/components/Match')).default;
  Ask = (await import('../../src/components/Ask')).default;
});

afterEach(() => {
  cleanup();
  for (const script of turnstileScripts()) script.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('shared widget across boxes', () => {
  it('shares one hidden container, script, execution and session, and removes the widget on last release', async () => {
    widget.api.execute.mockImplementation(() => {});
    const fetchMock = mockEndpoints();
    const ask = render(<Ask examples={[]} />);
    const match = render(<Match />);
    await waitFor(() => expect(screen.queryByText('getting ready…')).not.toBeInTheDocument());
    for (const input of screen.getAllByRole('textbox')) fireEvent.focus(input);
    await waitFor(() => expect(widget.api.render).toHaveBeenCalledOnce());
    const container = widget.api.render.mock.calls[0][0];
    expect(container).toHaveAttribute('hidden');
    expect(container.parentElement).toBe(document.body);
    expect(document.querySelectorAll('div[hidden]')).toHaveLength(1);
    expect(turnstileScripts()).toHaveLength(1);
    fireEvent.change(screen.getByRole('textbox', { name: 'Ask about my projects' }), { target: { value: 'Services?' } });
    fireEvent.submit(screen.getByRole('button', { name: 'Send' }).closest('form')!);
    await send();
    await waitFor(() => expect(widget.api.execute).toHaveBeenCalledOnce());
    await act(async () => { widget.options().callback('proof'); });
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === '/api/match')).toHaveLength(1));
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/ask')).toHaveLength(1);
    ask.unmount();
    expect(widget.api.remove).not.toHaveBeenCalled();
    match.unmount();
    expect(widget.api.remove).toHaveBeenCalledOnce();
    expect(container).not.toBeInTheDocument();
  });

  it('shares across a single page render and keeps a pending session alive after the initiating box unmounts', async () => {
    widget.api.execute.mockImplementation(() => {});
    const fetchMock = mockEndpoints();
    const boxes = render(<><Ask key="ask" examples={[]} /><Match key="match" /></>);
    await waitFor(() => expect(screen.queryByText('getting ready…')).not.toBeInTheDocument());
    fireEvent.change(screen.getByRole('textbox', { name: 'Ask about my projects' }), { target: { value: 'Services?' } });
    fireEvent.submit(screen.getByRole('button', { name: 'Send' }).closest('form')!);
    await send();
    await waitFor(() => expect(widget.api.execute).toHaveBeenCalledOnce());
    boxes.rerender(<><Match key="match" /></>);
    expect(widget.api.remove).not.toHaveBeenCalled();
    await act(async () => { widget.options().callback('proof'); });
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === '/api/match')).toHaveLength(1));
    boxes.unmount();
    expect(widget.api.remove).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/ask')).toBe(false);
  });

  it('renders both boxes together with one widget and one script', async () => {
    mockEndpoints();
    const { unmount } = render(<><Ask examples={[]} /><Match /></>);
    for (const input of screen.getAllByRole('textbox')) fireEvent.focus(input);
    await waitFor(() => expect(widget.api.render).toHaveBeenCalledOnce());
    expect(turnstileScripts()).toHaveLength(1);
    expect(document.querySelectorAll('div[hidden]')).toHaveLength(1);
    unmount();
    expect(widget.api.remove).toHaveBeenCalledOnce();
    expect(document.querySelectorAll('div[hidden]')).toHaveLength(0);
  });
});

describe('Match streaming and guide', () => {
  it('is lazy, has a six-row unlimited textarea, ignores empty input and needs no ping', async () => {
    const fetchMock = mockEndpoints();
    render(<Match />);
    const input = screen.getByRole('textbox', { name: 'Paste a job ad' });
    expect(input).toHaveAttribute('rows', '6');
    expect(input).not.toHaveAttribute('maxlength');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(turnstileScripts()).toHaveLength(0);
    await send('  \n ');
    expect(fetchMock).not.toHaveBeenCalled();
    const privacy = screen.getByRole('link', { name: 'See Privacy.' });
    expect(privacy).toHaveAttribute('href', '/privacy');
    expect(privacy.parentElement).toHaveTextContent(/^Your job ad is processed by AI providers\. See Privacy\.$/);
    fireEvent.focus(input);
    await waitFor(() => expect(widget.api.render).toHaveBeenCalledOnce());
    expect(widget.api.execute).not.toHaveBeenCalled();
  });

  it('shows ordered progress, live whole notes, then only the checked result and feedback after done', async () => {
    const stream = controlledStream();
    const fetchMock = mockEndpoints(async () => response(200, {}, stream.body));
    render(<Match />);
    await send();
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === '/api/match')).toBe(true));
    expect(screen.getByRole('button', { name: 'Match' })).toBeDisabled();
    await send('Duplicate');
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/match')).toHaveLength(1);
    const request = fetchMock.mock.calls.find(([url]) => url === '/api/match')![1];
    expect(request).toMatchObject({ method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer session-token' }, body: JSON.stringify({ input: 'Build and operate services' }) });
    for (const [stage, count, message] of [
      ['fetching', null, 'Fetching the job ad…'], ['reading', null, 'Reading the job ad…'],
      ['matching', null, 'Matching requirements to my project pages…'],
      ['matching', 6, 'Matching 6 requirements to my project pages…'],
      ['checking', null, 'Checking the sections…'], ['writing', null, 'Writing the guide…'],
    ] as const) {
      await act(async () => { stream.emit('stage', { stage, count }); });
      expect(screen.getByRole('status')).toHaveTextContent(message);
      expect(screen.queryByText('Where my pages cover this role:')).not.toBeInTheDocument();
    }
    await act(async () => { stream.emit('stage', { stage: 'invented' }); stream.emit('notes', notes); });
    expect(screen.getByRole('status')).toHaveTextContent('Writing the guide…');
    const panel = screen.getByText('How it matched').closest('details')!;
    expect(panel).not.toHaveAttribute('open');
    expect(panel).toHaveTextContent('r1');
    expect(panel).toHaveTextContent('found');
    expect(panel).toHaveTextContent('pending');
    expect(panel).toHaveTextContent('no model');
    expect(panel).toHaveTextContent('12 ms');
    expect(panel).toHaveTextContent('The job ad was cut to 20,000 characters');
    expect(panel.querySelector('a')).toHaveAttribute('href', section.url);
    await act(async () => { stream.emit('notes', { requirements: [{ id: 'r3', text: 'Replacement' }], findings: [] }); });
    expect(panel).toHaveTextContent('Replacement');
    expect(panel).not.toHaveTextContent('Build services');
    expect(panel).not.toHaveTextContent('12 ms');
    await act(async () => { stream.emit('result', result); });
    expect(screen.getByText('Where my pages cover this role:')).toBeInTheDocument();
    expect(screen.getByText(result.items[0].text)).toBeInTheDocument();
    expect(screen.queryByText(result.items[0].requirement)).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: section.title })).toHaveAttribute('href', section.url);
    const notCovered = screen.getByRole('list', { name: 'Not covered on the pages' });
    expect([...notCovered.querySelectorAll('li')].map((li) => li.textContent)).toEqual(['Kubernetes', 'Java']);
    expect(screen.getByText("1 requirement couldn't be checked")).toBeInTheDocument();
    expect(screen.getByText('Top 8 requirements of the job ad shown.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'This guide is wrong' })).not.toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); stream.close(); });
    const feedback = screen.getByRole('button', { name: 'This guide is wrong' });
    let resolveFeedback!: (value: ReturnType<typeof response>) => void;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { resolveFeedback = resolve; }));
    fireEvent.click(feedback);
    expect(feedback).toBeDisabled();
    expect(fetchMock.mock.calls.find(([url]) => url === '/api/feedback')?.[1]).toMatchObject({ body: JSON.stringify({ match_id: 'match-1' }), headers: { Authorization: 'Bearer session-token' } });
    await act(async () => { resolveFeedback(response(204)); });
    expect(screen.getByText('Thanks — noted')).toHaveAttribute('role', 'status');
  });

  it('shows plural unchecked alone for empty lists and skips items without sections', async () => {
    mockEndpoints(async () => eventResponse([
      ['result', { ...result, items: [{ ...result.items[0], sections: [] }], not_covered: [], unchecked: 2, top8: false }], ['done', {}],
    ]));
    render(<Match />);
    await send();
    expect(await screen.findByText("2 requirements couldn't be checked")).toBeInTheDocument();
    expect(screen.queryByText('Where my pages cover this role:')).not.toBeInTheDocument();
    expect(screen.queryByText(result.items[0].text)).not.toBeInTheDocument();
    expect(screen.queryByText(/Not covered on the pages/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Top 8 requirements/)).not.toBeInTheDocument();
  });

  it('lists each not-covered requirement on its own line, keeping commas inside one', async () => {
    const withComma = 'Attention to detail with messy data, onboarding data from spreadsheets, PDFs and exports';
    mockEndpoints(async () => eventResponse([
      ['result', { ...result, not_covered: [withComma, 'Build validation checks'] }], ['done', {}],
    ]));
    render(<Match />);
    await send();
    const list = await screen.findByRole('list', { name: 'Not covered on the pages' });
    expect([...list.querySelectorAll('li')].map((li) => li.textContent)).toEqual([withComma, 'Build validation checks']);
    expect(screen.getByText('Not covered on the pages:')).toBeInTheDocument();
  });

  it('names the project page of each section in the guide and the panel', async () => {
    const pageTitles = { '/projects/roleradar': 'RoleRadar', '/projects/oncallops': 'OnCallOps Operations Dashboard' };
    const twins = [
      { url: '/projects/roleradar#in-30-seconds', title: 'In 30 seconds' },
      { url: '/projects/oncallops#in-30-seconds', title: 'In 30 seconds' },
      { url: '/about#me', title: 'Me' },
    ];
    mockEndpoints(async () => eventResponse([
      ['notes', { requirements: [{ id: 'r1', text: 'SQL' }], findings: [{ requirement_id: 'r1', status: 'found', sections: twins }] }],
      ['result', { ...result, items: [{ ...result.items[0], sections: twins }] }], ['done', {}],
    ]));
    render(<Match pageTitles={pageTitles} />);
    await send();
    await screen.findByText('Where my pages cover this role:');
    const panel = screen.getByText('How it matched').closest('details')!;
    const guide = screen.getByText('Where my pages cover this role:').parentElement!;
    for (const scope of [guide, panel]) {
      expect([...scope.querySelectorAll('a')].map((a) => [a.textContent, a.getAttribute('href')])).toEqual([
        ['RoleRadar · In 30 seconds', '/projects/roleradar#in-30-seconds'],
        ['OnCallOps Operations Dashboard · In 30 seconds', '/projects/oncallops#in-30-seconds'],
        ['Me', '/about#me'],
      ]);
    }
  });

  it('shows only present panel metadata, including unchecked status', async () => {
    mockEndpoints(async () => eventResponse([
      ['notes', {
        requirements: [{ id: 'r1', text: 'Unverified' }, { id: 'r2', text: 'Verified' }],
        findings: [
          { requirement_id: 'r1', status: 'unchecked' },
          { requirement_id: 'r2', status: 'found', provider: 'provider-name', ms: 0, sections: [section] },
        ],
      }], ['done', {}],
    ]));
    render(<Match />);
    await send();
    const panel = (await screen.findByText('How it matched')).closest('details')!;
    expect(panel).toHaveTextContent("couldn't be checked");
    expect(panel).toHaveTextContent('provider-name');
    expect(panel).toHaveTextContent('0 ms');
    expect(panel).not.toHaveTextContent('no model');
    expect(panel).not.toHaveTextContent('undefined');
    expect(panel).not.toHaveTextContent('cut to');
  });

  it('renders hostile fields literally, omits extra fields and never links unsafe URLs', async () => {
    const hostile = '<script>alert(1)</script> [x](https://evil.example)';
    const sections = [{ ...section, title: hostile }, ...['https://evil.example/x', '//evil.example', 'javascript:alert(1)'].map((url) => ({ url, title: hostile }))];
    mockEndpoints(async () => eventResponse([
      ['notes', { requirements: [{ id: hostile, text: hostile }], findings: [{ requirement_id: hostile, status: 'not_found', sections, provider: hostile, ms: 5 }], reasoning: 'SECRET REASONING', model: 'SECRET MODEL' }],
      ['result', { ...result, items: [{ requirement_id: hostile, requirement: 'HIDDEN REQUIREMENT ' + hostile, text: hostile, sections }], not_covered: [hostile] }], ['done', {}],
    ]));
    const { container } = render(<Match />);
    await send();
    await screen.findByRole('button', { name: 'This guide is wrong' });
    expect(container.textContent).toContain(hostile);
    expect(container.textContent).not.toContain('HIDDEN REQUIREMENT');
    expect(container.textContent).not.toContain('SECRET');
    expect(container.querySelector('script')).toBeNull();
    expect(container.querySelectorAll('a')).toHaveLength(3);
    for (const link of container.querySelectorAll('a')) expect(link.getAttribute('href')).toMatch(/^\/(?!\/)/);
    const panel = screen.getByText('How it matched').closest('details')!;
    expect(panel).toHaveTextContent(hostile);
    expect(panel).toHaveTextContent('not found');
    expect(screen.getAllByText(hostile).length).toBeGreaterThan(3);
  });

  it('clears guide, panel, feedback and error for the next run', async () => {
    const stream = controlledStream();
    const endpoint = vi.fn<Endpoint>().mockResolvedValueOnce(eventResponse([
      ['notes', notes], ['result', result], ['error', { message: 'Old error' }],
    ])).mockResolvedValueOnce(response(200, {}, stream.body));
    mockEndpoints(endpoint);
    render(<Match />);
    await send();
    await screen.findByText('Old error');
    await send('Next ad');
    expect(screen.queryByText('Old error')).not.toBeInTheDocument();
    expect(screen.queryByText('How it matched')).not.toBeInTheDocument();
    expect(screen.queryByText(result.items[0].text)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'This guide is wrong' })).not.toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); stream.close(); });
  });

  it('silently re-enables failed feedback for retry and ignores its response after unmount', async () => {
    const fetchMock = mockEndpoints(async () => eventResponse([['result', result], ['done', {}]]));
    const { unmount, container } = render(<Match />);
    await send();
    const button = await screen.findByRole('button', { name: 'This guide is wrong' });
    fetchMock.mockResolvedValueOnce(response(429));
    await act(async () => { fireEvent.click(button); });
    expect(button).toBeEnabled();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    let resolveFeedback!: (value: ReturnType<typeof response>) => void;
    fetchMock.mockImplementationOnce(() => new Promise((resolve) => { resolveFeedback = resolve; }));
    fireEvent.click(button);
    expect(button).toBeDisabled();
    unmount();
    const calls = fetchMock.mock.calls.filter(([url]) => url === '/api/feedback');
    expect(calls).toHaveLength(2);
    expect(calls[1][1]?.signal?.aborted).toBe(true);
    await act(async () => { resolveFeedback(response(204)); });
    expect(container).toBeEmptyDOMElement();
  });

  it('blocks 20,001 code points but sends an emoji input within the code-point limit', async () => {
    const fetchMock = mockEndpoints();
    render(<Match />);
    await send('😀'.repeat(20_001));
    expect(screen.getByRole('alert')).toHaveTextContent(fallbacks[3][1]);
    expect(fetchMock).not.toHaveBeenCalled();
    const allowed = '😀'.repeat(10_001);
    expect(allowed.length).toBeGreaterThan(20_000);
    await send(allowed);
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => url === '/api/match')).toHaveLength(1));
    expect(JSON.parse(String(fetchMock.mock.calls.find(([url]) => url === '/api/match')![1]?.body))).toEqual({ input: allowed });
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('Match failures', () => {
  it.each(fallbacks)('prefers custom SSE messages and falls back for %s', async (code, fallback) => {
    const endpoint = vi.fn<Endpoint>().mockResolvedValueOnce(eventResponse([['error', { code, message: 'Custom server message' }]]))
      .mockResolvedValueOnce(eventResponse([['error', { code }]]))
      .mockResolvedValueOnce(eventResponse([['error', { code, message: '  ' }]]));
    mockEndpoints(endpoint);
    render(<Match />);
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent('Custom server message');
    await send();
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(fallback));
    await send();
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent(fallback));
  });

  it.each([
    [429, 'limit_day', fallbacks[6][1]],
    [429, 'slow_down', 'Too many requests — please wait a minute and try again.'],
    [422, 'whatever', fallbacks[3][1]],
    [403, 'unknown', fallbacks[8][1]],
    [503, 'unavailable', fallbacks[4][1]],
  ] as const)('maps HTTP %i %s', async (status, code, message) => {
    mockEndpoints(async () => response(status, { code, message: 'Untrusted HTTP message' }));
    render(<Match />);
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(message);
  });

  it.each([false, true])('refreshes auth once, repeated rejection = %s', async (repeat) => {
    sessionStorage.setItem('session', JSON.stringify({ token: 'old', expires_at: new Date(Date.now() + 3_600_000).toISOString() }));
    const endpoint = vi.fn<Endpoint>().mockResolvedValueOnce(response(401, { code: 'auth' }))
      .mockResolvedValueOnce(repeat ? response(401, { code: 'auth' }) : eventResponse([['result', result], ['done', {}]]));
    const fetchMock = mockEndpoints(endpoint);
    render(<Match />);
    await send();
    if (repeat) expect(await screen.findByRole('alert')).toHaveTextContent('Please try again.');
    else await screen.findByText(result.items[0].text);
    expect(endpoint).toHaveBeenCalledTimes(2);
    expect(widget.api.execute).toHaveBeenCalledOnce();
    expect(widget.api.reset).toHaveBeenCalledOnce();
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/session')).toHaveLength(1);
    expect(fetchMock.mock.calls.filter(([url]) => url === '/api/match').map(([, init]) => init?.headers)).toEqual([
      { 'Content-Type': 'application/json', Authorization: 'Bearer old' },
      { 'Content-Type': 'application/json', Authorization: 'Bearer session-token' },
    ]);
    if (repeat) expect((await import('../../src/lib/session')).getToken()).toBeNull();
  });

  it.each(['missing key', 'widget failure', 'invalid session'])('shows unavailable for verification failure: %s', async (failure) => {
    const fetchMock = mockEndpoints();
    if (failure === 'missing key') {
      vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', '');
      vi.spyOn(console, 'warn').mockImplementation(() => {});
    } else if (failure === 'widget failure') {
      widget.api.execute.mockImplementation(() => widget.options()['error-callback']());
    } else {
      fetchMock.mockResolvedValueOnce(response(200, { token: 42 }));
    }
    render(<Match />);
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(fallbacks[4][1]);
    expect(fetchMock.mock.calls.some(([url]) => url === '/api/match')).toBe(false);
  });

  it.each(['end', 'reader rejection', 'fetch rejection'])('reports dropped connection for %s', async (ending) => {
    mockEndpoints(async () => {
      if (ending === 'fetch rejection') throw new Error('Network failure');
      const stream = controlledStream();
      if (ending === 'end') stream.close(); else stream.fail();
      return response(200, {}, stream.body);
    });
    render(<Match />);
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(dropped);
    expect(screen.queryByRole('button', { name: 'This guide is wrong' })).not.toBeInTheDocument();
  });

  it('uses a generic message for an unknown malformed error', async () => {
    mockEndpoints(async () => eventResponse([['error', { code: { bad: true }, message: 42 }]]));
    render(<Match />);
    await send();
    expect(await screen.findByRole('alert')).toHaveTextContent(fallbacks[8][1]);
  });

  it('aborts on unmount and ignores a late match response', async () => {
    let resolveMatch!: (value: ReturnType<typeof response>) => void;
    const fetchMock = mockEndpoints(() => new Promise((resolve) => { resolveMatch = resolve; }));
    const { unmount, container } = render(<Match />);
    await send();
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => url === '/api/match')).toBe(true));
    const signal = fetchMock.mock.calls.find(([url]) => url === '/api/match')![1]?.signal;
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { resolveMatch(eventResponse([['result', result], ['done', {}]])); });
    expect(container).toBeEmptyDOMElement();
  });
});
