import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';
import HowItAnswered from '../../src/components/HowItAnswered';
import type { AnswerTrace } from '../../src/components/HowItAnswered';

let Assistant: typeof import('../../src/components/Assistant').default;

const examples = ['First question?', 'Second question?', 'Third question?'];
const pageTitles = { '/projects/roleradar': 'RoleRadar' };
const jobLink = 'https://boards.greenhouse.io/acme/jobs/1234567';
const badLink = 'https://boards.greenhouse.io/acme';
const badLinkMessage = 'Only Greenhouse and Lever job links can be read — paste the job ad text instead.';
const section = { url: '/projects/roleradar#architecture', title: 'Architecture' };
const matchResult = {
  match_id: 'match-1',
  items: [{ requirement_id: 'r1', requirement: 'Build backend services', text: 'See how I built services.', sections: [section] }],
  not_covered: ['Kubernetes', 'Java, Kotlin'], unchecked: 1, top8: true, language: 'en',
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
  ask = async () => eventResponse([['done', {}]]),
  match = async () => eventResponse([['done', {}]]),
}: { ask?: Endpoint; match?: Endpoint } = {}) {
  const fetchMock = vi.fn((input: RequestInfo | URL, _init?: RequestInit) => {
    if (input === '/api/ask') return ask();
    if (input === '/api/match') return match();
    if (input === '/api/session') {
      return Promise.resolve(response(200, { token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() }));
    }
    if (input === '/api/health') return Promise.resolve(response(200, { db: 'ok', pgvector: '0.8.0' }));
    if (input === '/api/ping' || input === '/api/feedback') return Promise.resolve(response());
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function urls(fetchMock: ReturnType<typeof mockEndpoints>) {
  return fetchMock.mock.calls.map(([url]) => url);
}

function box() {
  return screen.getByRole('textbox', { name: 'Ask a question, or paste a job ad' });
}

function draft(text: string) {
  fireEvent.change(box(), { target: { value: text } });
}

async function submit(text: string) {
  fireEvent.focus(box());
  draft(text);
  await act(async () => { fireEvent.submit(box().closest('form')!); });
}

function renderAssistant(props: Partial<Parameters<typeof Assistant>[0]> = {}) {
  return render(<Assistant examples={examples} pageTitles={pageTitles} {...props} />);
}

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  vi.stubEnv('DEV', false);
  vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
  mockTurnstile();
  Assistant = (await import('../../src/components/Assistant')).default;
});

afterEach(() => {
  cleanup();
  for (const script of turnstileScripts()) script.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('Assistant input box', () => {
  it('has one labelled textarea in section#ask and names its button from the input', () => {
    mockEndpoints();
    const { container } = renderAssistant();
    const section = container.querySelector('section#ask')!;
    expect(section).toHaveAttribute('aria-label', 'Ask about my projects');
    expect(section.querySelectorAll('textarea')).toHaveLength(1);
    expect(box()).toHaveAttribute('placeholder', 'Ask a question, or paste a job ad');
    expect(box()).not.toHaveAttribute('maxlength');
    expect(container.querySelector('#match')).toBeNull();
    expect(screen.getByRole('button', { name: 'Ask' })).toBeEnabled();
    draft('How does RoleRadar work?');
    expect(screen.getByRole('button', { name: 'Ask' })).toBeInTheDocument();
    draft(jobLink);
    expect(screen.getByRole('button', { name: 'Match job' })).toBeInTheDocument();
    draft('x'.repeat(501));
    expect(screen.getByRole('button', { name: 'Match job' })).toBeInTheDocument();
    draft('Short again');
    expect(screen.getByRole('button', { name: 'Ask' })).toBeInTheDocument();
  });

  it('switches by hand with the link under the box, but offers no question link past 500 characters', () => {
    mockEndpoints();
    renderAssistant();
    draft('Senior engineer, Python and React');
    fireEvent.click(screen.getByRole('button', { name: 'Match this as a job ad' }));
    expect(screen.getByRole('button', { name: 'Match job' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Ask this as a question instead' }));
    expect(screen.getByRole('button', { name: 'Ask' })).toBeInTheDocument();
    draft('x'.repeat(500));
    expect(screen.getByRole('button', { name: 'Ask' })).toBeInTheDocument();
    draft('x'.repeat(501));
    expect(screen.getByRole('button', { name: 'Match job' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Ask this as a question instead' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Match this as a job ad' })).not.toBeInTheDocument();
  });

  it('refuses a link that is not a Greenhouse or Lever job and sends nothing', async () => {
    const fetchMock = mockEndpoints();
    renderAssistant();
    await submit(badLink);
    const hint = screen.getByText(badLinkMessage, { selector: '.ask-hint' });
    expect(screen.getByRole('status')).toHaveTextContent(badLinkMessage);
    expect(box()).toHaveValue(badLink);
    expect(box()).toHaveAttribute('aria-describedby', hint.id);
    expect(urls(fetchMock)).not.toContain('/api/match');
    expect(urls(fetchMock)).not.toContain('/api/ask');
    expect(urls(fetchMock)).not.toContain('/api/session');
    draft(`${badLink}/jobs`);
    expect(screen.queryByText(badLinkMessage, { selector: '.ask-hint' })).not.toBeInTheDocument();
  });

  it('sends with Ctrl or Command and Enter, while a plain Enter is a new line', async () => {
    const fetchMock = mockEndpoints();
    renderAssistant();
    fireEvent.focus(box());
    draft('Question by keyboard?');
    fireEvent.keyDown(box(), { key: 'Enter' });
    expect(urls(fetchMock)).not.toContain('/api/ask');
    fireEvent.keyDown(box(), { key: 'Enter', metaKey: true });
    await waitFor(() => expect(urls(fetchMock)).toContain('/api/ask'));
    expect(JSON.parse(fetchMock.mock.calls.find(([url]) => url === '/api/ask')![1]!.body as string))
      .toMatchObject({ question: 'Question by keyboard?' });
    await waitFor(() => expect(box()).toHaveValue(''));
  });

  it('pings only on first focus and waits for it before sending a question', async () => {
    let resolvePing!: (value: ReturnType<typeof response>) => void;
    const fetchMock = mockEndpoints();
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((input, init) => input === '/api/ping'
      ? new Promise((resolve) => { resolvePing = resolve; }) as never : base(input, init));
    renderAssistant();
    expect(urls(fetchMock)).not.toContain('/api/ping');
    await submit('Early question?');
    expect(urls(fetchMock).filter((url) => url === '/api/ping')).toHaveLength(1);
    expect(screen.getByText('Getting ready…')).toBeInTheDocument();
    expect(urls(fetchMock)).not.toContain('/api/ask');
    await act(async () => { resolvePing(response()); });
    await waitFor(() => expect(urls(fetchMock)).toContain('/api/ask'));
    expect(screen.queryByText('Getting ready…')).not.toBeInTheDocument();
  });

  it('sends an example question as soon as it is clicked, after the "Try:" prefix', async () => {
    const fetchMock = mockEndpoints();
    renderAssistant();
    const group = screen.getByRole('group', { name: 'Example questions' });
    expect(group).toHaveTextContent(/^Try:/);
    const buttons = within(group).getAllByRole('button');
    expect(buttons.map((button) => button.textContent)).toEqual(examples);
    fireEvent.click(buttons[1]!);
    await waitFor(() => expect(urls(fetchMock)).toContain('/api/ask'));
    expect(JSON.parse(fetchMock.mock.calls.find(([url]) => url === '/api/ask')![1]!.body as string))
      .toMatchObject({ question: 'Second question?' });
    expect(await screen.findByText('Second question?', { selector: '.ask-question' })).toBeInTheDocument();
  });

  it('keeps the privacy sentence and shows the database line only in development', async () => {
    mockEndpoints();
    const { unmount } = renderAssistant();
    expect(screen.getByText(/processed by AI providers/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'See Privacy.' })).toHaveAttribute('href', '/privacy');
    expect(screen.queryByText(/db: /)).not.toBeInTheDocument();
    unmount();
    vi.stubEnv('DEV', true);
    renderAssistant();
    expect(await screen.findByText('db: ok · pgvector 0.8.0')).toBeInTheDocument();
  });
});

describe('Assistant answers', () => {
  it('shows the example slot until the first question, then the conversation', async () => {
    mockEndpoints();
    renderAssistant({ intro: <h1>Intro headline</h1>, example: <div className="example-answer">Example card</div> });
    expect(screen.getByRole('heading', { name: 'Intro headline' })).toBeInTheDocument();
    expect(screen.getByText('Example card')).toBeInTheDocument();
    await submit('First?');
    await waitFor(() => expect(screen.queryByText('Example card')).not.toBeInTheDocument());
    expect(screen.getByRole('list', { name: 'Conversation' })).toBeInTheDocument();
  });

  it('follows the hook steps with pills, streams into .ask-answer, then shows sources, Checked and feedback', async () => {
    const stream = controlledStream();
    const fetchMock = mockEndpoints({ ask: async () => response(200, {}, stream.body) });
    const { container } = renderAssistant();
    await submit('  Explain RoleRadar?  ');
    await waitFor(() => expect(urls(fetchMock)).toContain('/api/ask'));
    expect(box()).toHaveValue('');
    expect(screen.getByRole('status')).toHaveTextContent('Answering your question…');
    const answer = container.querySelector('.ask-answer')!;
    expect(answer).toHaveTextContent('AI answer');
    expect(answer.querySelector('.ask-shimmer')).not.toBeNull();
    await act(async () => {
      stream.emit('meta', { conversation_id: 'conversation-1', turn_id: 'turn-1' });
      stream.emit('route', { private: false, off_topic: false, language: 'en' });
      stream.emit('status', { node: 'recall', state: 'start' });
      stream.emit('status', { node: 'rewrite', state: 'start' });
    });
    const pills = () => [...answer.querySelectorAll('.ask-step')].map((pill) => [pill.textContent, pill.getAttribute('data-state')]);
    expect(pills()).toEqual([['Rewrite', 'active']]);
    await act(async () => {
      stream.emit('status', { node: 'rewrite', state: 'end' });
      stream.emit('status', { node: 'retrieve', state: 'start' });
    });
    expect(pills()).toEqual([['✓ Rewrite', 'done'], ['Search', 'active']]);
    await act(async () => {
      stream.emit('status', { node: 'retrieve', state: 'end' });
      stream.emit('status', { node: 'answer', state: 'start' });
      stream.emit('token', { text: 'Streaming ' });
    });
    expect(answer).toHaveTextContent('Streaming');
    expect(pills().map(([name]) => name)).toEqual(['✓ Rewrite', '✓ Search', 'Answer']);
    // The answer area is not a live region: the status line speaks instead.
    expect(container.querySelector('[aria-atomic]')).toBeNull();
    expect(answer).not.toHaveAttribute('aria-live');
    await act(async () => {
      stream.emit('answer', {
        text: 'Supported [1]. Unsupported sentence. <script>alert(1)</script> [2] [9]',
        citations: [
          { n: 1, url: '/projects/roleradar#x', heading: 'Architecture' },
          { n: 2, url: 'javascript:alert(1)', heading: 'Unsafe' },
          { n: 3, url: '/about', heading: 'About Skylar' },
        ],
      });
      stream.emit('status', { node: 'check', state: 'start' });
    });
    const chip = screen.getByRole('link', { name: 'Source 1: RoleRadar · Architecture' });
    expect(chip).toHaveTextContent(/^1$/);
    expect(chip).toHaveClass('ask-cite');
    expect(chip).toHaveAttribute('href', '/projects/roleradar#x');
    expect(screen.queryByRole('link', { name: /Source 2/ })).not.toBeInTheDocument();
    expect(answer.querySelector('script')).toBeNull();
    expect(answer).toHaveTextContent('<script>alert(1)</script>');
    expect(screen.queryByText('Sources')).not.toBeInTheDocument();
    await act(async () => { stream.emit('check', { passed: true, unsupported: [], retried: false }); });
    expect(screen.queryByText('Checked by a second model')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'This answer is wrong' })).not.toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); stream.close(); });
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Answer ready.'));
    expect(answer.querySelector('.ask-steps')).toBeNull();
    expect(answer.querySelector('.ask-shimmer')).toBeNull();
    const sources = answer.querySelector('.ask-sources')!;
    expect(within(sources as HTMLElement).getAllByRole('link').map((link) => [link.textContent, link.getAttribute('href')]))
      .toEqual([['RoleRadar · Architecture', '/projects/roleradar#x'], ['About Skylar', '/about']]);
    expect(screen.getByText('Checked by a second model')).toHaveClass('ask-checked');
    fireEvent.click(screen.getByRole('button', { name: 'This answer is wrong' }));
    expect(await screen.findByText('Thanks — noted')).toHaveAttribute('role', 'status');
    expect(fetchMock.mock.calls.find(([url]) => url === '/api/feedback')?.[1]).toMatchObject({ body: JSON.stringify({ turn_id: 'turn-1' }) });
  });

  it('keeps source boxes and the punctuation right after them on one line', async () => {
    const citations = [
      { n: 1, url: '/projects/roleradar#a', heading: 'A', quotes: [] },
      { n: 2, url: '/projects/roleradar#b', heading: 'B', quotes: [] },
    ];
    const text = 'Code decides the tier [1][2]. It reads ads [2], then stops [1] here. Wrong [1].';
    mockEndpoints({ ask: async () => eventResponse([
      ['meta', { conversation_id: 'c', turn_id: 't' }],
      ['answer', { text, citations }],
      ['check', { passed: false, unsupported: ['Wrong [1].'], reason: 'unsupported' }],
      ['done', {}],
    ]) });
    renderAssistant();
    await submit('Question?');
    await screen.findByRole('button', { name: 'This answer is wrong' });
    const paragraph = document.querySelector('.ask-answer-text')!;
    expect(paragraph.textContent).toBe('Code decides the tier 12. It reads ads 2, then stops 1 here. Wrong 1.');
    const runs = [...paragraph.querySelectorAll('.ask-cite-run')];
    expect(runs.map((run) => run.textContent)).toEqual(['12.', '2,', '1.']);
    expect(runs[0]!.querySelectorAll('a.ask-cite')).toHaveLength(2);
    expect(runs[2]!.closest('mark.ask-unsupported')).not.toBeNull();
    // A box followed by a word, not punctuation, is not wrapped.
    const loose = [...paragraph.querySelectorAll('a.ask-cite')].filter((link) => !link.closest('.ask-cite-run'));
    expect(loose.map((link) => link.textContent)).toEqual(['1']);
  });

  it('marks unsupported sentences and shows no Checked line after a failed check', async () => {
    mockEndpoints({ ask: async () => eventResponse([
      ['meta', { conversation_id: 'c', turn_id: 't' }],
      ['answer', { text: 'Fine. Unsupported sentence.', citations: [] }],
      ['check', { passed: false, unsupported: ['Unsupported sentence.'], reason: 'unsupported' }],
      ['done', {}],
    ]) });
    renderAssistant();
    await submit('Question?');
    const mark = await screen.findByText('Unsupported sentence.');
    expect(mark.tagName).toBe('MARK');
    expect(mark).toHaveClass('ask-unsupported');
    expect(mark).toHaveAttribute('title', 'Not found in the project pages');
    await screen.findByRole('button', { name: 'This answer is wrong' });
    expect(screen.queryByText('Checked by a second model')).not.toBeInTheDocument();
  });

  it('shows a private answer only after its check, with only the pills while waiting', async () => {
    const stream = controlledStream();
    mockEndpoints({ ask: async () => response(200, {}, stream.body) });
    const { container } = renderAssistant();
    await submit('Do you have work rights?');
    await waitFor(() => expect(container.querySelector('.ask-answer')).not.toBeNull());
    await act(async () => {
      stream.emit('route', { private: true, off_topic: false, language: 'en' });
      stream.emit('status', { node: 'answer', state: 'start' });
      stream.emit('token', { text: 'Secret draft' });
      stream.emit('answer', { text: 'Approved answer', citations: [] });
    });
    const answer = container.querySelector('.ask-answer')!;
    expect(answer).not.toHaveTextContent('Secret draft');
    expect(answer).not.toHaveTextContent('Approved answer');
    expect(answer.querySelector('.ask-step')).toHaveTextContent('Answer');
    await act(async () => { stream.emit('check', { passed: true, unsupported: [], retried: false }); });
    expect(answer).toHaveTextContent('Approved answer');
    await act(async () => { stream.emit('done', {}); stream.close(); });
  });

  it('shows errors in the card as .ask-error and says so in the status line', async () => {
    mockEndpoints({ ask: async () => eventResponse([['error', { code: 'busy' }]]) });
    const { container } = renderAssistant();
    await submit('Question?');
    const error = await screen.findByText('Please wait for the current answer to finish.', { selector: '.ask-answer p' });
    expect(error).toHaveClass('ask-error');
    expect(container.querySelector('.ask-answer')).toContainElement(error);
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Please wait for the current answer to finish.'));
  });

  it('reads back stored turns with data-restored and notes when the backend forgot them', async () => {
    sessionStorage.setItem('conversation', JSON.stringify({
      v: 1, id: 'old-conversation',
      turns: [{ turnId: 'turn-old', question: 'Earlier?', text: 'Earlier answer [1]', citations: [{ n: 1, url: '/projects/roleradar#x', heading: 'Architecture' }], unsupported: [], checkResult: { passed: true, reason: null }, feedback: 'idle' }],
    }));
    mockEndpoints({ ask: async () => eventResponse([
      ['meta', { conversation_id: 'new-conversation', turn_id: 'turn-new' }],
      ['answer', { text: 'New answer', citations: [] }], ['done', {}],
    ]) });
    const { container } = renderAssistant({ example: <div>Example card</div> });
    const restored = await waitFor(() => {
      const turn = container.querySelector('.ask-turn[data-restored]');
      expect(turn).not.toBeNull();
      return turn!;
    });
    expect(screen.queryByText('Example card')).not.toBeInTheDocument();
    expect(restored).toHaveTextContent('Earlier answer');
    expect(within(restored as HTMLElement).getByRole('button', { name: 'This answer is wrong' })).toBeEnabled();
    expect(screen.queryByText("Earlier questions are no longer in the assistant's memory.")).not.toBeInTheDocument();
    await submit('Now?');
    expect(await screen.findByText("Earlier questions are no longer in the assistant's memory.")).toBeInTheDocument();
    const turns = container.querySelectorAll('.ask-turn');
    expect(turns).toHaveLength(2);
    expect(turns[1]).not.toHaveAttribute('data-restored');
  });
});

describe('Assistant job match', () => {
  it('matches a job link with pills, then shows .match-result with requirement cards and the follow-up note', async () => {
    const stream = controlledStream();
    const fetchMock = mockEndpoints({ match: async () => response(200, {}, stream.body) });
    const { container } = renderAssistant();
    await submit(`  ${jobLink}  `);
    await waitFor(() => expect(urls(fetchMock)).toContain('/api/match'));
    expect(urls(fetchMock)).not.toContain('/api/ask');
    expect(fetchMock.mock.calls.find(([url]) => url === '/api/match')![1]).toMatchObject({ body: JSON.stringify({ input: jobLink }) });
    expect(box()).toHaveValue('');
    expect(screen.getByRole('status')).toHaveTextContent('Matching the job ad…');
    const card = container.querySelector('.match-result')!;
    expect(card.querySelector('.match-source')).toHaveTextContent('boards.greenhouse.io/acme/jobs/1234567');
    await act(async () => {
      stream.emit('stage', { stage: 'fetching' });
      stream.emit('stage', { stage: 'reading' });
    });
    expect([...card.querySelectorAll('.ask-step')].map((pill) => pill.textContent)).toEqual(['✓ Fetch job', 'Read job']);
    await act(async () => {
      stream.emit('stage', { stage: 'matching', count: 1 });
      stream.emit('notes', { requirements: [{ id: 'r1', text: 'Build backend services' }], findings: [{ requirement_id: 'r1', status: 'found', sections: [section], provider: null, ms: 12 }], truncated: false });
      stream.emit('result', matchResult);
    });
    expect(screen.getByRole('heading', { name: 'Where my pages cover this role' })).toBeInTheDocument();
    const item = card.querySelector('.match-item')!;
    expect(item.querySelector('.match-requirement')).toHaveTextContent('Build backend services');
    expect(item).toHaveTextContent('See how I built services.');
    expect(within(item as HTMLElement).getByRole('link', { name: 'RoleRadar: Architecture' })).toHaveAttribute('href', section.url);
    expect(item).toHaveTextContent('→ RoleRadar: Architecture');
    const gaps = screen.getByRole('list', { name: 'Not covered on the pages' });
    expect([...gaps.querySelectorAll('li')].map((li) => li.textContent)).toEqual(['Kubernetes', 'Java, Kotlin']);
    expect(screen.getByText("1 requirement couldn't be checked")).toBeInTheDocument();
    expect(screen.getByText('Showing up to 12 requirements from the ad.')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'This guide is wrong' })).not.toBeInTheDocument();
    await act(async () => { stream.emit('done', {}); stream.close(); });
    expect(await screen.findByText("Follow-up questions don't include this job ad.")).toBeInTheDocument();
    expect(card.querySelector('.ask-step')).toBeNull();
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Job match ready.'));
    expect(screen.getByText('How it matched').closest('details')).not.toHaveAttribute('open');
    fireEvent.click(screen.getByRole('button', { name: 'This guide is wrong' }));
    expect(await screen.findByText('Thanks — noted')).toBeInTheDocument();
    expect(fetchMock.mock.calls.find(([url]) => url === '/api/feedback')?.[1]).toMatchObject({ body: JSON.stringify({ match_id: 'match-1' }) });
    // The job match is never stored in the browser.
    expect(sessionStorage.getItem('conversation')).toBeNull();
  });

  it('matches pasted text over 500 characters without a link line, and cuts long requirements to 120 characters', async () => {
    const long = 'R'.repeat(200);
    const fetchMock = mockEndpoints({ match: async () => eventResponse([
      ['result', { ...matchResult, top8: false, items: [{ ...matchResult.items[0], requirement: long }] }], ['done', {}],
    ]) });
    const { container } = renderAssistant();
    await submit('Job ad text. '.repeat(50));
    await screen.findByRole('button', { name: 'This guide is wrong' });
    expect(urls(fetchMock)).toContain('/api/match');
    expect(container.querySelector('.match-source')).toBeNull();
    const requirement = container.querySelector('.match-requirement')!.textContent!;
    expect([...requirement]).toHaveLength(120);
    expect(requirement.endsWith('…')).toBe(true);
    // The line shows with every result, not only when the model listed more (SPEC, step 10).
    expect(screen.getByText('Showing up to 12 requirements from the ad.')).toBeInTheDocument();
    expect(screen.queryByText('Top 12 requirements of the job ad shown.')).toBeNull();
  });

  it('keeps the job match in its place among the questions', async () => {
    mockEndpoints({
      ask: async () => eventResponse([['answer', { text: 'Answer', citations: [] }], ['done', {}]]),
      match: async () => eventResponse([['result', matchResult], ['done', {}]]),
    });
    const { container } = renderAssistant();
    await submit('First?');
    await waitFor(() => expect(container.querySelectorAll('.ask-turn')).toHaveLength(1));
    await submit(jobLink);
    await screen.findByRole('button', { name: 'This guide is wrong' });
    await submit('Second?');
    await waitFor(() => expect(container.querySelectorAll('.ask-turn')).toHaveLength(2));
    const order = [...container.querySelectorAll('.ask-turns > li')].map((li) => li.className);
    expect(order).toEqual(['ask-turn', 'match-turn', 'ask-turn']);
  });

  it('renders hostile fields as text and never links unsafe URLs', async () => {
    const hostile = '<script>alert(1)</script> [x](https://evil.example)';
    const sections = [{ ...section, title: hostile }, ...['https://evil.example/x', '//evil.example', 'javascript:alert(1)'].map((url) => ({ url, title: hostile }))];
    mockEndpoints({ match: async () => eventResponse([
      ['result', { ...matchResult, items: [{ requirement_id: 'r1', requirement: hostile, text: hostile, sections }], not_covered: [hostile] }], ['done', {}],
    ]) });
    const { container } = renderAssistant();
    await submit(jobLink);
    await screen.findByRole('button', { name: 'This guide is wrong' });
    const card = container.querySelector('.match-result')!;
    expect(card.textContent).toContain(hostile);
    expect(card.querySelector('script')).toBeNull();
    expect(card.querySelectorAll('a')).toHaveLength(1);
    expect(card.querySelector('a')!.getAttribute('href')).toBe(section.url);
  });

  it('shows a too-long job ad error without a request', async () => {
    const fetchMock = mockEndpoints();
    renderAssistant();
    await submit('x'.repeat(20_001));
    expect(await screen.findByText("That's longer than 20,000 characters — please paste the main part.")).toHaveClass('ask-error');
    expect(urls(fetchMock)).not.toContain('/api/match');
  });
});

describe('Assistant panel variant', () => {
  it('renders the same box without the home slots', () => {
    mockEndpoints();
    const { container } = renderAssistant({ variant: 'panel', intro: <h1>Intro</h1>, example: <p>Example</p> });
    expect(container.querySelector('section#ask.assistant-panel')).not.toBeNull();
    expect(box()).toBeInTheDocument();
    expect(screen.queryByText('Intro')).not.toBeInTheDocument();
    expect(screen.queryByText('Example')).not.toBeInTheDocument();
  });
});

describe('Assistant fixes from the 9c1 check', () => {
  it('numbers sources 1..N by first appearance in the answer and keeps their links', async () => {
    mockEndpoints({ ask: async () => eventResponse([
      ['meta', { conversation_id: 'c', turn_id: 't' }],
      ['answer', {
        text: 'Ranked [6], then [2], and [6] again.',
        citations: [
          { n: 2, url: '/projects/roleradar#b', heading: 'Bravo' },
          { n: 4, url: '/about', heading: 'About Skylar' },
          { n: 5, url: 'javascript:alert(1)', heading: 'Unsafe' },
          { n: 6, url: '/projects/roleradar#a', heading: 'Alpha' },
        ],
      }],
      ['check', { passed: true, unsupported: [], retried: false }],
      ['done', {}],
    ]) });
    const { container } = renderAssistant();
    await submit('Which sources?');
    await screen.findByText('Sources');
    const chips = [...container.querySelectorAll('.ask-answer-text a.ask-cite')]
      .map((chip) => [chip.textContent, chip.getAttribute('href'), chip.getAttribute('aria-label')]);
    expect(chips).toEqual([
      ['1', '/projects/roleradar#a', 'Source 1: RoleRadar · Alpha'],
      ['2', '/projects/roleradar#b', 'Source 2: RoleRadar · Bravo'],
      ['1', '/projects/roleradar#a', 'Source 1: RoleRadar · Alpha'],
    ]);
    const sources = [...container.querySelectorAll('.ask-sources li')].map((item) => [
      item.querySelector('.ask-source-n')!.textContent,
      item.querySelector('a')!.textContent,
      item.querySelector('a')!.getAttribute('href'),
    ]);
    expect(sources).toEqual([
      ['1', 'RoleRadar · Alpha', '/projects/roleradar#a'],
      ['2', 'RoleRadar · Bravo', '/projects/roleradar#b'],
      ['3', 'About Skylar', '/about'],
    ]);
  });

  it('names each source box like its Sources line, page title included (10i)', async () => {
    mockEndpoints({ ask: async () => eventResponse([
      ['meta', { conversation_id: 'c', turn_id: 't' }],
      ['answer', {
        text: 'One [1], other [2], about [3].',
        citations: [
          { n: 1, url: '/projects/roleradar#in-30-seconds', heading: 'In 30 seconds' },
          { n: 2, url: '/projects/careermate#in-30-seconds', heading: 'In 30 seconds' },
          { n: 3, url: '/about', heading: 'About Skylar' },
        ],
      }],
      ['check', { passed: true, unsupported: [], retried: false }],
      ['done', {}],
    ]) });
    const { container } = renderAssistant({ pageTitles: { ...pageTitles, '/projects/careermate': 'CareerMate' } });
    await submit('Which page?');
    await screen.findByText('Sources');
    const chips = [...container.querySelectorAll('.ask-answer-text a.ask-cite')].map((chip) => chip.getAttribute('aria-label'));
    expect(chips).toEqual(['Source 1: RoleRadar · In 30 seconds', 'Source 2: CareerMate · In 30 seconds', 'Source 3: About Skylar']);
    const lines = [...container.querySelectorAll('.ask-sources li a')].map((link, index) => `Source ${index + 1}: ${link.textContent}`);
    expect(chips).toEqual(lines);
  });

  it('scrolls a new question or job match into view on the home page, gently unless motion is reduced', async () => {
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    let top = 2000;
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(() => ({ top } as DOMRect));
    let reduce = true;
    vi.stubGlobal('matchMedia', vi.fn(() => ({ matches: reduce })));
    try {
      mockEndpoints();
      renderAssistant();
      await submit('First question?');
      await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Answer ready.'));
      expect(scroll).toHaveBeenCalledTimes(1);
      expect(scroll).toHaveBeenLastCalledWith({ behavior: 'auto', block: 'start' });
      expect(scroll.mock.contexts[0]).toHaveClass('ask-turn');
      expect(scroll.mock.contexts[0]).toHaveTextContent('First question?');

      reduce = false;
      await submit(jobLink);
      await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Job match ready.'));
      expect(scroll).toHaveBeenCalledTimes(2);
      expect(scroll).toHaveBeenLastCalledWith({ behavior: 'smooth', block: 'start' });
      expect(scroll.mock.contexts[1]).toHaveClass('match-turn');

      // Already in the upper part of the screen: no jump.
      top = 120;
      await submit('Second question?');
      await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Answer ready.'));
      expect(container().querySelectorAll('.ask-turn')).toHaveLength(2);
      expect(scroll).toHaveBeenCalledTimes(2);
      cleanup();

      // The project-page panel keeps its own follow-scroll.
      top = 2000;
      renderAssistant({ variant: 'panel' });
      await submit('Panel question?');
      await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('Answer ready.'));
      expect(scroll).toHaveBeenCalledTimes(2);
    } finally {
      delete (Element.prototype as Partial<Element>).scrollIntoView;
    }

    function container() {
      return document.querySelector('section#ask')!;
    }
  });

  it('speaks the offline sentence through the status line and ties it to the box', async () => {
    const fetchMock = mockEndpoints();
    const base = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((input, init) => input === '/api/ping'
      ? Promise.reject(new TypeError('Failed to fetch')) : base(input, init));
    renderAssistant();
    await submit('Are you there?');
    const hint = await screen.findByText('The assistant is offline right now — please try again later.', { selector: '.ask-hint' });
    expect(hint.id).not.toBe('');
    expect(box().getAttribute('aria-describedby')?.split(' ')).toContain(hint.id);
    await waitFor(() => expect(screen.getByRole('status'))
      .toHaveTextContent('The assistant is offline right now — please try again later.'));
    expect(screen.getAllByRole('status')).toHaveLength(1);
    expect(urls(fetchMock)).not.toContain('/api/ask');
  });

  it('pings again when Ask is pressed after going offline, and asks once it is back', async () => {
    const fetchMock = mockEndpoints();
    const base = fetchMock.getMockImplementation()!;
    let online = false;
    fetchMock.mockImplementation((input, init) => input === '/api/ping' && !online
      ? Promise.reject(new TypeError('Failed to fetch')) : base(input, init));
    renderAssistant();
    fireEvent.focus(box());
    await screen.findByText('The assistant is offline right now — please try again later.', { selector: '.ask-hint' });
    // Still offline: the second ping fails too and the sentence comes back.
    await submit('Still there?');
    await waitFor(() => expect(urls(fetchMock).filter((url) => url === '/api/ping')).toHaveLength(2));
    await screen.findByText('The assistant is offline right now — please try again later.', { selector: '.ask-hint' });
    expect(urls(fetchMock)).not.toContain('/api/ask');
    online = true;
    await submit('Back online?');
    await waitFor(() => expect(urls(fetchMock)).toContain('/api/ask'));
    expect(urls(fetchMock).filter((url) => url === '/api/ping')).toHaveLength(3);
    expect(JSON.parse(fetchMock.mock.calls.find(([url]) => url === '/api/ask')![1]!.body as string))
      .toMatchObject({ question: 'Back online?' });
    expect(screen.queryByText('The assistant is offline right now — please try again later.')).not.toBeInTheDocument();
  });

  it('keeps text typed into the server-rendered box before hydration', async () => {
    mockEndpoints();
    const { renderToString } = await import('react-dom/server');
    const host = document.createElement('div');
    host.innerHTML = renderToString(<Assistant examples={examples} pageTitles={pageTitles} />);
    document.body.appendChild(host);
    host.querySelector('textarea')!.value = jobLink;
    await act(async () => {
      render(<Assistant examples={examples} pageTitles={pageTitles} />, { container: host, hydrate: true });
    });
    expect(box()).toHaveValue(jobLink);
    expect(screen.getByRole('button', { name: 'Match job' })).toBeInTheDocument();
    // Later renders (the ping on focus) keep it too.
    fireEvent.focus(box());
    await waitFor(() => expect(urls(vi.mocked(fetch))).toContain('/api/ping'));
    await act(async () => {});
    expect(box()).toHaveValue(jobLink);
  });
});

describe('Assistant quoted sentences (step 10)', () => {
  const roleradarHtml = `<!doctype html><html><body><main>${readFileSync(join(process.cwd(), 'tests/fixtures/quote-pages/roleradar.html'), 'utf8')}</main></body></html>`;
  const once = 'The second layer runs per user, in code.';

  function withPages(fetchMock: ReturnType<typeof mockEndpoints>, pages: Record<string, string>) {
    const api = fetchMock.getMockImplementation()!;
    const pageFetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const html = pages[String(input)];
      if (html === undefined) return api(input, init);
      return Promise.resolve({
        ok: true, status: 200, headers: new Headers({ 'content-type': 'text/html; charset=utf-8' }),
        text: async () => html,
      });
    });
    vi.stubGlobal('fetch', pageFetch);
    return pageFetch;
  }

  function answer(citations: unknown[]) {
    return async () => eventResponse([
      ['meta', { conversation_id: 'c', turn_id: 't' }],
      ['answer', { text: 'It runs per user [3]. Old answers [4].', citations }],
      ['check', { passed: true, unsupported: [], retried: false }],
      ['done', {}],
    ]);
  }

  it('links a quote found once on its page with a Text Fragment and passes the quotes to highlight.ts', async () => {
    const pageFetch = withPages(mockEndpoints({ ask: answer([
      { n: 3, url: '/projects/roleradar#how-roleradar-matches-jobs', heading: 'How RoleRadar matches jobs', quotes: [once] },
      { n: 4, url: '/projects/roleradar#architecture', heading: 'Architecture' },
    ]) }), { '/projects/roleradar': roleradarHtml });
    const { container } = renderAssistant();
    await submit('How does RoleRadar decide the fit tier?');
    await screen.findByText('Sources');
    const fragment = '/projects/roleradar#how-roleradar-matches-jobs:~:text=The%20second%20layer%20runs%20per%20user%2C%20in%20code.';
    await waitFor(() => expect(container.querySelector('.ask-answer-text a.ask-cite')).toHaveAttribute('href', fragment));
    const chip = container.querySelector('.ask-answer-text a.ask-cite')!;
    expect(JSON.parse(chip.getAttribute('data-quotes')!)).toEqual([once]);
    const links = [...container.querySelectorAll('.ask-sources a')];
    expect(links.map((link) => link.getAttribute('href'))).toEqual([fragment, '/projects/roleradar#architecture']);
    // A source without quotes (or from an older answer) is a plain section link, as before.
    expect(links[1]!.hasAttribute('data-quotes')).toBe(false);
    expect(pageFetch.mock.calls.filter(([url]) => url === '/projects/roleradar')).toHaveLength(1);
  });

  it('leaves the Text Fragment off when the quote appears more than once on that page', async () => {
    const twice = roleradarHtml.replace('</main>', `<p>${once}</p></main>`);
    withPages(mockEndpoints({ ask: answer([
      { n: 3, url: '/projects/roleradar#how-roleradar-matches-jobs', heading: 'How RoleRadar matches jobs', quotes: [once] },
    ]) }), { '/projects/roleradar': twice });
    const { container } = renderAssistant();
    await submit('How does RoleRadar decide the fit tier?');
    await screen.findByText('Sources');
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 20)); });
    const chip = container.querySelector('.ask-answer-text a.ask-cite')!;
    expect(chip).toHaveAttribute('href', '/projects/roleradar#how-roleradar-matches-jobs');
    expect(JSON.parse(chip.getAttribute('data-quotes')!)).toEqual([once]);
  });

  it('reads back a stored turn from before quotes and links its sources as before', async () => {
    sessionStorage.setItem('conversation', JSON.stringify({ v: 1, id: 'c-old', turns: [{
      turnId: 't1', question: 'Old?', text: 'Old answer [1].', unsupported: [], feedback: 'idle',
      citations: [{ n: 1, url: '/projects/roleradar#a', heading: 'Alpha' }],
    }] }));
    mockEndpoints();
    const { container } = renderAssistant();
    expect(await screen.findByText('Old?')).toBeInTheDocument();
    const chip = container.querySelector('.ask-answer-text a.ask-cite')!;
    expect(chip).toHaveAttribute('href', '/projects/roleradar#a');
    expect(chip.hasAttribute('data-quotes')).toBe(false);
  });
});

describe('HowItAnswered', () => {
  it.each([
    ['not_in_about', "The About page doesn't answer this."],
    ['inferred', 'The draft said more than the About page does.'],
    ['unsupported', "Some sentences weren't found in the project pages."],
  ])('explains a failed check with reason %s', (reason, sentence) => {
    render(<HowItAnswered trace={{ nodes: [], sections: [], memory: [] }} checkResult={{ passed: false, reason }} />);
    fireEvent.click(screen.getByText('How it answered'));
    expect(screen.getByRole('heading', { name: 'Check' }).parentElement).toHaveClass('ask-panel-block');
    expect(screen.getByText(sentence).tagName).toBe('P');
  });

  it('omits the reason block for a passed check and renders nothing without a trace', () => {
    const { container, rerender } = render(<HowItAnswered checkResult={{ passed: false, reason: 'unsupported' }} />);
    expect(container).toBeEmptyDOMElement();
    rerender(<HowItAnswered trace={{ nodes: [], sections: [], memory: [] }} checkResult={{ passed: true, reason: 'unsupported' }} />);
    fireEvent.click(screen.getByText('How it answered'));
    expect(screen.queryByRole('heading', { name: 'Check' })).not.toBeInTheDocument();
  });

  it('starts collapsed and lists every step with an icon, safe passages and recalled memory', () => {
    const trace: AnswerTrace = {
      nodes: [
        { name: 'recall', provider: null, model: null, ms: 1 },
        ...['retrieve', 'answer', 'check'].map((name) => ({ name, provider: 'Provider', model: 'Model', ms: 123 })),
      ],
      sections: [
        { url: '/projects/roleradar#x', heading: 'RoleRadar', score: 0.5 },
        { url: '//example.com', heading: 'Unsafe heading', score: 0.0164 },
      ],
      memory: [{ turn_id: 'old', question: 'Earlier question?', score: 0.8 }],
    };
    const { container } = render(<HowItAnswered trace={trace} />);
    const details = container.querySelector('details')!;
    expect(details).not.toHaveAttribute('open');
    fireEvent.click(screen.getByText('How it answered'));
    const lists = within(details).getAllByRole('list');
    const steps = within(lists[0]!).getAllByRole('listitem');
    expect(steps.map((step) => step.querySelector('.ask-trace-icon')?.getAttribute('aria-hidden'))).toEqual(['true', 'true', 'true', 'true']);
    expect(steps.map((step) => step.lastElementChild?.textContent)).toEqual([
      'recall · no model · 1 ms',
      ...['retrieve', 'answer', 'check'].map((name) => `${name} · Provider · Model · 123 ms`),
    ]);
    expect(screen.getByRole('link', { name: 'RoleRadar' })).toHaveAttribute('href', '/projects/roleradar#x');
    expect(screen.queryByRole('link', { name: 'Unsafe heading' })).not.toBeInTheDocument();
    expect(lists[1]).toHaveTextContent('RoleRadar · 0.500');
    expect(lists[1]).toHaveTextContent('Unsafe heading · 0.016');
    expect(lists[2]).toHaveTextContent('Earlier question? · 0.80');
  });

  it('shows None for empty passages and memory', () => {
    render(<HowItAnswered trace={{ nodes: [], sections: [], memory: [] }} />);
    fireEvent.click(screen.getByText('How it answered'));
    expect(screen.getAllByText('None')).toHaveLength(2);
  });
});

it('uses no literal colors and only existing CSS variables in the assistant sources and styles', () => {
  const tokens = readFileSync(join(process.cwd(), 'src/styles/tokens.css'), 'utf8');
  const defined = new Set(Array.from(tokens.matchAll(/^\s*(--[\w-]+)\s*:/gm), (match) => match[1]));
  const sources = ['src/components/Assistant.tsx', 'src/components/HowItAnswered.tsx', 'src/styles/assistant.css']
    .map((path) => readFileSync(join(process.cwd(), path), 'utf8'));
  for (const source of sources) {
    expect(source).not.toMatch(/#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\w-])|\b(?:rgba?|hsl)\s*\(/gi);
    const referenced = Array.from(source.matchAll(/var\(\s*(--[\w-]+)/g), (match) => match[1]);
    expect(referenced.filter((name) => !defined.has(name))).toEqual([]);
  }
  // The old chat styles left global.css.
  expect(readFileSync(join(process.cwd(), 'src/styles/global.css'), 'utf8')).not.toMatch(/\.ask-/);
});
