import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AskPlaceholder from '../../src/components/AskPlaceholder';

const examples = ['First question?', 'Second question?', 'Fourth question?'];

function response(status: number, body: unknown = {}) {
  return { status, json: async (): Promise<unknown> => body };
}

function mockEndpoints(
  ping = async () => response(200),
  health = async () => response(200, { status: 'ok', db: 'ok', pgvector: '0.8.0' }),
) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    if (input === '/api/ping') return ping();
    if (input === '/api/health') return health();
    throw new Error(`Unexpected request: ${String(input)}`);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => {
  vi.stubEnv('DEV', true);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

describe('AskPlaceholder', () => {
  it('renders a labelled textarea, disabled Send button and three example buttons', async () => {
    mockEndpoints();
    render(<AskPlaceholder examples={examples} />);
    expect(screen.getByRole('textbox', { name: 'Ask about my projects' }))
      .toHaveAttribute('maxlength', '500');
    const send = screen.getByRole('button', { name: 'Send' });
    expect(send).toBeDisabled();
    expect(send).toHaveAttribute('type', 'submit');
    const group = screen.getByRole('group', { name: 'Example questions' });
    expect(within(group).getAllByRole('button').map((button) => button.textContent))
      .toEqual(examples);
    for (const button of within(group).getAllByRole('button')) {
      expect(button).toHaveAttribute('type', 'button');
    }
    expect(await screen.findByText('Assistant coming soon')).toBeInTheDocument();
  });

  it('fills the controlled textarea from each example and allows editing without sending', async () => {
    const fetchMock = mockEndpoints();
    render(<AskPlaceholder examples={examples} />);
    await screen.findByText('db: ok · pgvector 0.8.0');
    const textarea = screen.getByRole('textbox', { name: 'Ask about my projects' });
    for (const example of examples) {
      fireEvent.click(screen.getByRole('button', { name: example }));
      expect(textarea).toHaveValue(example);
    }
    fireEvent.change(textarea, { target: { value: 'My own question' } });
    expect(textarea).toHaveValue('My own question');
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('prevents form submission and never sends a question', async () => {
    const fetchMock = mockEndpoints();
    render(<AskPlaceholder examples={examples} />);
    await screen.findByText('db: ok · pgvector 0.8.0');
    const form = screen.getByRole('button', { name: 'Send' }).closest('form');
    expect(form).not.toBeNull();
    expect(fireEvent.submit(form!)).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it.each([
    [200, 'Assistant coming soon'],
    [201, 'Assistant is offline'],
    [500, 'Assistant is offline'],
  ])('shows the correct backend status for ping %i', async (status, message) => {
    mockEndpoints(async () => response(status));
    render(<AskPlaceholder examples={examples} />);
    expect(await screen.findByText(message)).toHaveAttribute('role', 'status');
  });

  it('shows offline when ping rejects', async () => {
    mockEndpoints(async () => { throw new Error('Network failure'); });
    render(<AskPlaceholder examples={examples} />);
    expect(await screen.findByText('Assistant is offline')).toHaveAttribute('role', 'status');
  });

  it('shows a pending message until ping resolves', async () => {
    let resolvePing!: (value: ReturnType<typeof response>) => void;
    mockEndpoints(() => new Promise((resolve) => { resolvePing = resolve; }));
    render(<AskPlaceholder examples={examples} />);
    expect(screen.getByRole('status')).toHaveTextContent('Checking assistant…');
    expect(screen.queryByText('Assistant coming soon')).not.toBeInTheDocument();
    expect(screen.queryByText('Assistant is offline')).not.toBeInTheDocument();
    await act(async () => { resolvePing(response(200)); });
    expect(await screen.findByText('Assistant coming soon')).toBeInTheDocument();
  });

  it.each([
    [200, { status: 'ok', db: 'ok', pgvector: '0.8.0' }, 'db: ok · pgvector 0.8.0'],
    [503, { db: 'ok', pgvector: null }, 'db: ok · pgvector missing'],
    [503, { db: 'down', pgvector: null }, 'db: down'],
    [200, { db: 'ok' }, 'db: down'],
    [200, { pgvector: null }, 'db: down'],
    [200, { pgvector: 8 }, 'db: down'],
    [503, {}, 'db: down'],
    [503, null, 'db: down'],
    [500, { db: 'ok', pgvector: '0.8.0' }, 'db: down'],
  ])('handles health %i with body %j as %s', async (status, body, message) => {
    mockEndpoints(undefined, async () => response(status, body));
    render(<AskPlaceholder examples={examples} />);
    expect(await screen.findByText(message)).toBeInTheDocument();
  });

  it('shows db: down when health rejects', async () => {
    mockEndpoints(undefined, async () => { throw new Error('Network failure'); });
    render(<AskPlaceholder examples={examples} />);
    expect(await screen.findByText('db: down')).toBeInTheDocument();
  });

  it('shows db: down when the health body is not JSON', async () => {
    mockEndpoints(undefined, async () => ({
      status: 503,
      json: async () => { throw new SyntaxError('Invalid JSON'); },
    }));
    render(<AskPlaceholder examples={examples} />);
    expect(await screen.findByText('db: down')).toBeInTheDocument();
  });

  it('omits health requests and the dev line in production while still checking ping', async () => {
    vi.stubEnv('DEV', false);
    const fetchMock = mockEndpoints();
    render(<AskPlaceholder examples={examples} />);
    await screen.findByText('Assistant coming soon');
    expect(fetchMock).toHaveBeenCalledWith('/api/ping', expect.objectContaining({
      signal: expect.any(AbortSignal),
    }));
    expect(fetchMock.mock.calls.some(([input]) => input === '/api/health')).toBe(false);
    expect(screen.queryByText(/^db:/)).not.toBeInTheDocument();
  });

  it('includes the exact privacy sentence and link', async () => {
    mockEndpoints();
    render(<AskPlaceholder examples={examples} />);
    const link = screen.getByRole('link', { name: 'See Privacy.' });
    expect(link).toHaveAttribute('href', '/privacy');
    expect(link.parentElement).toHaveTextContent(
      /^Your question is processed by AI providers\. See Privacy\.$/,
    );
    await screen.findByText('Assistant coming soon');
  });

  it('aborts both requests on unmount and ignores late results', async () => {
    const requests: { signal: AbortSignal; resolve: (value: ReturnType<typeof response>) => void }[] = [];
    const fetchMock = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => (
      new Promise<ReturnType<typeof response>>((resolve) => {
        requests.push({ signal: init!.signal!, resolve });
      })
    ));
    vi.stubGlobal('fetch', fetchMock);
    const { unmount, container } = render(<AskPlaceholder examples={examples} />);
    expect(requests).toHaveLength(2);
    expect(requests.every(({ signal }) => !signal.aborted)).toBe(true);
    unmount();
    expect(requests.every(({ signal }) => signal.aborted)).toBe(true);
    const json = vi.fn(async () => ({ db: 'ok', pgvector: '0.8.0' }));
    await act(async () => {
      for (const { resolve } of requests) resolve({ status: 200, json });
    });
    expect(json).not.toHaveBeenCalled();
    expect(container).toBeEmptyDOMElement();
  });

  it('ignores health JSON that finishes after unmount', async () => {
    let resolveJson!: (body: unknown) => void;
    const json = vi.fn(() => new Promise<unknown>((resolve) => { resolveJson = resolve; }));
    mockEndpoints(undefined, async () => ({ status: 200, json }));
    const { unmount, container } = render(<AskPlaceholder examples={examples} />);
    await waitFor(() => expect(json).toHaveBeenCalledOnce());
    unmount();
    await act(async () => { resolveJson({ db: 'ok', pgvector: '0.8.0' }); });
    expect(container).toBeEmptyDOMElement();
  });
});

it('uses no literal colors and only existing CSS variables in the component source', () => {
  const source = readFileSync(join(process.cwd(), 'src/components/AskPlaceholder.tsx'), 'utf8');
  const tokens = readFileSync(join(process.cwd(), 'src/styles/tokens.css'), 'utf8');
  expect(source).not.toMatch(/#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\w-])|\b(?:rgba?|hsl)\s*\(/gi);
  const defined = new Set(Array.from(tokens.matchAll(/^\s*(--[\w-]+)\s*:/gm), (match) => match[1]));
  const referenced = Array.from(source.matchAll(/var\(\s*(--[\w-]+)/g), (match) => match[1]);
  expect(referenced.length).toBeGreaterThan(0);
  expect(referenced.filter((name) => !defined.has(name))).toEqual([]);
});
