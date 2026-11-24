import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';

afterEach(() => {
  cleanup();
  for (const script of turnstileScripts()) script.remove();
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
});

it('labels a finished blog citation with its article title and heading', async () => {
  sessionStorage.clear();
  vi.stubEnv('DEV', false);
  vi.stubEnv('PUBLIC_TURNSTILE_SITEKEY', 'test-sitekey');
  mockTurnstile();
  vi.resetModules();
  const Assistant = (await import('../../src/components/Assistant')).default;
  const events = [
    ['answer', { text: 'An answer [1].', citations: [{ n: 1, url: '/blog/a-post#section', heading: 'Section' }] }],
    ['done', {}],
  ];
  vi.stubGlobal('fetch', vi.fn(async (input) => {
    if (input === '/api/ask') {
      return new Response(events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join(''));
    }
    return new Response(JSON.stringify(input === '/api/session'
      ? { token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() }
      : {}));
  }));
  render(<Assistant examples={[]} pageTitles={{ '/blog/a-post': 'Article title' }} />);
  const box = screen.getByRole('textbox', { name: 'Ask a question, or paste a job ad' });
  fireEvent.change(box, { target: { value: 'A question?' } });
  fireEvent.submit(box.closest('form')!);
  expect(await screen.findByRole('link', { name: 'Article title · Section' }))
    .toHaveAttribute('href', '/blog/a-post#section');
});
