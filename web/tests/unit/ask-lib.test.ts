import { describe, expect, it } from 'vitest';
import { answerSegments, safeSiteUrl } from '../../src/lib/answer';
import { parseSse } from '../../src/lib/sse';

function stream(chunks: Uint8Array[]) {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });
}

async function events(chunks: Uint8Array[]) {
  const result = [];
  for await (const event of parseSse(stream(chunks))) result.push(event);
  return result;
}

describe('parseSse', () => {
  it('preserves split event lines and UTF-8 bytes, with several events per chunk', async () => {
    const bytes = new TextEncoder().encode('event: token\ndata: {"text":"你"}\n\nevent: done\ndata:{}\n\n');
    const split = bytes.indexOf(0xe4) + 1;
    expect(await events([bytes.slice(0, 4), bytes.slice(4, split), bytes.slice(split, split + 1), bytes.slice(split + 1)]))
      .toEqual([{ event: 'token', data: { text: '你' } }, { event: 'done', data: {} }]);
  });

  it('joins data lines, handles split CRLF, comments, optional spaces, default names and invalid JSON', async () => {
    const encoder = new TextEncoder();
    expect(await events([
      encoder.encode(': keep alive\r\nevent: answer\r'),
      encoder.encode('\ndata: {"text":\r\ndata:"hello"}\r\n\r\nevent: bad\ndata:invalid\n\ndata: {"ok":true}\n\n'),
    ])).toEqual([
      { event: 'answer', data: { text: 'hello' } },
      { event: 'message', data: { ok: true } },
    ]);
  });

  it('cancels and releases the reader when the consumer stops at a terminal event', async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) { controller.enqueue(new TextEncoder().encode('event: done\ndata:{}\n\n')); },
      cancel() { cancelled = true; },
    });
    for await (const event of parseSse(body)) {
      expect(event.event).toBe('done');
      break;
    }
    expect(cancelled).toBe(true);
    expect(body.locked).toBe(false);
  });
});

describe('safeSiteUrl', () => {
  it.each(['//example.com', '/\\example.com', 'javascript:alert(1)', 'https://example.com', '/pro\njects', '/pro\tjects', '/two words', '/control\u0000', '/control\u007f', ''])('rejects %j', (url) => {
    expect(safeSiteUrl(url)).toBeNull();
  });

  it('keeps safe site paths and fragments unchanged', () => {
    expect(safeSiteUrl('/projects/roleradar#x')).toBe('/projects/roleradar#x');
  });
});

describe('answerSegments', () => {
  const citations = [
    { n: 1, url: '/projects/roleradar#x', heading: 'RoleRadar' },
    { n: 2, url: '//example.com', heading: 'Unsafe' },
  ];

  it('links matching safe citations and leaves unmatched or unsafe markers as text', () => {
    expect(answerSegments('See [1], [2], [9].', citations, [])).toEqual([
      { type: 'text', text: 'See ' },
      { type: 'link', text: '[1]', href: '/projects/roleradar#x', title: 'RoleRadar' },
      { type: 'text', text: ', [2], [9].' },
    ]);
  });

  it('merges overlapping unsupported occurrences and preserves a citation inside a mark', () => {
    expect(answerSegments('Claim [1] here. Claim.', citations, ['Claim [1]', '[1] here', 'Claim', '', 'missing']))
      .toEqual([
        { type: 'unsupported', children: [
          { type: 'text', text: 'Claim ' },
          { type: 'link', text: '[1]', href: '/projects/roleradar#x', title: 'RoleRadar' },
          { type: 'text', text: ' here' },
        ] },
        { type: 'text', text: '. ' },
        { type: 'unsupported', children: [{ type: 'text', text: 'Claim' }] },
        { type: 'text', text: '.' },
      ]);
  });

  it('keeps HTML-looking model output as plain segment text', () => {
    expect(answerSegments('<script>alert(1)</script>', [], [])).toEqual([
      { type: 'text', text: '<script>alert(1)</script>' },
    ]);
  });
});
