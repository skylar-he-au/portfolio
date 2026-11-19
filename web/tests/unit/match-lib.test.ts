import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let match: typeof import('../../src/lib/match');
let session: typeof import('../../src/lib/session');

beforeEach(async () => {
  vi.resetModules();
  sessionStorage.clear();
  match = await import('../../src/lib/match');
  session = await import('../../src/lib/session');
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('match event parsing', () => {
  it('uses code points for the input limit', () => {
    expect(match.MATCH_MAX_CHARS).toBe(20_000);
    expect(match.matchInputLength('😀'.repeat(10_001))).toBe(10_001);
  });

  it.each([null, 42, 'text', [], { unexpected: true }])('defaults malformed payload %j without throwing', (data) => {
    for (const event of ['stage', 'notes', 'result', 'error', 'done', 'unknown']) {
      expect(() => match.parseMatchEvent({ event, data })).not.toThrow();
    }
    expect(match.parseMatchEvent({ event: 'unknown', data })).toBeNull();
  });

  it('validates every result field and caps arrays and strings', () => {
    const long = 'x'.repeat(700);
    const item = { requirement_id: long, requirement: long, text: long, sections: Array.from({ length: 20 }, () => ({ url: long, title: long })) };
    const event = match.parseMatchEvent({ event: 'result', data: {
      match_id: long, items: [null, 2, ...Array.from({ length: 20 }, () => item)],
      not_covered: [42, null, ...Array(20).fill(long)], unchecked: -1, top8: 'true', language: 42,
      reasoning: 'secret',
    } });
    expect(event?.event).toBe('result');
    if (event?.event !== 'result') throw new Error('Expected result');
    expect(event.data.match_id).toHaveLength(500);
    expect(event.data.items.length).toBeLessThanOrEqual(16);
    expect(event.data.items.length).toBeGreaterThan(0);
    for (const parsed of event.data.items) {
      expect(parsed.requirement_id.length).toBeLessThanOrEqual(500);
      expect(parsed.requirement).toHaveLength(500);
      expect(parsed.text).toHaveLength(500);
      expect(parsed.sections.length).toBeLessThanOrEqual(8);
      for (const section of parsed.sections) {
        expect(section.url).toHaveLength(500);
        expect(section.title).toHaveLength(500);
      }
    }
    expect(event.data.not_covered.length).toBeLessThanOrEqual(16);
    expect(event.data.not_covered.every((value) => typeof value === 'string' && value.length <= 500)).toBe(true);
    expect(event.data.unchecked).toBe(0);
    expect(event.data.top8).toBe(false);
    expect(event.data.language).toBeUndefined();
    expect(event.data).not.toHaveProperty('reasoning');
  });

  it('drops invalid nested fields and preserves only present valid panel fields', () => {
    const event = match.parseMatchEvent({ event: 'notes', data: {
      requirements: [{ id: 'r1', text: 'Original' }, { id: 42, text: {} }],
      findings: [{ requirement_id: 'r1', status: 'made_up', provider: 4, ms: '12', sections: [null, { url: 4, title: {} }] }],
      truncated: 'yes', reasoning: 'secret',
    } });
    if (event?.event !== 'notes') throw new Error('Expected notes');
    expect(event.data.requirements).toEqual([{ id: 'r1', text: 'Original' }]);
    expect(event.data.findings).toEqual([{ requirement_id: 'r1', sections: [] }]);
    expect(event.data.truncated).toBe(false);
    expect(event.data).not.toHaveProperty('reasoning');
  });

  it('caps notes lists and validates finite numeric fields', () => {
    const event = match.parseMatchEvent({ event: 'notes', data: {
      requirements: Array.from({ length: 50 }, () => ({ id: 'r', text: 'x'.repeat(600) })),
      findings: Array.from({ length: 50 }, () => ({ requirement_id: 'r', status: 'unchecked', provider: null, ms: Infinity, sections: Array.from({ length: 10 }, () => ({ url: '/x', title: 'x' })) })),
      truncated: true,
    } });
    if (event?.event !== 'notes') throw new Error('Expected notes');
    expect(event.data.requirements).toHaveLength(16);
    expect(event.data.requirements[0].text).toHaveLength(500);
    expect(event.data.findings).toHaveLength(16);
    expect(event.data.findings[0]).toMatchObject({ status: 'unchecked', provider: null });
    expect(event.data.findings[0].ms).toBeUndefined();
    expect(event.data.findings[0].sections).toHaveLength(8);
    expect(event.data.truncated).toBe(true);
    expect(match.parseMatchEvent({ event: 'stage', data: { stage: 'matching', count: Infinity } }))
      .toEqual({ event: 'stage', data: { stage: 'matching' } });
  });

  it('validates stage, result scalars and malformed sections without coercion', () => {
    expect(match.parseMatchEvent({ event: 'stage', data: { stage: {}, count: '6' } })).toBeNull();
    const event = match.parseMatchEvent({ event: 'result', data: {
      match_id: {}, items: [{ requirement_id: 4, requirement: [], text: {}, sections: [{ title: 'Title only' }] }],
      not_covered: {}, unchecked: Infinity, top8: 1, language: [],
    } });
    if (event?.event !== 'result') throw new Error('Expected result');
    expect(event.data).toEqual({ items: [], not_covered: [], unchecked: 0, top8: false });
  });
});

describe('match feedback', () => {
  it('does not request without a token', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(await match.sendMatchFeedback('match-1')).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([204, 429, 'network'] as const)('posts only match_id and handles %s silently', async (status) => {
    session.setToken({ token: 'session-token', expires_at: new Date(Date.now() + 3_600_000).toISOString() });
    const fetchMock = vi.fn();
    if (status === 'network') fetchMock.mockRejectedValue(new Error('Offline'));
    else fetchMock.mockResolvedValue(new Response(null, { status }));
    vi.stubGlobal('fetch', fetchMock);
    const controller = new AbortController();
    expect(await match.sendMatchFeedback('match-1', controller.signal)).toBe(status === 204);
    expect(fetchMock).toHaveBeenCalledExactlyOnceWith('/api/feedback', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer session-token' },
      body: JSON.stringify({ match_id: 'match-1' }), signal: controller.signal,
    });
    expect(session.getToken()).toBe('session-token');
  });
});

describe('sectionLabel', () => {
  const titles = { '/projects/roleradar': 'RoleRadar', constructor: 'not a page' };
  it('prefixes the title of the page the section is on', () => {
    expect(match.sectionLabel({ url: '/projects/roleradar#in-30-seconds', title: 'In 30 seconds' }, titles)).toBe('RoleRadar · In 30 seconds');
  });
  it('shows the heading alone for pages without a known title', () => {
    expect(match.sectionLabel({ url: '/projects/other#a', title: 'A' }, titles)).toBe('A');
    expect(match.sectionLabel({ url: '/projects/roleradar', title: 'B' }, {})).toBe('B');
    expect(match.sectionLabel({ url: 'toString#x', title: 'C' }, {})).toBe('C');
  });
});
