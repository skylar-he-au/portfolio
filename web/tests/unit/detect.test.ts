import { join } from 'node:path';
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { ASK_MAX_CHARS, BAD_LINK_MESSAGE, detectInput } from '../../src/lib/detect';

const links = [
  'https://boards.greenhouse.io/acme/jobs/1234567',
  'https://job-boards.greenhouse.io/acme/jobs/1234567?gh_src=x',
  'https://job-boards.anz.greenhouse.io/acme/jobs/1234567',
  'https://jobs.lever.co/acme/123e4567-e89b-12d3-a456-426614174000',
  'https://jobs.eu.lever.co/acme/123e4567-e89b-12d3-a456-426614174000/apply',
];

// Translate only the constructs used by these Python raw f-string patterns.
function backendPatterns(): RegExp[] {
  const source = readFileSync(join(process.cwd(), '../api/app/match/fetch.py'), 'utf8');
  const constants: Record<string, string> = {};
  for (const name of ['_BOARD', '_GREENHOUSE_ID', '_LEVER_ID', '_SUFFIX']) {
    const value = source.match(new RegExp(`^${name} = r"([^"\\n]+)"$`, 'm'))?.[1];
    if (!value) throw new Error(`Cannot parse backend constant ${name}`);
    constants[name] = value;
  }
  return ['_GREENHOUSE_URL', '_LEVER_URL'].map((name) => {
    const block = source.match(new RegExp(`${name} = re\\.compile\\(\\n([\\s\\S]*?)\\n\\)`))?.[1];
    if (!block) throw new Error(`Cannot parse backend pattern ${name}`);
    const parts = [...block.matchAll(/r(f)?"([^"\n]*)"/g)];
    if (!parts.length) throw new Error(`No raw strings in ${name}`);
    const pattern = parts.map((part) => {
      const value = part[2]!;
      return part[1] ? value.replace(/\{(_[A-Z_]+)\}/g, (_, key: string) => {
        if (constants[key] === undefined) throw new Error(`Unknown backend constant ${key}`);
        return constants[key];
      }) : value;
    }).join('')
      .replace(/\(\?ai:([^)]*)\)/g, (_, host: string) => `(?:${host.replace(/[a-zA-Z]/g, (letter) => `[${letter.toLowerCase()}${letter.toUpperCase()}]`)})`)
      .replace(/\(\?P<\w+>/g, '(?:');
    return new RegExp(`^(?:${pattern})$`, 'u');
  });
}

describe('detectInput', () => {
  it.each(links)('recognizes %s', (link) => {
    expect(detectInput(link)).toEqual({ kind: 'match', source: 'link' });
  });

  it.each(['https://boards.greenhouse.io/acme', 'https://www.seek.com.au/job/12345678', 'http://boards.greenhouse.io/acme/jobs/1'])('rejects a lone unreadable URL %s', (link) => {
    expect(detectInput(link)).toEqual({ kind: 'bad-link' });
  });

  it('exports the input limit and fixed message', () => {
    expect(ASK_MAX_CHARS).toBe(500);
    expect(BAD_LINK_MESSAGE).toBe('Only Greenhouse and Lever job links can be read — paste the job ad text instead.');
  });

  it.each(['', ' \n ', 'q'.repeat(300), 'q'.repeat(500), ` \n${'q'.repeat(500)} \n`, '😀'.repeat(500), `${links[0]} ${links[1]}`, `${links[0]} please explain`])('treats short mixed text as a question', (text) => {
    expect(detectInput(text)).toEqual({ kind: 'ask' });
  });

  it.each(['q'.repeat(501), '😀'.repeat(501)])('counts trimmed code points', (text) => {
    expect(detectInput(text)).toEqual({ kind: 'match', source: 'text' });
  });

  it('trims job links', () => {
    expect(detectInput(` \n${links[0]} \n`)).toEqual({ kind: 'match', source: 'link' });
  });

  it('matches the regexes read from the backend source', () => {
    const patterns = backendPatterns();
    const samples = [
      ...links,
      'https://seek.com.au/job/123', 'http://boards.greenhouse.io/acme/jobs/123',
      'https://boards.greenhouse.io/acme', 'https://boards.greenhouse.io/acme/123',
      'https://boards.greenhouse.io/acme/jobs/nope', 'https://boards.greenhouse.io/acme/jobs/123/extra',
      'https://boards.greenhouse.io/acme/jobs/123/', 'https://boards.greenhouse.io/acme/jobs/123?x=y#top',
      'https://BOARDS.GREENHOUSE.IO/Acme_A-B/jobs/123', 'https://JOB-BOARDS.ANZ.GREENHOUSE.IO/a/jobs/1',
      'https://JOBS.EU.LEVER.CO/Acme_A-B/123E4567-E89B-12D3-A456-426614174000/apply/?x=y#top',
      'https://jobs.lever.co/acme/123e4567-e89b-12d3-a456-426614174000/APPLY',
      'https://boards.greenhouse.io/acme/JOBS/123', 'https://boards.greenhouse.io/acme/jobs/123?x=\u2003y',
      'https://boards.greenhouse.io/acme/jobs/123#top#again', ` \n${links[0]} \n`,
      'https://boards.greenhouse.io/acme/jobs/123 extra', 'HTTPS://boards.greenhouse.io/acme/jobs/123',
    ];
    for (const sample of samples) {
      const detected = detectInput(sample);
      expect(detected.kind === 'match' && detected.source === 'link', sample)
        .toBe(patterns.some((pattern) => pattern.test(sample.trim())));
    }
  });
});
