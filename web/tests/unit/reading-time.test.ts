import { describe, expect, it } from 'vitest';
import { readingMinutes } from '../../src/lib/reading-time';

describe('readingMinutes', () => {
  it('rounds up at 200 words and returns at least one minute', () => {
    expect(readingMinutes('')).toBe(1);
    expect(readingMinutes('word '.repeat(200))).toBe(1);
    expect(readingMinutes('word '.repeat(201))).toBe(2);
    expect(readingMinutes('word '.repeat(401))).toBe(3);
  });

  it('drops backtick and tilde fences and multiline HTML comments', () => {
    const code = 'word '.repeat(500);
    expect(readingMinutes(`Hello\n\n\`\`\`js\n${code}\n\`\`\`\n~~~text\n${code}\n~~~\n<!--\n${code}\n-->`)).toBe(1);
  });

  it('counts prose and link labels without Markdown markers or link destinations', () => {
    expect(readingMinutes(`${'word '.repeat(197)}\n## **bold** _text_ [label](https://example.com/long/path)\n---\n> *`)).toBe(1);
    expect(readingMinutes(`${'word '.repeat(200)}\n## Last`)).toBe(2);
  });

  it('keeps shorter or mismatched fences inside code and ignores an unclosed fence', () => {
    expect(readingMinutes(`Hello\n\`\`\`\`js\n\`\`\`\n~~~\n${'code '.repeat(500)}\n\`\`\`\``)).toBe(1);
    expect(readingMinutes(`Hello\n~~~\n${'code '.repeat(500)}`)).toBe(1);
  });
});
