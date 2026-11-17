export function parseExamples(markdown: string): string[] {
  const matches = markdown.split(/\r?\n/).flatMap((line) => {
    const match = /^\s*(\d+)\.\s+(.+)$/.exec(line);
    return match ? [match] : [];
  });
  if (matches.length < 4) {
    throw new Error('Examples must contain at least 4 numbered items, including items 1, 2 and 4');
  }

  const items = new Map(matches.map((match) => [Number(match[1]), match[2]!]));
  return [1, 2, 4].map((number) => {
    const text = items.get(number);
    if (text === undefined) {
      throw new Error('Examples must include items 1, 2 and 4');
    }
    return text.replace(/\s*\([^()]*\)\s*$/, '');
  });
}
