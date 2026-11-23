// The examples page lists numbered questions, each ending in "(Page: Section)".
// Home shows items 1, 2 and 4; a project page shows the items whose page name is its slug.
const TOPIC = /\s*\(([^()]*)\)\s*$/;

function question(text: string): string {
  return text.replace(TOPIC, '');
}

/** The page name in an item's trailing "(Page: Section)", lowercased, or undefined. */
function pageOf(text: string): string | undefined {
  return TOPIC.exec(text)?.[1]?.split(':')[0]?.trim().toLowerCase();
}

export function parseExamples(markdown: string, page?: string): string[] {
  const matches = markdown.split(/\r?\n/).flatMap((line) => {
    const match = /^\s*(\d+)\.\s+(.+)$/.exec(line);
    return match ? [match] : [];
  });
  if (matches.length < 4) {
    throw new Error('Examples must contain at least 4 numbered items, including items 1, 2 and 4');
  }

  const items = new Map(matches.map((match) => [Number(match[1]), match[2]!]));

  if (page !== undefined) {
    const slug = page.toLowerCase();
    return [...items.entries()]
      .sort(([a], [b]) => a - b)
      .filter(([, text]) => pageOf(text) === slug)
      .map(([, text]) => question(text));
  }

  return [1, 2, 4].map((number) => {
    const text = items.get(number);
    if (text === undefined) {
      throw new Error('Examples must include items 1, 2 and 4');
    }
    return question(text);
  });
}
