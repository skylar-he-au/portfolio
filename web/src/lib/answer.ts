import { usableQuotes } from './quote-text';

export interface Citation {
  n: number;
  url: string;
  heading: string;
  /** Sentences the API found word for word in that section (step 10); absent in older answers. */
  quotes?: string[];
}

export type InlineSegment =
  | { type: 'text'; text: string }
  | { type: 'link'; text: string; href: string; title: string; quotes: string[] };

/** Per source number, the `:~:text=…` part its link carries (see quote-text.ts `fragmentDirective`). */
export type QuoteFragments = ReadonlyMap<number, string>;

/** Citations from an `answer` event: only the expected fields, and quotes that are strings. */
export function citationList(value: unknown): Citation[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item: unknown) => {
    if (typeof item !== 'object' || item === null) return [];
    const { n, url, heading, quotes } = item as Record<string, unknown>;
    if (typeof n !== 'number' || !Number.isFinite(n) || typeof url !== 'string' || typeof heading !== 'string') return [];
    return [{ n, url, heading, quotes: Array.isArray(quotes) ? usableQuotes(quotes) : [] }];
  });
}

/**
 * A source's link: the site URL, plus a Text Fragment when its quotes appear only once on that
 * page, so a new tab marks them too. Null for anything but a site address.
 */
export function citationHref(citation: Citation, fragments?: QuoteFragments): string | null {
  const href = safeSiteUrl(citation.url);
  if (!href) return null;
  const directive = fragments?.get(citation.n);
  if (!directive || !directive.startsWith(':~:text=')) return href;
  return `${href}${href.includes('#') ? '' : '#'}${directive}`;
}

export type AnswerSegment = InlineSegment | { type: 'unsupported'; children: InlineSegment[] };

export function safeSiteUrl(url: string): string | null {
  if (!url.startsWith('/') || url[1] === '/' || /[\\\s\u0000-\u001f\u007f-\u009f]/.test(url)) return null;
  try {
    return new URL(url, location.origin).origin === location.origin ? url : null;
  } catch {
    return null;
  }
}

export function unreferencedCitations(text: string, citations: Citation[]): Citation[] {
  return citations.filter(({ n, url }) => !text.includes(`[${n}]`) && safeSiteUrl(url) !== null);
}

export function answerSegments(
  text: string,
  citations: Citation[],
  unsupported: string[],
  fragments?: QuoteFragments,
): AnswerSegment[] {
  function linkify(piece: string): InlineSegment[] {
    const result: InlineSegment[] = [];
    let cursor = 0;
    for (const match of piece.matchAll(/\[(\d+)\]/g)) {
      const citation = citations.find(({ n }) => n === Number(match[1]));
      const href = citation ? citationHref(citation, fragments) : null;
      if (!citation || !href) continue;
      if (match.index > cursor) result.push({ type: 'text', text: piece.slice(cursor, match.index) });
      result.push({ type: 'link', text: match[0], href, title: citation.heading, quotes: usableQuotes(citation.quotes) });
      cursor = match.index + match[0].length;
    }
    if (cursor < piece.length) result.push({ type: 'text', text: piece.slice(cursor) });
    return result;
  }

  const ranges: { start: number; end: number }[] = [];
  for (const span of unsupported) {
    if (!span) continue;
    let start = text.indexOf(span);
    while (start >= 0) {
      ranges.push({ start, end: start + span.length });
      start = text.indexOf(span, start + 1);
    }
  }
  ranges.sort((a, b) => a.start - b.start);
  const merged: typeof ranges = [];
  for (const range of ranges) {
    const previous = merged[merged.length - 1];
    if (previous && range.start <= previous.end) previous.end = Math.max(previous.end, range.end);
    else merged.push({ ...range });
  }

  const result: AnswerSegment[] = [];
  let cursor = 0;
  for (const { start, end } of merged) {
    result.push(...linkify(text.slice(cursor, start)));
    result.push({ type: 'unsupported', children: linkify(text.slice(start, end)) });
    cursor = end;
  }
  result.push(...linkify(text.slice(cursor)));
  return result;
}
