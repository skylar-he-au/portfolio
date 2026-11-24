// The browser half of the plain-text rule for quoted sentences (docs/briefs/step10.md, "出处的格式").
// The API sends each source with `quotes`: sentences it found word for word in one block of the
// section's plain text, after folding typography (api/app/qa/output.py `fold`, `find_quote`).
// Pages are rendered with smartypants (curly quotes, dashes), so the page text is folded the same
// way here before searching, and offsets are mapped back to the page's own text nodes.

export const MIN_QUOTE_CHARS = 12;
export const MAX_QUOTES = 5;
// A text directive longer than this many words is written as `start,end`.
const LONG_WORDS = 12;
const EDGE_WORDS = 5;

const TYPOGRAPHY: Record<string, string> = {
  '‘': "'",
  '’': "'",
  '“': '"',
  '”': '"',
  '—': '--',
  '–': '-',
  '…': '...',
  ' ': ' ',
};

// Blocks the API's plain text keeps apart: a quote never runs from one into the next.
const BLOCK = 'p, li, h1, h2, h3, h4, h5, h6, td, th, dt, dd, pre, blockquote, figcaption, caption';
// Text the API's plain text does not have: the "#" beside each heading, client-rendered islands.
const SKIP = '.anchor, astro-island, script, style, template, noscript';

/** Typography folded to ASCII and whitespace collapsed to one space (same as the API's `fold`). */
export function fold(text: string): string {
  return foldWithMap(text).text;
}

/** The folded text, and for each of its characters the index of the raw character it came from. */
export function foldWithMap(raw: string): { text: string; origin: number[] } {
  let text = '';
  const origin: number[] = [];
  let space = false;
  for (let index = 0; index < raw.length; index += 1) {
    const char = raw[index];
    const folded = TYPOGRAPHY[char] ?? char;
    if (/\s/.test(folded)) {
      space = text.length > 0;
      continue;
    }
    if (space) {
      text += ' ';
      origin.push(index);
      space = false;
    }
    for (const piece of folded) {
      text += piece;
      origin.push(index);
    }
  }
  return { text, origin };
}

export interface TextBlock {
  /** The block's text nodes, in document order, and where each starts in `raw`. */
  nodes: Text[];
  starts: number[];
  raw: string;
  folded: string;
  origin: number[];
}

/** The text of every block under `roots`, in document order; text outside any block is its own block. */
export function textBlocks(roots: Iterable<Element>): TextBlock[] {
  const blocks: TextBlock[] = [];
  const byElement = new Map<Element, TextBlock>();
  for (const root of roots) {
    const doc = root.ownerDocument;
    const walker = doc.createTreeWalker(root, 4 /* NodeFilter.SHOW_TEXT */);
    for (let node = walker.nextNode() as Text | null; node; node = walker.nextNode() as Text | null) {
      const parent = node.parentElement;
      if (!parent || parent.closest(SKIP)) continue;
      let owner = parent.closest(BLOCK);
      if (!owner || !root.contains(owner)) owner = root.contains(parent) ? parent : root;
      let block = byElement.get(owner);
      if (!block) {
        block = { nodes: [], starts: [], raw: '', folded: '', origin: [] };
        byElement.set(owner, block);
        blocks.push(block);
      }
      block.starts.push(block.raw.length);
      block.nodes.push(node);
      block.raw += node.data;
    }
  }
  for (const block of blocks) {
    const { text, origin } = foldWithMap(block.raw);
    block.folded = text;
    block.origin = origin;
  }
  return blocks;
}

/** Quotes worth looking for: strings, long enough, at most MAX_QUOTES, no repeats. */
export function usableQuotes(quotes: readonly unknown[] | undefined): string[] {
  if (!Array.isArray(quotes)) return [];
  const result: string[] = [];
  for (const quote of quotes) {
    if (typeof quote !== 'string') continue;
    const folded = fold(quote);
    if (folded.length >= MIN_QUOTE_CHARS && !result.includes(folded)) result.push(folded);
    if (result.length === MAX_QUOTES) break;
  }
  return result;
}

export interface QuoteMatch {
  block: TextBlock;
  /** Raw offsets in `block.raw`, end exclusive. */
  start: number;
  end: number;
}

function rawSpan(block: TextBlock, foldedStart: number, foldedLength: number): QuoteMatch {
  const start = block.origin[foldedStart];
  const last = block.origin[foldedStart + foldedLength - 1];
  // A folded "--" or "..." comes from one raw character.
  return { block, start, end: last + 1 };
}

/** The first place the quote appears, word for word, inside one block. */
export function findQuote(blocks: readonly TextBlock[], quote: string): QuoteMatch | null {
  const folded = fold(quote);
  if (folded.length < MIN_QUOTE_CHARS) return null;
  for (const block of blocks) {
    const at = block.folded.indexOf(folded);
    if (at >= 0) return rawSpan(block, at, folded.length);
  }
  return null;
}

function occurrences(haystack: string, needle: string): number {
  let count = 0;
  for (let at = haystack.indexOf(needle); at >= 0; at = haystack.indexOf(needle, at + 1)) count += 1;
  return count;
}

/** How often the quote appears in the blocks, ignoring case as a browser's text search does. */
export function countQuote(blocks: readonly TextBlock[], quote: string): number {
  const folded = fold(quote).toLowerCase();
  if (!folded) return 0;
  return blocks.reduce((total, block) => total + occurrences(block.folded.toLowerCase(), folded), 0);
}

/**
 * Each text node a match runs through, with the part of it to mark: across bold and links the
 * marks go inside those elements, which stay as they are.
 */
export function matchPieces(match: QuoteMatch): { node: Text; start: number; end: number }[] {
  const pieces: { node: Text; start: number; end: number }[] = [];
  match.block.nodes.forEach((node, index) => {
    const nodeStart = match.block.starts[index];
    const start = Math.max(match.start, nodeStart);
    const end = Math.min(match.end, nodeStart + node.data.length);
    if (start < end) pieces.push({ node, start: start - nodeStart, end: end - nodeStart });
  });
  return pieces;
}

const WORD = /[\p{L}\p{N}]/u;

function encode(text: string): string {
  // Text directives reserve "-", "," and "&" (encodeURIComponent leaves "-" alone).
  return encodeURIComponent(text).replace(/-/g, '%2D');
}

/**
 * One `text=` directive for a quote, in the page's own writing (curly quotes, dashes), or null
 * when the browser could mark a different place: the quote appears more than once on the page,
 * or starts or ends inside a word. `pageBlocks` is the whole page; `sectionBlocks` the cited section.
 */
export function textDirective(
  pageBlocks: readonly TextBlock[],
  sectionBlocks: readonly TextBlock[],
  quote: string,
): string | null {
  if (countQuote(pageBlocks, quote) !== 1) return null;
  const match = findQuote(sectionBlocks, quote);
  if (!match) return null;
  const { raw } = match.block;
  if ((match.start > 0 && WORD.test(raw[match.start - 1]) && WORD.test(raw[match.start]))
    || (match.end < raw.length && WORD.test(raw[match.end]) && WORD.test(raw[match.end - 1]))) return null;
  const pageText = raw.slice(match.start, match.end).replace(/\s+/g, ' ').trim();
  const words = pageText.split(' ');
  if (words.length > LONG_WORDS) {
    const start = words.slice(0, EDGE_WORDS).join(' ');
    const end = words.slice(-EDGE_WORDS).join(' ');
    // The browser marks from the first `start` on the page to the first `end` after it.
    const block = fold(match.block.raw.slice(match.start)).toLowerCase();
    const endAt = block.indexOf(fold(end).toLowerCase(), fold(start).length);
    if (countQuote(pageBlocks, start) === 1 && endAt === fold(pageText).length - fold(end).length) {
      return `text=${encode(start)},${encode(end)}`;
    }
  }
  return `text=${encode(pageText)}`;
}

/** The `:~:text=…&text=…` part for a source's quotes, or '' when none can be written safely. */
export function fragmentDirective(
  pageBlocks: readonly TextBlock[],
  sectionBlocks: readonly TextBlock[],
  quotes: readonly string[],
): string {
  const directives = usableQuotes(quotes)
    .map((quote) => textDirective(pageBlocks, sectionBlocks, quote))
    .filter((directive): directive is string => directive !== null);
  return directives.length ? `:~:${directives.join('&')}` : '';
}

export interface TextDirectivePart {
  start: string;
  end?: string;
}

/** The `text=start[,end]` parts of a URL fragment (`#id:~:text=…&text=…`); prefixes and suffixes are not used here. */
export function parseTextDirectives(fragment: string): TextDirectivePart[] {
  const at = fragment.indexOf(':~:');
  if (at < 0) return [];
  const parts: TextDirectivePart[] = [];
  for (const directive of fragment.slice(at + 3).split('&')) {
    if (!directive.startsWith('text=')) continue;
    const terms = directive.slice(5).split(',');
    if (terms.length > 2 || terms.some((term) => term.endsWith('-') || term.startsWith('-'))) continue;
    try {
      const [start, end] = terms.map((term) => decodeURIComponent(term));
      if (start) parts.push(end ? { start, end } : { start });
    } catch {
      // A malformed directive is skipped; the others still apply.
    }
  }
  return parts;
}

/** Where a `text=start[,end]` directive falls inside one block, ignoring case as the browser does. */
export function findDirective(blocks: readonly TextBlock[], part: TextDirectivePart): QuoteMatch | null {
  const start = fold(part.start).toLowerCase();
  const end = part.end === undefined ? undefined : fold(part.end).toLowerCase();
  if (!start) return null;
  for (const block of blocks) {
    const text = block.folded.toLowerCase();
    for (let at = text.indexOf(start); at >= 0; at = text.indexOf(start, at + 1)) {
      if (end === undefined) return rawSpan(block, at, start.length);
      const endAt = text.indexOf(end, at + start.length);
      if (endAt >= 0) return rawSpan(block, at, endAt + end.length - at);
    }
  }
  return null;
}

/** The h2 a fragment names when it heads a section of a page body (the part before any `:~:`). */
export function sectionHeadingIn(doc: Document, fragment: string): HTMLElement | null {
  const name = fragment.split(':~:')[0];
  if (!name) return null;
  let id: string;
  try {
    id = decodeURIComponent(name);
  } catch {
    return null;
  }
  const heading = doc.getElementById(id);
  return heading?.tagName === 'H2' && heading.closest('.markdown') ? heading : null;
}

/** The heading and every sibling up to the next h2. */
export function sectionElements(heading: Element): Element[] {
  const section: Element[] = [heading];
  for (let next = heading.nextElementSibling; next && next.tagName !== 'H2'; next = next.nextElementSibling) {
    section.push(next);
  }
  return section;
}
