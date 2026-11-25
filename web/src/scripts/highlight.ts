// Source highlight (docs/design.md "项目页"), on every page: a link to a section marks the
// whole section, from its h2 to the next h2, for about 3 seconds. Page bodies are flat
// (h2 and paragraphs are siblings), so :target alone would only mark the heading.
// A source that carries quoted sentences (step 10, SPEC "出处标到句子") marks only those
// sentences inside that section; when none of them is found there, the whole section as before.
// Styles are in project-page.css; it is imported here so every page that loads this script
// gets them (sources also point to /about#… and /privacy#…).
import '../styles/project-page.css';
import {
  findDirective,
  findQuote,
  matchPieces,
  parseTextDirectives,
  sectionElements,
  sectionHeadingIn,
  textBlocks,
  usableQuotes,
} from '../lib/quote-text';
import type { TextDirectivePart } from '../lib/quote-text';

const MARK = 'section-highlight';
const QUOTE_MARK = 'quote-highlight';
const HOLD_MS = 3000;
// The fade at the end of the CSS animation (project-page.css), after the hold.
const FADE_MS = 600;
// A source followed to another page leaves its quotes here for the page it opens.
const PENDING_KEY = 'source-quotes';
const PENDING_MS = 30_000;

let marked: Element[] = [];
let wrappers: HTMLElement[] = [];
let timer: number | undefined;

function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** The h2 a fragment names, when it heads a section of a page body. */
function sectionHeading(fragment: string): HTMLElement | null {
  return sectionHeadingIn(document, fragment);
}

function clear(): void {
  window.clearTimeout(timer);
  for (const element of marked) element.classList.remove(MARK);
  marked = [];
  const parents = new Set<Node>();
  for (const wrapper of wrappers) {
    const parent = wrapper.parentNode;
    if (!parent) continue;
    while (wrapper.firstChild) parent.insertBefore(wrapper.firstChild, wrapper);
    parent.removeChild(wrapper);
    parents.add(parent);
  }
  for (const parent of parents) parent.normalize();
  wrappers = [];
}

/**
 * Wraps the found quotes' text in <mark> elements, one per text node it runs through, so bold
 * and links inside a sentence keep their own elements. Returns the marks, in document order.
 */
function markQuotes(section: Element[], quotes: string[], directives: readonly TextDirectivePart[]): HTMLElement[] {
  const blocks = textBlocks(section);
  const byNode = new Map<Text, { start: number; end: number }[]>();
  const nodes: Text[] = [];
  const matches = [
    ...quotes.map((quote) => findQuote(blocks, quote)),
    ...directives.map((part) => findDirective(blocks, part)),
  ];
  for (const match of matches) {
    if (!match) continue;
    for (const piece of matchPieces(match)) {
      if (!byNode.has(piece.node)) {
        byNode.set(piece.node, []);
        nodes.push(piece.node);
      }
      byNode.get(piece.node)!.push({ start: piece.start, end: piece.end });
    }
  }
  const result: HTMLElement[] = [];
  for (const node of nodes) {
    // Overlapping pieces in one node become one mark; wrap from the end so offsets stay valid.
    const pieces = byNode.get(node)!.sort((a, b) => a.start - b.start);
    const merged: { start: number; end: number }[] = [];
    for (const piece of pieces) {
      const previous = merged[merged.length - 1];
      if (previous && piece.start <= previous.end) previous.end = Math.max(previous.end, piece.end);
      else merged.push({ ...piece });
    }
    const made: HTMLElement[] = [];
    for (const { start, end } of merged.reverse()) {
      if (end < node.data.length) node.splitText(end);
      const target = start > 0 ? node.splitText(start) : node;
      const wrapper = document.createElement('mark');
      wrapper.className = QUOTE_MARK;
      target.parentNode!.insertBefore(wrapper, target);
      wrapper.appendChild(target);
      made.unshift(wrapper);
    }
    result.push(...made);
  }
  return result.sort((a, b) => (a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING ? -1 : 1));
}

/**
 * Marks the quotes found in the heading's section, or, when there are none, the heading and
 * every sibling up to the next h2; marking again restarts it. Returns the first quote marked.
 */
export function markSection(
  heading: HTMLElement,
  quotes: readonly string[] = [],
  directives: readonly TextDirectivePart[] = [],
): HTMLElement | null {
  clear();
  const section = sectionElements(heading);
  const found = markQuotes(section, usableQuotes(quotes), directives);
  // Read layout once so a second mark of the same section restarts the animation.
  void heading.offsetWidth;
  if (found.length > 0) {
    wrappers = found;
  } else {
    for (const element of section) element.classList.add(MARK);
    marked = section;
  }
  timer = window.setTimeout(clear, reducedMotion() ? HOLD_MS : HOLD_MS + FADE_MS);
  return found[0] ?? null;
}

function samePage(url: URL): boolean {
  const path = (value: string) => value.replace(/\/+$/, '') || '/';
  return url.origin === window.location.origin && path(url.pathname) === path(window.location.pathname);
}

/** The quotes a source link carries (Assistant.tsx writes them as JSON in data-quotes). */
function linkQuotes(link: HTMLAnchorElement): string[] {
  const raw = link.dataset.quotes;
  if (!raw) return [];
  try {
    const value: unknown = JSON.parse(raw);
    return Array.isArray(value) ? usableQuotes(value) : [];
  } catch {
    return [];
  }
}

function sectionId(hash: string): string {
  return hash.replace(/^#/, '').split(':~:')[0];
}

function pathOf(url: URL | Location): string {
  return url.pathname.replace(/\/+$/, '') || '/';
}

function rememberQuotes(url: URL, quotes: string[]): void {
  try {
    if (quotes.length === 0) sessionStorage.removeItem(PENDING_KEY);
    else sessionStorage.setItem(PENDING_KEY, JSON.stringify({ path: pathOf(url), id: sectionId(url.hash), quotes, at: Date.now() }));
  } catch {
    // Without storage the page it opens marks the whole section, as before.
  }
}

/** Quotes a source on the previous page left for this section; read once. */
function pendingQuotes(id: string): string[] {
  try {
    const raw = sessionStorage.getItem(PENDING_KEY);
    if (!raw) return [];
    sessionStorage.removeItem(PENDING_KEY);
    const value: unknown = JSON.parse(raw);
    if (typeof value !== 'object' || value === null) return [];
    const { path, id: pendingId, quotes, at } = value as Record<string, unknown>;
    if (path !== pathOf(window.location) || pendingId !== id || typeof at !== 'number'
      || Date.now() - at > PENDING_MS || !Array.isArray(quotes)) return [];
    return usableQuotes(quotes);
  } catch {
    return [];
  }
}

document.addEventListener('click', (event) => {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const link = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null;
  if (!link || (link.target && link.target !== '_self')) return;
  // Contents lists and the "#" beside each heading are plain jumps, not sources.
  if (link.closest('nav') || link.classList.contains('anchor')) return;
  const url = new URL(link.href, window.location.href);
  if (!samePage(url)) {
    if (url.origin === window.location.origin) rememberQuotes(url, linkQuotes(link));
    return;
  }
  const heading = sectionHeading(url.hash.slice(1));
  if (!heading) return;

  event.preventDefault();
  // A source inside the phone's ask dialog: close the dialog first, then scroll the page.
  const dialog = link.closest('dialog');
  if (dialog?.open && dialog.matches(':modal')) dialog.close();
  // The page's own address keeps only the section; the quotes are marked below, not by the browser.
  const hash = `#${sectionId(url.hash)}`;
  if (window.location.hash !== hash) window.history.pushState(null, '', hash);
  const quote = markSection(heading, linkQuotes(link));
  const behavior = reducedMotion() ? 'auto' : 'smooth';
  if (quote) quote.scrollIntoView({ behavior, block: 'center' });
  else heading.scrollIntoView({ behavior, block: 'start' });
  if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1');
  heading.focus({ preventScroll: true });
});

/**
 * The Text Fragment this page was opened with (a source link opened in a new tab). The browser
 * strips it from `location`, but the navigation entry keeps the address as it was opened.
 */
function arrivalDirectives(id: string): TextDirectivePart[] {
  try {
    const entry = performance.getEntriesByType('navigation')[0];
    if (!entry) return [];
    const url = new URL(entry.name);
    return pathOf(url) === pathOf(window.location) && sectionId(url.hash) === id ? parseTextDirectives(url.hash) : [];
  } catch {
    return [];
  }
}

/**
 * Whether the browser marks this arrival itself: it supports Text Fragments and the page was
 * opened with one that is in the section. The browser keeps its marks until the visitor
 * clicks; splitting and joining the text nodes under them for our own marks lost all but its
 * first range when ours ended (10v), so then the page leaves the marking to the browser.
 */
export function browserMarks(heading: HTMLElement, directives: readonly TextDirectivePart[], doc: Document = document): boolean {
  if (directives.length === 0 || !('fragmentDirective' in doc)) return false;
  const blocks = textBlocks(sectionElements(heading));
  return directives.some((part) => findDirective(blocks, part) !== null);
}

function markFromLocation(arriving: boolean): void {
  const id = sectionId(window.location.hash);
  const heading = sectionHeading(id);
  if (!heading) return;
  const quotes = pendingQuotes(id);
  const directives = arriving ? arrivalDirectives(id) : [];
  if (browserMarks(heading, directives)) return;
  markSection(heading, quotes, directives);
}

// Arriving from another page (/projects/x#section-id), and Back or Forward between sources.
window.addEventListener('hashchange', () => markFromLocation(false));
markFromLocation(true);
