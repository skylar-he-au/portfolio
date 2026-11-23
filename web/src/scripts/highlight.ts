// Source highlight (docs/design.md "项目页"), on every page: a link to a section marks the
// whole section, from its h2 to the next h2, for about 3 seconds. Page bodies are flat
// (h2 and paragraphs are siblings), so :target alone would only mark the heading.
// Styles are in project-page.css; it is imported here so every page that loads this script
// gets them (sources also point to /about#… and /privacy#…).
import '../styles/project-page.css';

const MARK = 'section-highlight';
const HOLD_MS = 3000;
// The fade at the end of the CSS animation (project-page.css), after the hold.
const FADE_MS = 600;

let marked: Element[] = [];
let timer: number | undefined;

function reducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** The h2 a fragment names, when it heads a section of a page body. */
function sectionHeading(fragment: string): HTMLElement | null {
  if (!fragment) return null;
  let id: string;
  try {
    id = decodeURIComponent(fragment);
  } catch {
    return null;
  }
  const heading = document.getElementById(id);
  return heading?.tagName === 'H2' && heading.closest('.markdown') ? heading : null;
}

function clear(): void {
  window.clearTimeout(timer);
  for (const element of marked) element.classList.remove(MARK);
  marked = [];
}

/** Marks the heading and every sibling up to the next h2; marking again restarts it. */
export function markSection(heading: HTMLElement): void {
  clear();
  const section: Element[] = [heading];
  for (let next = heading.nextElementSibling; next && next.tagName !== 'H2'; next = next.nextElementSibling) {
    section.push(next);
  }
  // Read layout once so a second mark of the same section restarts the animation.
  void heading.offsetWidth;
  for (const element of section) element.classList.add(MARK);
  marked = section;
  timer = window.setTimeout(clear, reducedMotion() ? HOLD_MS : HOLD_MS + FADE_MS);
}

function samePage(url: URL): boolean {
  const path = (value: string) => value.replace(/\/+$/, '') || '/';
  return url.origin === window.location.origin && path(url.pathname) === path(window.location.pathname);
}

document.addEventListener('click', (event) => {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const link = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>('a[href]') : null;
  if (!link || (link.target && link.target !== '_self')) return;
  // Contents lists and the "#" beside each heading are plain jumps, not sources.
  if (link.closest('nav') || link.classList.contains('anchor')) return;
  const url = new URL(link.href, window.location.href);
  if (!samePage(url)) return;
  const heading = sectionHeading(url.hash.slice(1));
  if (!heading) return;

  event.preventDefault();
  // A source inside the phone's ask dialog: close the dialog first, then scroll the page.
  const dialog = link.closest('dialog');
  if (dialog?.open && dialog.matches(':modal')) dialog.close();
  if (window.location.hash !== url.hash) window.history.pushState(null, '', url.hash);
  heading.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' });
  if (!heading.hasAttribute('tabindex')) heading.setAttribute('tabindex', '-1');
  heading.focus({ preventScroll: true });
  markSection(heading);
});

function markFromLocation(): void {
  const heading = sectionHeading(window.location.hash.slice(1));
  if (heading) markSection(heading);
}

// Arriving from another page (/projects/x#section-id), and Back or Forward between sources.
window.addEventListener('hashchange', markFromLocation);
markFromLocation();
