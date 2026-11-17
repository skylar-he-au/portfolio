import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import GithubSlugger from 'github-slugger';
import { describe, expect, it } from 'vitest';

const root = process.cwd();
const dist = join(root, 'dist');
if (!existsSync(dist)) throw new Error('Missing dist/: run `npm run build` first');

const pages = {
  '/projects/roleradar': 'projects/roleradar.md',
  '/projects/oncallops': 'projects/oncallops.md',
  '/about': 'about.md',
  '/privacy': 'privacy.md',
  '/': 'home.md',
} as const;

interface Section {
  heading: string;
  id: string;
}

const fixtures = JSON.parse(
  readFileSync(join(root, 'tests/fixtures/section-ids.json'), 'utf8'),
) as Record<string, Section[]>;

function filesWithin(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesWithin(path) : [path];
  });
}

function readPage(path: string): Document {
  return new DOMParser().parseFromString(
    readFileSync(join(dist, path.slice(1), 'index.html'), 'utf8'),
    'text/html',
  );
}

function frontMatter(markdown: string, field: 'title' | 'summary'): string {
  const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(markdown)?.[1];
  // These content files use single-line plain scalars for title and summary.
  const value = front?.match(new RegExp(`^${field}: (.+)$`, 'm'))?.[1]?.trim();
  if (!value) throw new Error(`Missing front matter field: ${field}`);
  return value;
}

function expectLocalResource(value: string, context: string, allowData = true): void {
  const url = value.trim();
  if (allowData && /^data:/i.test(url)) return;
  expect(url, context).not.toMatch(/^(?:[a-z][a-z\d+.-]*:|\/\/)/i);
}

function checkCss(css: string, context: string): void {
  for (const match of css.matchAll(/url\(\s*(?:"([^"]*)"|'([^']*)'|([^)]*?))\s*\)/gi)) {
    expectLocalResource(match[1] ?? match[2] ?? match[3] ?? '', context);
  }
}

function srcsetUrls(value: string): string[] {
  const urls: string[] = [];
  let remaining = value;
  // URL tokens may contain commas (data URIs); descriptors end at the next comma.
  while ((remaining = remaining.replace(/^[\s,]+/, ''))) {
    const token = /^\S+/.exec(remaining)![0];
    urls.push(token.replace(/,+$/, ''));
    remaining = remaining.slice(token.length);
    if (!token.endsWith(',')) remaining = remaining.replace(/^[^,]*(?:,|$)/, '');
  }
  return urls;
}

it('builds exactly the five public HTML pages and excludes examples', () => {
  expect(filesWithin(dist).filter((path) => path.endsWith('.html'))
    .map((path) => relative(dist, path)).sort()).toEqual([
    'about/index.html',
    'index.html',
    'privacy/index.html',
    'projects/oncallops/index.html',
    'projects/roleradar/index.html',
  ]);
  expect(existsSync(join(dist, 'examples/index.html'))).toBe(false);
});

it('pins all 32 sections in document order', () => {
  expect(Object.keys(fixtures)).toEqual(Object.keys(pages));
  expect(Object.values(fixtures).map((sections) => sections.length)).toEqual([9, 9, 5, 9, 0]);
  expect(Object.values(fixtures).flat()).toHaveLength(32);
});

describe.each(Object.entries(pages))('%s', (path, source) => {
  const markdown = readFileSync(join(root, '../content', source), 'utf8');
  const document = readPage(path);

  it('uses content metadata and renders the body after its h1 inside main', () => {
    const title = frontMatter(markdown, 'title');
    expect(document.documentElement.lang).toBe('en');
    expect(document.title).toBe(path === '/' ? 'Skylar He' : `${title} · Skylar He`);
    if (path === '/about') expect(document.title).toBe('About Skylar · Skylar He');
    expect(document.querySelector('meta[name="description"]')?.getAttribute('content'))
      .toBe(frontMatter(markdown, 'summary'));
    expect(document.querySelector('meta[charset]')?.getAttribute('charset')).toBe('utf-8');
    expect(document.querySelector('meta[name="viewport"]')).not.toBeNull();
    expect(document.querySelectorAll('main')).toHaveLength(1);
    expect(document.querySelectorAll('h1')).toHaveLength(1);
    const main = document.querySelector('main#main')!;
    expect(main.firstElementChild?.tagName).toBe('H1');
    expect(main.querySelector('h1')?.textContent).toBe(title);
    if (path === '/') expect(main.querySelector('h1')?.textContent).toBe('Skylar He');
    expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);
    expect(main.querySelector('h1')?.nextElementSibling?.classList.contains('markdown')).toBe(true);
  });

  it('preserves the fixture headings, Astro ids, and accessible section anchors', () => {
    const sections = fixtures[path]!;
    const headings = Array.from(document.querySelectorAll('.markdown h2'));
    expect(headings.map((heading) => heading.id)).toEqual(sections.map((section) => section.id));
    expect(Array.from(markdown.matchAll(/^## (.+)$/gm), (match) => match[1]))
      .toEqual(sections.map((section) => section.heading));
    const slugger = new GithubSlugger();
    for (const [index, section] of sections.entries()) {
      expect(slugger.slug(section.heading)).toBe(section.id);
      const heading = headings[index]!;
      const anchors = heading.querySelectorAll(':scope > a.anchor');
      expect(anchors).toHaveLength(1);
      const anchor = anchors[0]!;
      expect(anchor.getAttribute('href')).toBe(`#${section.id}`);
      expect(anchor.getAttribute('aria-label')).toBe('Link to this section');
      expect(anchor.textContent).toBe('#');
      const text = heading.cloneNode(true) as HTMLElement;
      text.querySelector('a.anchor')!.remove();
      expect(text.textContent?.trim()).toBe(section.heading);
    }
    expect(document.querySelectorAll('h1 .anchor, h3 .anchor, h4 .anchor, h5 .anchor, h6 .anchor'))
      .toHaveLength(0);
  });

  it('provides the shared navigation, footer, and first-focusable skip link', () => {
    expect(document.querySelector('header a[href="/"]')?.textContent).toBe('Skylar He');
    expect(document.querySelector('header a[href="/#projects"]')?.textContent).toBe('Projects');
    expect(document.querySelector('header a[href="/about"]')?.textContent).toBe('About');
    expect(document.querySelector('footer a[href="/privacy"]')?.textContent).toBe('Privacy');
    expect(document.querySelector('footer')?.textContent).toContain('© 2026 Skylar He');
    const firstFocusable = document.querySelector(
      'a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    expect(firstFocusable?.getAttribute('href')).toBe('#main');
    expect(firstFocusable?.textContent).toBe('Skip to content');
  });
});

it('renders two project cards followed by the Ask component on Home', () => {
  const document = readPage('/');
  const projects = document.querySelector('section#projects')!;
  const headings = projects.querySelectorAll('h2');
  expect(headings).toHaveLength(1);
  const heading = headings[0]!;
  expect(heading.textContent).toBe('Projects');
  expect(heading.querySelector('a.anchor')).toBeNull();
  const grid = projects.querySelector('.project-grid')!;
  expect(grid).not.toBeNull();
  expect(heading.compareDocumentPosition(grid) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(Array.from(projects.querySelectorAll('a'), (link) => link.getAttribute('href')))
    .toEqual(['/projects/roleradar', '/projects/oncallops']);
  const cards = projects.querySelectorAll('article');
  expect(cards).toHaveLength(2);
  expect(heading.compareDocumentPosition(cards[0]!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  for (const [index, file] of ['projects/roleradar.md', 'projects/oncallops.md'].entries()) {
    const markdown = readFileSync(join(root, '../content', file), 'utf8');
    expect(cards[index]!.firstElementChild?.tagName).toBe('H3');
    expect(cards[index]!.querySelector('h3')?.textContent).toBe(frontMatter(markdown, 'title'));
    expect(cards[index]!.querySelector('p')?.textContent).toBe(frontMatter(markdown, 'summary'));
  }
  const ask = document.querySelector('section#ask[aria-label="Ask about my projects"]')!;
  expect(document.querySelectorAll('section#ask')).toHaveLength(1);
  expect(projects.nextElementSibling).toBe(ask);
  // The AskPlaceholder component is server-rendered into the page.
  const field = ask.querySelector('textarea, input[type="text"]')!;
  expect(field).not.toBeNull();
  expect(ask.querySelector(`label[for="${field.id}"]`)).not.toBeNull();
  expect(ask.querySelector('a[href="/privacy"]')).not.toBeNull();
});

it('loads no third-party resources from any generated HTML or CSS file', () => {
  for (const path of filesWithin(dist)) {
    if (!/\.(?:html|css)$/.test(path)) continue;
    const source = readFileSync(path, 'utf8');
    if (path.endsWith('.css')) {
      checkCss(source, path);
      continue;
    }
    const document = new DOMParser().parseFromString(source, 'text/html');
    for (const element of document.querySelectorAll(
      'script[src], link[href], img[src], source[src], iframe[src]',
    )) {
      const attribute = element.tagName === 'LINK' ? 'href' : 'src';
      expectLocalResource(element.getAttribute(attribute)!, `${path}: ${element.outerHTML}`, element.tagName !== 'SCRIPT');
    }
    for (const element of document.querySelectorAll('source[srcset], img[srcset]')) {
      for (const url of srcsetUrls(element.getAttribute('srcset')!)) {
        expectLocalResource(url, `${path}: ${element.outerHTML}`);
      }
    }
    for (const style of document.querySelectorAll('style')) checkCss(style.textContent ?? '', path);
    for (const element of document.querySelectorAll('[style]')) checkCss(element.getAttribute('style')!, path);
  }
});
