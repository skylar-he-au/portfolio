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

it('builds exactly the six public HTML pages and excludes examples', () => {
  expect(filesWithin(dist).filter((path) => path.endsWith('.html'))
    .map((path) => relative(dist, path)).sort()).toEqual([
    'about/index.html',
    'evals/index.html',
    'index.html',
    'privacy/index.html',
    'projects/oncallops/index.html',
    'projects/roleradar/index.html',
  ]);
  expect(existsSync(join(dist, 'examples/index.html'))).toBe(false);
});

it('renders the Evals heading, and run tables when summaries exist or the empty state when none do', () => {
  const evalsDir = join(root, '../content/evals');
  const published = existsSync(evalsDir)
    ? readdirSync(evalsDir).filter((name) => name.endsWith('.json'))
    : [];
  const document = readPage('/evals');
  const main = document.querySelector('main');
  expect(document.querySelector('main h1')?.textContent).toBe('Evals');
  if (published.length > 0) {
    expect(main?.querySelectorAll('table').length).toBeGreaterThan(0);
    expect(main?.textContent).not.toContain('No eval runs published yet.');
  } else {
    expect(main?.textContent).toContain('No eval runs published yet.');
    expect(document.querySelectorAll('main section, main table')).toHaveLength(0);
  }
  expect(document.querySelectorAll('script')).toHaveLength(0);
  expect(document.querySelector('footer a[href="/evals"]')?.textContent).toBe('Evals');
});

it('shows a Job match block with a score table for each published match summary', () => {
  const evalsDir = join(root, '../content/evals');
  const matchRuns = existsSync(evalsDir)
    ? readdirSync(evalsDir).filter((name) => name.endsWith('.json')).filter((name) =>
      JSON.parse(readFileSync(join(evalsDir, name), 'utf8')).kind === 'match')
    : [];
  const block = [...readPage('/evals').querySelectorAll('main section')]
    .find((section) => section.querySelector('h2')?.textContent === 'Job match');
  if (matchRuns.length === 0) {
    expect(block).toBeUndefined();
    return;
  }
  expect(block?.querySelectorAll('article.match-run')).toHaveLength(matchRuns.length);
  for (const run of block!.querySelectorAll('article.match-run')) {
    expect(run.querySelector('.pass-lines')?.textContent).toContain('Section recall ≥ 0.80');
    expect(run.querySelectorAll('tbody tr').length).toBeGreaterThan(0);
    // API judges by model id; round 2 on, the subscription grader by name, never RAGAS.
    expect(run.textContent).toMatch(
      /Judge: (claude-|Claude Sonnet 5\.5 \(subscription, following GRADING\.md\), not RAGAS)/,
    );
    expect(run.textContent).toMatch(/Cost: <?\$\S+ (paid \+ <?\$\S+ provider credit|total)/);
  }
});

it('shows the per-step model comparison with one table per step when it is published', () => {
  const compareDir = join(root, '../content/evals/compare');
  const published = existsSync(join(compareDir, 'step6-r2.json'));
  const section = [...readPage('/evals').querySelectorAll('main section')]
    .find((s) => s.querySelector('h2')?.textContent === 'Choosing a model for each step');
  if (!published) {
    expect(section).toBeUndefined();
    return;
  }
  const steps = [...section!.querySelectorAll('article.compare-step h3')].map((h) => h.textContent);
  expect(steps).toEqual(['Read the ad', 'Match each requirement', 'Check each link', 'Write the guide']);
  for (const table of section!.querySelectorAll('article.compare-step table')) {
    expect(table.querySelector('tbody tr th')?.textContent).toContain('(chosen)');
    expect(table.textContent).toContain('v4 prompt');
  }
  expect(section!.textContent).toContain('not RAGAS');
});

it('shows every run cost on the evals page as paid plus provider credit, or as a total', () => {
  const costs = [...readPage('/evals').querySelectorAll('main .meta')]
    .map((meta) => meta.textContent?.replace(/\s+/g, ' ').trim().match(/Cost: (.*)$/)?.[1])
    .filter((cost) => cost !== undefined);
  expect(costs.length).toBeGreaterThan(0);
  for (const cost of costs) {
    expect(cost).toMatch(/^<?\$\d+\.\d{2} (paid \+ <?\$\d+\.\d{2} provider credit|total)$/);
  }
});

it('explains prompt v9 and the cut-off medium-thinking verdicts on the evals page', () => {
  const notes = [...readPage('/evals').querySelectorAll('main .note')].map((note) =>
    note.textContent?.replace(/\s+/g, ' ').trim(),
  );
  expect(notes.some((note) => note?.startsWith('Prompt v9 only changed the output of the general answer check'))).toBe(true);
  const cotRun = existsSync(join(root, '../content/evals/20261118T071804-v8-cot.json'));
  expect(notes.includes(
    'Medium thinking counts toward the 1024-token output cap and ran past it on 55 of 186 calls, so those verdicts were cut off.',
  )).toBe(cotRun);
});

it('pins all 31 sections in document order', () => {
  expect(Object.keys(fixtures)).toEqual(Object.keys(pages));
  expect(Object.values(fixtures).map((sections) => sections.length)).toEqual([9, 9, 5, 8, 0]);
  expect(Object.values(fixtures).flat()).toHaveLength(31);
});

describe.each(Object.entries(pages))('%s', (path, source) => {
  const markdown = readFileSync(join(root, '../content', source), 'utf8');
  const document = readPage(path);

  it('uses content metadata in the head and has one main and one h1', () => {
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
  });

  // One branch per page layout (step 9): each block edits only its own page's branch.
  it('renders the page body in main', () => {
    const title = frontMatter(markdown, 'title');
    const main = document.querySelector('main#main')!;
    if (path === '/') {
      // Home (9h).
      expect(main.firstElementChild?.tagName).toBe('H1');
      expect(main.querySelector('h1')?.textContent).toBe('Skylar He');
      expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);
      expect(main.querySelector('h1')?.nextElementSibling?.classList.contains('markdown')).toBe(true);
    } else if (path.startsWith('/projects/')) {
      // Project pages (9p).
      expect(main.firstElementChild?.tagName).toBe('H1');
      expect(main.querySelector('h1')?.textContent).toBe(title);
      expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);
      expect(main.querySelector('h1')?.nextElementSibling?.classList.contains('markdown')).toBe(true);
    } else if (path === '/about') {
      // About (9e).
      expect(main.firstElementChild?.tagName).toBe('H1');
      expect(main.querySelector('h1')?.textContent).toBe(title);
      expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);
      expect(main.querySelector('h1')?.nextElementSibling?.classList.contains('markdown')).toBe(true);
    } else {
      // Every other content page (/privacy).
      expect(main.firstElementChild?.tagName).toBe('H1');
      expect(main.querySelector('h1')?.textContent).toBe(title);
      expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);
      expect(main.querySelector('h1')?.nextElementSibling?.classList.contains('markdown')).toBe(true);
    }
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
    expect(Array.from(document.querySelectorAll('header nav a'), (link) =>
      [link.getAttribute('href'), link.textContent])).toEqual([
      ['/#projects', 'Projects'],
      ['/evals', 'Evals'],
      ['/about', 'About'],
    ]);
    expect(document.querySelector('footer a[href="/privacy"]')?.textContent).toBe('Privacy');
    expect(document.querySelector('footer a[href="/evals"]')?.textContent).toBe('Evals');
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
  // The Ask component is server-rendered into the page.
  const field = ask.querySelector('textarea, input[type="text"]')!;
  expect(field).not.toBeNull();
  expect(ask.querySelector(`label[for="${field.id}"]`)).not.toBeNull();
  expect(ask.querySelector('a[href="/privacy"]')).not.toBeNull();
});

it('self-hosts Instrument Sans and JetBrains Mono and no longer ships Inter or Source Serif', () => {
  const css = filesWithin(dist).filter((path) => path.endsWith('.css'))
    .map((path) => readFileSync(path, 'utf8')).join('\n');
  const faces = Array.from(css.matchAll(/@font-face\s*\{[^}]*\}/g), (match) => match[0]);
  const families = new Set(faces.map((face) => /font-family:\s*['"]?([^;'"]+)/.exec(face)?.[1]));
  expect([...families].sort()).toEqual(['Instrument Sans Variable', 'JetBrains Mono Variable']);
  // Vite inlines subsets under 4 kB (JetBrains Mono Cyrillic-ext) as data: URIs.
  for (const face of faces) expect(face).toMatch(/url\(\s*["']?(?:\/_astro\/[^)]+\.woff2|data:font\/woff2;)/);
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
