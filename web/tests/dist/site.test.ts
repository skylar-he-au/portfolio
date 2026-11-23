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

function frontMatter(markdown: string, field: 'title' | 'summary' | 'kicker'): string {
  const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(markdown)?.[1];
  // These content files use single-line plain scalars for title and summary.
  const value = front?.match(new RegExp(`^${field}: (.+)$`, 'm'))?.[1]?.trim();
  if (!value) throw new Error(`Missing front matter field: ${field}`);
  return value;
}

interface Proof {
  value: string;
  label: string;
}

// The project pages write proof as a block list of { value, label } plain scalars.
function proofs(markdown: string): Proof[] {
  const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(markdown)?.[1] ?? '';
  const block = /^proof:\r?\n((?:[ \t]+.*(?:\r?\n|$))+)/m.exec(front)?.[1] ?? '';
  return Array.from(
    block.matchAll(/^\s*- value: (.+)\r?\n\s+label: (.+)$/gm),
    (match) => ({ value: match[1]!.trim(), label: match[2]!.trim() }),
  );
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

it('shows headline scores from the greatest qualifying run ids with published versions and dates', () => {
  const evalsDir = join(root, '../content/evals');
  const runs = existsSync(evalsDir)
    ? readdirSync(evalsDir).filter((name) => name.endsWith('.json'))
      .map((name) => JSON.parse(readFileSync(join(evalsDir, name), 'utf8')))
    : [];
  const document = readPage('/evals');
  const list = document.querySelector('ul.headline-cards');
  if (runs.length === 0) {
    expect(list).toBeNull();
    return;
  }
  // Read raw summaries independently of the library: dates do not select headlines.
  const greatestIdFirst = (a: { run_id: string }, b: { run_id: string }) =>
    a.run_id === b.run_id ? 0 : a.run_id > b.run_id ? -1 : 1;
  const qa = runs.filter((run) => run.kind === 'qa' && run.route === 'default' && run.split === 'all')
    .sort(greatestIdFirst)[0];
  const match = runs.filter((run) => run.kind === 'match').sort(greatestIdFirst)[0];
  expect(qa).toBeDefined();
  expect(match).toBeDefined();
  const cards = [...(list?.querySelectorAll(':scope > li') ?? [])];
  expect(list?.getAttribute('aria-label')).toBe('Headline scores');
  expect(document.querySelector('main h1')?.nextElementSibling).toBe(list);
  expect(list?.nextElementSibling?.classList.contains('intro')).toBe(true);
  expect(cards).toHaveLength(4);
  expect(cards.map((card) => card.querySelector('.headline-label')?.textContent)).toEqual([
    'Faithfulness, held-out questions', 'Project questions, code checks',
    'Chinese questions, code checks', 'Job match, code checks',
  ]);
  expect(cards.map((card) => card.firstElementChild?.textContent)).toEqual([
    qa.ragas.held_out.faithfulness.mean.toFixed(2),
    `${qa.code_checks.project.passed}/${qa.code_checks.project.total}`,
    `${qa.code_checks.zh.passed}/${qa.code_checks.zh.total}`,
    `${match.code_checks.all.passed}/${match.code_checks.all.total}`,
  ]);
  expect(cards.map((card) => card.querySelector('small')?.textContent)).toEqual(
    [qa, qa, qa, match].map((run) => `prompt ${run.prompt_version} · ${run.date}`),
  );
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
      // Home (9h): kicker, the two-line headline, then home.md as the intro. These may move
      // into the assistant's intro slot (9a1b), so only their order is pinned, not their parent.
      const h1 = main.querySelector('h1')!;
      expect(Array.from(h1.querySelectorAll(':scope > span'), (line) => line.textContent))
        .toEqual(['Ask about my work.', 'Every answer cites its page.']);
      expect(h1.textContent).toBe('Ask about my work. Every answer cites its page.');
      expect(h1.previousElementSibling?.classList.contains('home-kicker')).toBe(true);
      expect(h1.previousElementSibling?.textContent).toBe(frontMatter(markdown, 'kicker'));
      const intro = h1.nextElementSibling!;
      expect(intro.classList.contains('markdown')).toBe(true);
      // Markdown renders ' as a typographic apostrophe.
      expect(intro.textContent?.replace(/’/g, "'").replace(/\s+/g, ' ').trim())
        .toBe(markdown.replace(/^---[\s\S]*?\n---\r?\n/, '').replace(/\s+/g, ' ').trim());
    } else if (path.startsWith('/projects/')) {
      // Project pages (9p): title area, contents (side list and collapsed copy), body, then
      // one assistant whose dialog is the right column on desktop and a sheet on phones.
      const page = main.firstElementChild!;
      expect(page.matches('div.project-page')).toBe(true);
      // Toc.astro's bundled script lands beside the lists; it takes no grid cell.
      const blocks = Array.from(page.children).filter((child) => child.tagName !== 'SCRIPT');
      expect(blocks.map((child) => child.className)).toEqual([
        'project-head', 'toc toc-side', 'toc toc-collapse', 'markdown project-body', 'project-ask',
      ]);
      const head = page.querySelector(':scope > .project-head')!;
      expect(head.firstElementChild?.matches('p.project-page-kicker')).toBe(true);
      expect(head.firstElementChild?.textContent).toBe(frontMatter(markdown, 'kicker'));
      expect(head.querySelector(':scope > h1')?.textContent).toBe(title);
      expect(head.children[1]?.tagName).toBe('H1');
      const figures = Array.from(head.querySelectorAll('ul.project-page-proof > li'), (item) => ({
        value: item.querySelector('.project-page-proof-value')?.textContent,
        label: item.querySelector('.project-page-proof-label')?.textContent,
      }));
      expect(figures.length).toBeGreaterThan(0);
      expect(figures).toEqual(proofs(markdown));

      const sections = fixtures[path]!;
      const side = page.querySelector(':scope > nav.toc-side[aria-label="On this page"]')!;
      const collapse = page.querySelector(':scope > details.toc-collapse')!;
      expect(collapse.querySelector(':scope > summary')?.textContent).toBe('On this page');
      for (const list of [side, collapse.querySelector('nav[aria-label="On this page"]')!]) {
        expect(Array.from(list.querySelectorAll('ol > li > a'), (link) => [link.getAttribute('href'), link.textContent]))
          .toEqual(sections.map((section) => [`#${section.id}`, section.heading]));
      }
      expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);

      const ask = page.querySelector(':scope > .project-ask')!;
      const bar = ask.querySelector(':scope > button.ask-bar')!;
      expect(bar.getAttribute('aria-controls')).toBe('ask-dialog');
      expect(bar.getAttribute('aria-haspopup')).toBe('dialog');
      expect(bar.querySelector('.ask-bar-text')?.textContent).toBe('Ask about my projects');
      expect(bar.querySelector('.ask-bar-go')?.getAttribute('aria-hidden')).toBe('true');
      const dialog = ask.querySelector(':scope > dialog#ask-dialog')!;
      expect(dialog.hasAttribute('open')).toBe(false);
      expect(document.getElementById(dialog.getAttribute('aria-labelledby')!)?.textContent)
        .toBe('Ask about my projects');
      expect(dialog.querySelector('form[method="dialog"] button')?.textContent).toBe('Close');
      // One assistant instance on the page, so one conversation.
      expect(document.querySelectorAll('section#ask')).toHaveLength(1);
      expect(document.querySelectorAll('textarea')).toHaveLength(1);
      expect(dialog.querySelector('section#ask[aria-label="Ask about my projects"]')).not.toBeNull();
      // This page's example questions only: RoleRadar 1, 2, 3; OnCallOps 4, 5 (examples.md).
      const examples = readFileSync(join(root, '../content/examples.md'), 'utf8');
      const numbers = path === '/projects/roleradar' ? [1, 2, 3] : [4, 5];
      const expected = numbers.map((number) =>
        new RegExp(`^${number}\\. (.+?) \\([^()]*\\)$`, 'm').exec(examples)?.[1]);
      expect(Array.from(dialog.querySelectorAll('[role="group"][aria-label="Example questions"] button'),
        (button) => button.textContent)).toEqual(expected);
    } else if (path === '/about') {
      // About (9e).
      expect(main.firstElementChild?.tagName).toBe('H1');
      expect(main.querySelector('h1')?.textContent).toBe(title);
      expect(main.querySelector('.markdown')?.children.length).toBeGreaterThan(0);
      const lede = main.querySelector('h1')?.nextElementSibling;
      expect(lede?.matches('p.lede')).toBe(true);
      expect(lede?.textContent).toBe(frontMatter(markdown, 'summary'));
      const nav = lede?.nextElementSibling;
      expect(nav?.matches('nav.about-cards[aria-label="About sections"]')).toBe(true);
      expect(nav?.nextElementSibling).toBe(main.querySelector('.markdown'));
      const cards = [...(nav?.querySelectorAll('ul > li') ?? [])];
      expect(cards).toHaveLength(4);
      const sections = fixtures[path]!.filter((section) => section.id !== 'about-me');
      const links = cards.map((card) => card.querySelector(':scope > a')!);
      expect(links.map((link) => link.getAttribute('href')))
        .toEqual(['#work-rights', '#location', '#availability', '#contact']);
      expect(links.map((link) => link.getAttribute('href')))
        .toEqual(sections.map((section) => `#${section.id}`));
      for (const [index, link] of links.entries()) {
        const heading = main.querySelector(`.markdown h2${link.getAttribute('href')}`)!;
        expect(heading).not.toBeNull();
        const headingText = heading.cloneNode(true) as HTMLElement;
        headingText.querySelector('a.anchor')!.remove();
        const linkText = link.cloneNode(true) as HTMLElement;
        expect(linkText.querySelector('span[aria-hidden="true"]')?.textContent).toBe('→');
        linkText.querySelector('span[aria-hidden="true"]')!.remove();
        expect(linkText.textContent?.trim()).toBe(headingText.textContent?.trim());
        expect(linkText.textContent?.trim()).toBe(sections[index]!.heading);
      }
      expect(cards[3]?.querySelector('span[data-email-link]')?.textContent)
        .toBe('turn on JavaScript to see the email link');
      expect(cards[3]?.querySelector('a[href="https://www.linkedin.com/in/skylar-he"]')?.textContent)
        .toBe('LinkedIn');
      const workRights = main.querySelector('.markdown h2#work-rights')!;
      const workRightsText = markdown.match(/^## Work rights\r?\n\r?\n([^\n]+)/m)?.[1];
      expect(workRightsText).toBeDefined();
      // Astro's Markdown renderer turns apostrophes into typographic quotes.
      expect(workRights.nextElementSibling?.textContent?.replace(/’/g, "'")).toBe(workRightsText);
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

// 9p: sources point to project pages, /about#… and /privacy#…, so those pages load the section
// highlight as a bundled module script (never is:inline; csp.test.ts checks its hash).
// /evals is never a source and keeps no <script> (the Evals test above), so it is left out.
it('loads the section highlight script and its styles on every page a source can point to', () => {
  const highlightCss = filesWithin(join(dist, '_astro'))
    .filter((file) => file.endsWith('.css') && readFileSync(file, 'utf8').includes('.section-highlight'))
    .map((file) => `/${relative(dist, file)}`);
  expect(highlightCss.length).toBeGreaterThan(0);
  const html = filesWithin(dist).filter((name) => name.endsWith('.html'));
  expect(html.map((file) => relative(dist, file))).toContain('evals/index.html');
  for (const file of html.filter((name) => relative(dist, name) !== 'evals/index.html')) {
    const document = new DOMParser().parseFromString(readFileSync(file, 'utf8'), 'text/html');
    const scripts = Array.from(document.querySelectorAll('script[type="module"]'));
    // Astro inlines small bundled scripts (with a CSP hash) and links larger ones.
    const highlight = scripts.filter((script) => {
      const src = script.getAttribute('src');
      const code = src ? readFileSync(join(dist, src.replace(/^\//, '')), 'utf8') : script.textContent ?? '';
      return code.includes('section-highlight');
    });
    expect(highlight.length, relative(dist, file)).toBeGreaterThan(0);
    expect(document.querySelectorAll('script[is\\:inline]')).toHaveLength(0);
    const sheets = Array.from(document.querySelectorAll('link[rel="stylesheet"]'), (link) => link.getAttribute('href'));
    expect(sheets.some((href) => highlightCss.includes(href!)), relative(dist, file)).toBe(true);
  }
});

it('keeps every proof figure to at most 3, with numbers the project page body states', () => {
  for (const file of ['projects/roleradar.md', 'projects/oncallops.md']) {
    const markdown = readFileSync(join(root, '../content', file), 'utf8');
    const body = markdown.replace(/^---[\s\S]*?\n---\r?\n/, '');
    const items = proofs(markdown);
    expect(items.length, file).toBeGreaterThan(0);
    expect(items.length, file).toBeLessThanOrEqual(3);
    for (const { value } of items) {
      for (const number of value.match(/\d[\d,.]*/g) ?? []) {
        expect(body, `${file}: ${value}`).toMatch(new RegExp(`(?<![\\d.,])${number.replace(/[.,]/g, '\\$&')}(?![\\d])`));
      }
    }
  }
});

it('shows an unchecked example answer that cites the RoleRadar matching section on Home', () => {
  const document = readPage('/');
  const example = document.querySelector('main .example-answer')!;
  expect(example.querySelector('.example-label')?.textContent).toBe('Example answer');
  expect(example.querySelector('.example-question')?.textContent)
    .toBe('How does RoleRadar decide which jobs match a user?');
  // The example says what the cited section says, in its own words.
  const roleradar = readFileSync(join(root, '../content/projects/roleradar.md'), 'utf8');
  const claim = 'an AI model reads each full ad once, and plain code decides the fit tier';
  expect(roleradar).toContain(claim);
  expect(example.querySelector('.example-reply')?.textContent).toContain(claim);
  const href = '/projects/roleradar#how-roleradar-matches-jobs';
  expect(example.querySelector('.example-reply a.example-cite')?.getAttribute('href')).toBe(href);
  expect(example.querySelector('.example-reply a.example-cite')?.textContent).toBe('1');
  const source = example.querySelector('.example-sources a')!;
  expect(source.getAttribute('href')).toBe(href);
  expect(source.textContent).toBe('RoleRadar · How RoleRadar matches jobs');
  // Not run through the checker and no quoted source text, like a real answer.
  expect(example.textContent).not.toMatch(/Checked/);
  expect(example.querySelector('blockquote, q, mark')).toBeNull();
  expect(document.querySelector('main noscript')?.textContent)
    .toContain('The assistant needs JavaScript. The project pages work without it.');
});

it('renders the Ask component, then two project cards with proof figures and a coming-soon card, on Home', () => {
  const document = readPage('/');
  const projects = document.querySelector('section#projects[aria-label="Projects"]')!;
  const headings = projects.querySelectorAll('h2');
  expect(headings).toHaveLength(1);
  const heading = headings[0]!;
  expect(heading.textContent).toBe('Projects');
  expect(heading.querySelector('a.anchor')).toBeNull();
  expect(projects.querySelector('.home-projects-count')?.textContent).toBe('2 projects · 1 coming');
  const grid = projects.querySelector('.project-grid')!;
  expect(grid).not.toBeNull();
  expect(heading.compareDocumentPosition(grid) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(Array.from(projects.querySelectorAll('a'), (link) => link.getAttribute('href')))
    .toEqual(['/projects/roleradar', '/projects/oncallops']);
  const cards = grid.querySelectorAll(':scope > .project-card');
  expect(cards).toHaveLength(3);
  for (const [index, file] of ['projects/roleradar.md', 'projects/oncallops.md'].entries()) {
    const markdown = readFileSync(join(root, '../content', file), 'utf8');
    const card = cards[index]!;
    expect(card.tagName).toBe('ARTICLE');
    expect(card.firstElementChild?.tagName).toBe('H3');
    expect(card.querySelector('h3 > a')?.getAttribute('href')).toBe(`/${file.replace(/\.md$/, '')}`);
    expect(card.querySelector('h3')?.textContent).toBe(frontMatter(markdown, 'title'));
    expect(card.querySelector('.project-kicker')?.textContent).toBe(frontMatter(markdown, 'kicker'));
    expect(card.querySelector('p')?.textContent).toBe(frontMatter(markdown, 'summary'));
    // The card shows the first proof figure; the project page shows them all.
    const [first] = proofs(markdown);
    expect(card.querySelector('.project-proof-value')?.textContent).toBe(first!.value);
    expect(card.querySelector('.project-proof-label')?.textContent).toBe(first!.label);
  }
  expect(cards[1]!.querySelector('.project-kicker')?.textContent).toBe('Team project');
  // The third card lives in the template, not the content collection: no link, no page.
  const next = cards[2]!;
  expect(next.classList.contains('project-card-next')).toBe(true);
  expect(next.querySelector('a')).toBeNull();
  expect(next.textContent?.replace(/\s+/g, ' ').trim()).toBe('Next project · Coming soon');
  const ask = document.querySelector('section#ask[aria-label="Ask about my projects"]')!;
  expect(document.querySelectorAll('section#ask')).toHaveLength(1);
  // The assistant is on the first screen, above the projects.
  expect(ask.compareDocumentPosition(projects) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  // The assistant (9a1b) is server-rendered into the page: one box for questions and job ads.
  expect(ask.querySelectorAll('textarea')).toHaveLength(1);
  const field = ask.querySelector('textarea')!;
  expect(ask.querySelector(`label[for="${field.id}"]`)?.textContent).toBe('Ask a question, or paste a job ad');
  expect(ask.querySelector('button[type="submit"]')?.textContent).toBe('Ask');
  expect(ask.querySelectorAll('[role="group"][aria-label="Example questions"] button').length).toBeGreaterThan(0);
  expect(ask.querySelector('a[href="/privacy"]')).not.toBeNull();
  expect(ask.querySelector('[aria-atomic]')).toBeNull();
  expect(document.querySelector('#match')).toBeNull();
});

it('self-hosts Instrument Sans and JetBrains Mono and no longer ships Inter or Source Serif', () => {
  const css = filesWithin(dist).filter((path) => path.endsWith('.css'))
    .map((path) => readFileSync(path, 'utf8')).join('\n');
  const faces = Array.from(css.matchAll(/@font-face\s*\{[^}]*\}/g), (match) => match[0]);
  const families = new Set(faces.map((face) => /font-family:\s*['"]?([^;'"]+)/.exec(face)?.[1]));
  expect([...families].sort()).toEqual(['Instrument Sans Variable', 'JetBrains Mono Variable']);
  // astro.config.mjs keeps every font a file: CSP default-src 'self' blocks data: fonts.
  expect(css).not.toMatch(/data:(?:font|application\/(?:x-)?font)/i);
  for (const face of faces) {
    for (const match of face.matchAll(/url\(\s*["']?([^"')]+)/g)) {
      expect(match[1]).toMatch(/^\/_astro\/[^/]+\.woff2$/);
    }
  }
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
