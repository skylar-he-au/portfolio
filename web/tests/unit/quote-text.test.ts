// Quoted sentences on the page (step 10, SPEC "出处标到句子"): the browser half of the plain-text
// rule, run on the rendered HTML of the same sections as the API's shared fixture
// (api/tests/fixtures/quotes.json), with the same quotes.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { citationHref, citationList } from '../../src/lib/answer';
import {
  findDirective,
  findQuote,
  fold,
  parseTextDirectives,
  fragmentDirective,
  sectionElements,
  sectionHeadingIn,
  textBlocks,
  textDirective,
} from '../../src/lib/quote-text';

interface FixtureSection {
  page: string;
  section_id: string;
  heading: string;
  match: { quote: string; expect: string; why: string }[];
  no_match: { quote: string; why: string }[];
}

const shared = JSON.parse(readFileSync(join(process.cwd(), '../api/tests/fixtures/quotes.json'), 'utf8')) as {
  sections: FixtureSection[];
};
const PAGE_FILES: Record<string, string> = {
  '/projects/roleradar': 'roleradar.html',
  '/projects/oncallops': 'oncallops.html',
  '/privacy': 'privacy.html',
  '/about': 'about.html',
};

function pageHtml(page: string): string {
  return readFileSync(join(process.cwd(), 'tests/fixtures/quote-pages', PAGE_FILES[page]!), 'utf8');
}

function load(page: string): void {
  document.body.innerHTML = `<main>${pageHtml(page)}</main>`;
}

function section(id: string): Element[] {
  const heading = sectionHeadingIn(document, id);
  if (!heading) throw new Error(`no section ${id}`);
  return sectionElements(heading);
}

let markSection: typeof import('../../src/scripts/highlight').markSection;

beforeAll(async () => {
  window.matchMedia = vi.fn().mockReturnValue({ matches: false }) as unknown as typeof window.matchMedia;
  Element.prototype.scrollIntoView = vi.fn();
  ({ markSection } = await import('../../src/scripts/highlight'));
});

afterEach(() => {
  vi.useRealTimers();
  document.body.innerHTML = '';
});

describe('fold (same rule as api/app/qa/output.py fold)', () => {
  it('straightens quotes, folds dashes and ellipses, collapses spaces', () => {
    expect(fold('  It’s “live” – 2026—now… done \n ok ')).toBe('It\'s "live" - 2026--now... done ok');
  });
});

describe('the shared fixture on the rendered pages', () => {
  for (const fixture of shared.sections) {
    describe(`${fixture.page}#${fixture.section_id}`, () => {
      it('finds every quote the API keeps, inside the section', () => {
        load(fixture.page);
        const blocks = textBlocks(section(fixture.section_id));
        for (const { expect: quote, why } of fixture.match) {
          expect(findQuote(blocks, quote), why).not.toBeNull();
        }
      });

      it('finds none of the quotes the API drops (changed words, across blocks, too short)', () => {
        load(fixture.page);
        const blocks = textBlocks(section(fixture.section_id));
        for (const { quote, why } of fixture.no_match) {
          expect(findQuote(blocks, fold(quote)), why).toBeNull();
        }
      });
    });
  }

  it('does not read the "#" beside each heading as text', () => {
    load('/projects/roleradar');
    const blocks = textBlocks(section('how-roleradar-matches-jobs'));
    expect(blocks[0]!.folded).toBe('How RoleRadar matches jobs');
  });
});

function marks(): string[] {
  return [...document.querySelectorAll('mark.quote-highlight')].map((mark) => mark.textContent ?? '');
}

function wholeSectionMarked(id: string): boolean {
  return section(id).every((element) => element.classList.contains('section-highlight'));
}

describe('marking quotes in a section (highlight.ts)', () => {
  it('marks a sentence in one paragraph, and only that', () => {
    load('/projects/roleradar');
    const quote = 'Because tiers come from code, every reason traces back to a line in the ad';
    const first = markSection(sectionHeadingIn(document, 'how-roleradar-matches-jobs')!, [quote]);
    expect(marks().join('')).toBe(quote);
    expect(first?.textContent).toBe(quote);
    expect(document.querySelector('.section-highlight')).toBeNull();
  });

  it('marks a sentence across bold and a link, keeping those elements', () => {
    load('/privacy');
    const bold = 'Anthropic deletes API inputs and outputs within 30 days.';
    const link = "Cloudflare's own description is in its Turnstile Privacy Addendum.";
    const heading = sectionHeadingIn(document, 'how-long-the-ai-providers-keep-it')!;
    const strong = [...document.querySelectorAll('strong')].find((element) => element.textContent === 'Anthropic');
    const anchor = [...document.querySelectorAll('a')].find((element) => element.textContent === 'Turnstile Privacy Addendum');
    expect(strong && anchor).toBeTruthy();
    markSection(heading, [bold, link]);
    expect(fold(marks().join(''))).toBe(fold(`${bold}${link.replace("'", '’')}`));
    // The bold and the link are still there, each with a mark inside.
    expect(strong!.isConnected && strong!.querySelector('mark.quote-highlight')).toBeTruthy();
    expect(anchor!.isConnected && anchor!.getAttribute('href')).toContain('cloudflare.com');
    expect(anchor!.querySelector('mark.quote-highlight')?.textContent).toBe('Turnstile Privacy Addendum');
  });

  it('finds a straight apostrophe where the page has a curly one, and marks the page text', () => {
    load('/projects/roleradar');
    markSection(sectionHeadingIn(document, 'how-roleradar-matches-jobs')!, [
      "So I added a per-user resume check of the ad's must-have requirements, again with quotes that code verifies.",
    ]);
    expect(marks().join('')).toBe('So I added a per-user resume check of the ad’s must-have requirements, again with quotes that code verifies.');
  });

  it('marks the first time a sentence appears twice in the section', () => {
    document.body.innerHTML = `<div class="markdown">
      <h2 id="twice">Twice</h2>
      <p>First: the same sentence is here. Then more.</p>
      <p>Second: the same sentence is here. End.</p>
      <h2 id="next">Next</h2>
      <p>The same sentence is here.</p>
    </div>`;
    const first = markSection(sectionHeadingIn(document, 'twice')!, ['the same sentence is here']);
    expect(marks()).toEqual(['the same sentence is here']);
    expect(first?.parentElement?.textContent).toContain('First:');
  });

  it('marks the whole section when no quote is found, or there are none', () => {
    load('/projects/roleradar');
    const heading = sectionHeadingIn(document, 'how-roleradar-matches-jobs')!;
    expect(markSection(heading, ['RoleRadar matches jobs with three layers.'])).toBeNull();
    expect(marks()).toEqual([]);
    expect(wholeSectionMarked('how-roleradar-matches-jobs')).toBe(true);
    markSection(heading, []);
    expect(wholeSectionMarked('how-roleradar-matches-jobs')).toBe(true);
  });

  it('puts the page text back as it was when the mark ends', () => {
    vi.useFakeTimers();
    load('/privacy');
    const before = document.body.innerHTML;
    markSection(sectionHeadingIn(document, 'how-long-the-ai-providers-keep-it')!, ['Anthropic deletes API inputs and outputs within 30 days.']);
    expect(document.body.innerHTML).not.toBe(before);
    vi.advanceTimersByTime(3600);
    expect(document.body.innerHTML).toBe(before);
  });

  it('marks the quote when a source link with quotes on this page is clicked', () => {
    load('/projects/roleradar');
    const link = document.createElement('a');
    link.href = '#how-roleradar-matches-jobs:~:text=Because%20tiers';
    link.dataset.quotes = JSON.stringify(['Because tiers come from code, every reason traces back to a line in the ad']);
    link.textContent = '1';
    document.body.append(link);
    link.click();
    expect(marks().join('')).toBe('Because tiers come from code, every reason traces back to a line in the ad');
    expect(window.location.hash).toBe('#how-roleradar-matches-jobs');
  });
});

function pageBlocks(page: string) {
  load(page);
  return textBlocks([document.body]);
}

describe('a page opened from a source link with a Text Fragment', () => {
  it('reads text= directives back, start,end included, and skips malformed ones', () => {
    expect(parseTextDirectives('#s:~:text=The%20second%20layer&text=Every%20extracted,find%20there.&text=%E0%A4%A&other=x'))
      .toEqual([{ start: 'The second layer' }, { start: 'Every extracted', end: 'find there.' }]);
    expect(parseTextDirectives('#s')).toEqual([]);
  });

  it('finds the directive written for a quote in the section, ignoring case like the browser', () => {
    load('/projects/roleradar');
    const blocks = textBlocks(section('how-roleradar-matches-jobs'));
    const long = 'Every extracted fact has to include a verbatim quote from the ad, and code drops or downgrades any quote it cannot find there.';
    const directive = textDirective(textBlocks([document.body]), blocks, long)!;
    const [part] = parseTextDirectives(`#how-roleradar-matches-jobs:~:${directive}`);
    const match = findDirective(blocks, part!)!;
    expect(fold(match.block.raw.slice(match.start, match.end))).toBe(long);
    expect(findDirective(blocks, { start: 'THE SECOND LAYER RUNS' })).not.toBeNull();
    expect(findDirective(blocks, { start: 'not on this page at all' })).toBeNull();
  });

  it('marks the directive sentences, not the whole section, when the page is opened with them', () => {
    load('/projects/roleradar');
    markSection(sectionHeadingIn(document, 'how-roleradar-matches-jobs')!, [], [{ start: 'The second layer runs per user, in code.' }]);
    expect(marks()).toEqual(['The second layer runs per user, in code.']);
    expect(document.querySelector('.section-highlight')).toBeNull();
  });
});

describe('Text Fragment for a source link', () => {
  it('uses the page writing, encoded, for a quote that appears once', () => {
    const page = pageBlocks('/projects/roleradar');
    const sectionBlocks = textBlocks(section('how-roleradar-matches-jobs'));
    expect(textDirective(page, sectionBlocks, 'The second layer runs per user, in code.'))
      .toBe('text=The%20second%20layer%20runs%20per%20user%2C%20in%20code.');
    // The curly apostrophe and the hyphen as the page writes them; "-" is reserved in directives.
    const directive = textDirective(page, sectionBlocks, "So I added a per-user resume check of the ad's must-have requirements");
    expect(directive).toBe('text=So%20I%20added%20a%20per%2Duser%20resume%20check%20of%20the%20ad%E2%80%99s%20must%2Dhave%20requirements');
  });

  it('writes a long sentence as start,end and several quotes as several text=', () => {
    const page = pageBlocks('/projects/roleradar');
    const sectionBlocks = textBlocks(section('how-roleradar-matches-jobs'));
    const long = 'Every extracted fact has to include a verbatim quote from the ad, and code drops or downgrades any quote it cannot find there.';
    expect(textDirective(page, sectionBlocks, long))
      .toBe('text=Every%20extracted%20fact%20has%20to,quote%20it%20cannot%20find%20there.');
    const fragment = fragmentDirective(page, sectionBlocks, [long, 'Because tiers come from code, every reason traces back to a line in the ad']);
    expect(fragment).toBe(':~:text=Every%20extracted%20fact%20has%20to,quote%20it%20cannot%20find%20there.'
      + '&text=Because%20tiers%20come%20from%20code%2C,a%20line%20in%20the%20ad');
  });

  it('leaves out a quote that appears more than once on the page', () => {
    document.body.innerHTML = `<div class="markdown">
      <h2 id="a">A</h2><p>Code checks every quote it gets.</p>
      <h2 id="b">B</h2><p>As before, code checks every quote it gets.</p>
    </div>`;
    const page = textBlocks([document.body]);
    const sectionBlocks = textBlocks(sectionElements(sectionHeadingIn(document, 'a')!));
    expect(textDirective(page, sectionBlocks, 'Code checks every quote it gets.')).toBeNull();
    expect(fragmentDirective(page, sectionBlocks, ['Code checks every quote it gets.'])).toBe('');
  });

  it('adds the fragment to the site link, and still refuses addresses off the site', () => {
    const fragments = new Map([[1, ':~:text=Because%20tiers'], [2, ':~:text=x']]);
    expect(citationHref({ n: 1, url: '/projects/roleradar#how-roleradar-matches-jobs', heading: 'h' }, fragments))
      .toBe('/projects/roleradar#how-roleradar-matches-jobs:~:text=Because%20tiers');
    expect(citationHref({ n: 2, url: 'https://evil.example/#x', heading: 'h' }, fragments)).toBeNull();
    expect(citationHref({ n: 2, url: '//evil.example/x', heading: 'h' }, fragments)).toBeNull();
    expect(citationHref({ n: 3, url: '/about', heading: 'h' }, fragments)).toBe('/about');
  });

  it('keeps only expected citation fields and string quotes from an answer event', () => {
    expect(citationList([
      { n: 1, url: '/a#b', heading: 'B', quotes: ['A sentence that is long enough.', 7, 'short'], extra: 1 },
      { n: 2, url: '/a#c', heading: 'C' },
      { n: '3', url: '/a', heading: 'D' },
    ])).toEqual([
      { n: 1, url: '/a#b', heading: 'B', quotes: ['A sentence that is long enough.'] },
      { n: 2, url: '/a#c', heading: 'C', quotes: [] },
    ]);
    expect(citationList('nope')).toEqual([]);
  });
});
