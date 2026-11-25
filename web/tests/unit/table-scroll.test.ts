import { markdownToHtml } from 'satteri';
import { describe, expect, it } from 'vitest';
import { tableLabels, tableScroll } from '../../src/lib/table-scroll';

// The real Markdown processor Astro uses, with the plugin as astro.config.mjs registers it.
function render(markdown: string): Document {
  const { html } = markdownToHtml(markdown, { hastPlugins: [tableScroll], features: { gfm: true } } as never);
  return new DOMParser().parseFromString(`<main>${html}</main>`, 'text/html');
}

const TABLE = '| Setup | Accuracy |\n|---|---|\n| Verdict only | 0.967 |\n';

describe('table-scroll (Sätteri hast plugin)', () => {
  it('wraps a table in a focusable region named after the heading above it', () => {
    const doc = render(`## Should the checker *write* its reasoning first?\n\nIntro.\n\n${TABLE}`);
    const wrapper = doc.querySelector('main > div.table-wrap')!;
    expect(wrapper).not.toBeNull();
    expect(wrapper.getAttribute('tabindex')).toBe('0');
    expect(wrapper.getAttribute('role')).toBe('region');
    expect(wrapper.getAttribute('aria-label')).toBe('Table: Should the checker write its reasoning first?');
    expect(wrapper.children).toHaveLength(1);
    expect(wrapper.firstElementChild!.tagName).toBe('TABLE');
    expect(wrapper.querySelector('td')!.textContent).toBe('Verdict only');
    expect(doc.querySelector('main > p')!.textContent).toBe('Intro.');
  });

  it('numbers tables that share a heading, in document order, including nested ones', () => {
    const doc = render([TABLE, '## Results', TABLE, `> ${TABLE.trim().split('\n').join('\n> ')}`, '### Cost', TABLE].join('\n\n'));
    const labels = [...doc.querySelectorAll('div.table-wrap')].map((div) => div.getAttribute('aria-label'));
    expect(labels).toEqual(['Table', 'Table: Results (1 of 2)', 'Table: Results (2 of 2)', 'Table: Cost']);
    expect(doc.querySelectorAll('table')).toHaveLength(4);
    expect([...doc.querySelectorAll('table')].every((table) => table.parentElement!.matches('div.table-wrap'))).toBe(true);
  });

  it('leaves a page without tables as it was', () => {
    const markdown = '## A\n\nB and `c`.\n';
    expect(markdownToHtml(markdown, { hastPlugins: [tableScroll] } as never).html).toBe(markdownToHtml(markdown).html);
  });

  it('labels from heading texts', () => {
    expect(tableLabels(['', 'A', 'A', 'B'])).toEqual(['Table', 'Table: A (1 of 2)', 'Table: A (2 of 2)', 'Table: B']);
  });
});
