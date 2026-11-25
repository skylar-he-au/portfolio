import { describe, expect, it } from 'vitest';
import { addSectionAnchors } from '../../src/lib/anchors';

describe('addSectionAnchors', () => {
  it('adds an accessible anchor while preserving the rendered heading and id', () => {
    expect(addSectionAnchors('<h2 id="astro-chose-this">A <em>section</em> &amp; more</h2>'))
      .toBe('<h2 id="astro-chose-this" aria-labelledby="astro-chose-this:title"><span id="astro-chose-this:title">A <em>section</em> &amp; more</span> '
        + '<a class="anchor" href="#astro-chose-this" aria-label="Link to A section &amp; more">#</a></h2>');
  });

  it('names the heading by its words only and the link after the section', () => {
    const container = document.createElement('div');
    container.innerHTML = addSectionAnchors('<h2 id="in-30-seconds">In 30 seconds</h2><h2 id="why">Why “<code>quotes</code>” and "more"</h2>');
    const [first, second] = [...container.querySelectorAll('h2')];
    const label = (heading: Element) => container.querySelector(`[id="${heading.getAttribute('aria-labelledby')}"]`)!;
    expect(label(first!).textContent).toBe('In 30 seconds');
    expect(label(first!).contains(first!.querySelector('a.anchor'))).toBe(false);
    expect(first!.querySelector('a.anchor')!.getAttribute('aria-label')).toBe('Link to In 30 seconds');
    expect(label(second!).textContent).toBe('Why “quotes” and "more"');
    expect(second!.querySelector('a.anchor')!.getAttribute('aria-label')).toBe('Link to Why “quotes” and "more"');
    expect(first!.id).toBe('in-30-seconds');
    expect(container.querySelectorAll('[id]')).toHaveLength(4);
  });

  it('preserves duplicate suffixes, Unicode ids, and document order', () => {
    const html = '<h2 id="你好">你好</h2>\n<h2 id="你好-1">你好</h2>';
    const result = addSectionAnchors(html);
    expect(result.match(/href="#[^"]+"/g)).toEqual(['href="#你好"', 'href="#你好-1"']);
  });

  it('handles multiline content and attributes around a single-quoted id', () => {
    const result = addSectionAnchors("<h2 class='section' id='details' tabindex='-1'>First\nsecond</h2>");
    expect(result).toContain("<h2 class='section' id='details' tabindex='-1' aria-labelledby=\"details:title\"><span id=\"details:title\">First\nsecond</span>");
    expect(result).toContain('aria-label="Link to First second"');
    expect(result).toContain('href="#details"');
  });

  it('only adds anchors to h2 elements with ids', () => {
    const html = '<h1 id="title">Title</h1><h2>No id</h2><h3 id="small">Small</h3>'
      + '<p>Text</p><pre>&lt;h2 id="code"&gt;Code&lt;/h2&gt;</pre>';
    expect(addSectionAnchors(html)).toBe(html);
    expect(addSectionAnchors('')).toBe('');
  });
});
