import { describe, expect, it } from 'vitest';
import { addSectionAnchors } from '../../src/lib/anchors';

describe('addSectionAnchors', () => {
  it('adds an accessible anchor while preserving the rendered heading and id', () => {
    expect(addSectionAnchors('<h2 id="astro-chose-this">A <em>section</em> &amp; more</h2>'))
      .toBe('<h2 id="astro-chose-this">A <em>section</em> &amp; more <a class="anchor" href="#astro-chose-this" aria-label="Link to this section">#</a></h2>');
  });

  it('preserves duplicate suffixes, Unicode ids, and document order', () => {
    const html = '<h2 id="你好">你好</h2>\n<h2 id="你好-1">你好</h2>';
    const result = addSectionAnchors(html);
    expect(result.match(/href="#[^"]+"/g)).toEqual(['href="#你好"', 'href="#你好-1"']);
  });

  it('handles multiline content and attributes around a single-quoted id', () => {
    const result = addSectionAnchors("<h2 class='section' id='details' tabindex='-1'>First\nsecond</h2>");
    expect(result).toContain("<h2 class='section' id='details' tabindex='-1'>First\nsecond");
    expect(result).toContain('href="#details"');
  });

  it('only adds anchors to h2 elements with ids', () => {
    const html = '<h1 id="title">Title</h1><h2>No id</h2><h3 id="small">Small</h3>'
      + '<p>Text</p><pre>&lt;h2 id="code"&gt;Code&lt;/h2&gt;</pre>';
    expect(addSectionAnchors(html)).toBe(html);
    expect(addSectionAnchors('')).toBe('');
  });
});
