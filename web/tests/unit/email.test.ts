import { describe, expect, it } from 'vitest';
import { emailAddress, mountEmailLinks, renderEmailPlaceholder } from '../../src/lib/email';

describe('emailAddress', () => {
  it('assembles the address from its parts', () => {
    expect(emailAddress()).toBe('hello@example.com');
  });
});

describe('renderEmailPlaceholder', () => {
  it('replaces every placeholder in rendered HTML with a fallback span', () => {
    expect(renderEmailPlaceholder('<p>{{email}} or {{email}}</p>', 'Email link')).toBe(
      '<p><span data-email-link>Email link</span> or <span data-email-link>Email link</span></p>',
    );
  });

  it('leaves HTML without a placeholder unchanged', () => {
    const html = '<p>Read <a href="/about">About</a>.</p>';
    expect(renderEmailPlaceholder(html, 'Email link')).toBe(html);
    expect(renderEmailPlaceholder('', 'Email link')).toBe('');
  });

  it('escapes the plain-text fallback and preserves replacement tokens literally', () => {
    expect(renderEmailPlaceholder('{{email}}', `<a title="Email">Tom & 'Skylar'</a> $&`)).toBe(
      '<span data-email-link>&lt;a title=&quot;Email&quot;&gt;Tom &amp; &#39;Skylar&#39;&lt;/a&gt; $&amp;</span>',
    );
  });

  it('does not expose an address in the rendered output', () => {
    const html = renderEmailPlaceholder('<p>Contact {{email}}.</p>', 'Email link');
    expect(html).not.toContain('@');
    expect(html).not.toContain(emailAddress());
  });
});

describe('mountEmailLinks', () => {
  it('upgrades every span in the supplied document and is idempotent', () => {
    const root = new DOMParser().parseFromString(
      renderEmailPlaceholder('<p>{{email}} or {{email}}</p>', 'Email link'),
      'text/html',
    );
    mountEmailLinks(root);
    const links = [...root.querySelectorAll('a')];
    expect(links).toHaveLength(2);
    for (const link of links) {
      expect(link.href).toBe('mailto:' + emailAddress());
      expect(link.textContent).toBe(emailAddress());
      expect(link.className).toBe('email-link');
    }
    expect(root.querySelectorAll('span')).toHaveLength(0);
    const html = root.body.innerHTML;
    mountEmailLinks(root);
    expect(root.body.innerHTML).toBe(html);
    expect(root.querySelector('a')).toBe(links[0]);
  });

  it('defaults to the current document', () => {
    const container = document.createElement('div');
    container.innerHTML = renderEmailPlaceholder('{{email}}', 'Email link');
    document.body.append(container);
    try {
      mountEmailLinks();
      expect(container.querySelector('a')?.href).toBe('mailto:' + emailAddress());
      expect(container.querySelector('span')).toBeNull();
    } finally {
      container.remove();
    }
  });

  it('does nothing when there are no placeholders', () => {
    const root = new DOMParser().parseFromString(
      '<p><span>Contact</span> <a href="/about">About</a></p>',
      'text/html',
    );
    const html = root.body.innerHTML;
    const link = root.querySelector('a');
    mountEmailLinks(root);
    expect(root.body.innerHTML).toBe(html);
    expect(root.querySelector('a')).toBe(link);
  });
});
