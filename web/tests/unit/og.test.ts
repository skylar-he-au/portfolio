// @vitest-environment node
import satori from 'satori';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  OG_HEIGHT, OG_WIDTH, blogPosting, canonicalUrl, jsonLd, lightColors, loadFonts, ogImagePath, person, readTokens,
  renderOgPng, withoutTrailingSlash,
} from '../../src/lib/og';

const site = new URL('https://example.com');

afterEach(() => { vi.unstubAllGlobals(); });

describe('canonical URLs', () => {
  it('drops the trailing slash except on the home page', () => {
    expect(canonicalUrl('/', site)).toBe('https://example.com/');
    expect(canonicalUrl('/about/', site)).toBe('https://example.com/about');
    expect(canonicalUrl('/about', site)).toBe('https://example.com/about');
    expect(canonicalUrl('/blog/evals/', site)).toBe('https://example.com/blog/evals');
    expect(withoutTrailingSlash('https://example.com/projects/roleradar/')).toBe('https://example.com/projects/roleradar');
  });
});

describe('ogImagePath', () => {
  it('gives articles and project pages their own card and every other page the generic one', () => {
    expect(ogImagePath('/blog/evals')).toBe('/og/blog/evals.png');
    expect(ogImagePath('/blog/evals/')).toBe('/og/blog/evals.png');
    expect(ogImagePath('/projects/roleradar/')).toBe('/og/projects/roleradar.png');
    for (const path of ['/', '/blog', '/blog/', '/about', '/evals', '/404', '/blog/a/b']) {
      expect(ogImagePath(path), path).toBe('/og/default.png');
    }
  });
});

describe('jsonLd', () => {
  it('escapes "<" so a title cannot close the script block, and stays the same JSON', () => {
    const data = { headline: 'Why </script><script>alert(1)</script> & <b> matter', note: 'a\u2028b' };
    const text = jsonLd(data);
    expect(text).not.toContain('<');
    expect(text).not.toContain('>');
    expect(text).not.toContain('&');
    expect(text).not.toContain('\u2028');
    expect(JSON.parse(text)).toEqual(data);
  });

  it('writes a BlogPosting with Skylar as the author and absolute URLs', () => {
    const posting = blogPosting({
      pathname: '/blog/evals', title: 'Evals', summary: 'How.', date: new Date('2026-11-20'), updated: new Date('2026-11-24'),
    }, site);
    expect(posting).toEqual({
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      headline: 'Evals',
      description: 'How.',
      datePublished: '2026-11-20',
      dateModified: '2026-11-24',
      author: person(site),
      image: 'https://example.com/og/blog/evals.png',
      url: 'https://example.com/blog/evals',
      mainEntityOfPage: 'https://example.com/blog/evals',
    });
    expect(person(site)).toMatchObject({ '@type': 'Person', name: 'Skylar He', url: 'https://example.com/' });
  });
});

describe('share card', () => {
  it('reads the light colours from tokens.css', async () => {
    const colors = lightColors(await readTokens());
    expect(colors.background).toMatch(/^#[\da-f]{6}$/i);
    expect(colors.text).toMatch(/^#[\da-f]{6}$/i);
    expect(colors.accent).toMatch(/^#[\da-f]{6}$/i);
  });

  it('loads static 400 and 700 font files, so bold renders differently from regular', async () => {
    const fonts = await loadFonts();
    expect(fonts.map((font) => font.weight)).toEqual([400, 700]);
    const draw = (fontWeight: number) => satori(
      { type: 'div', props: { style: { display: 'flex', fontSize: 64, fontWeight }, children: 'Skylar He' } } as never,
      { width: 600, height: 120, fonts },
    );
    // A variable font file would render at its default weight both times (identical paths).
    expect(await draw(700)).not.toBe(await draw(400));
  });

  it('renders a 1200x630 PNG without any network request', async () => {
    const fetch = vi.fn(() => { throw new Error('network request during the card build'); });
    vi.stubGlobal('fetch', fetch);
    const png = Buffer.from(await renderOgPng({ title: 'RoleRadar', label: 'Project', tagline: 'AI and full-stack developer' }));
    expect(png.subarray(1, 4).toString('ascii')).toBe('PNG');
    expect(png.readUInt32BE(16)).toBe(OG_WIDTH);
    expect(png.readUInt32BE(20)).toBe(OG_HEIGHT);
    expect(fetch).not.toHaveBeenCalled();
  });
});
