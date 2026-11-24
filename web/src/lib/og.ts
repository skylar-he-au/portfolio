// Share preview cards, canonical URLs and JSON-LD for the page <head> (docs/blog.md "网站部件").
// Cards are rendered at build time from local files only: no font or image is fetched.
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';

export const AUTHOR_NAME = 'Skylar He';
export const LINKEDIN_URL = 'https://www.linkedin.com/in/skylar-he';
export const OG_WIDTH = 1200;
export const OG_HEIGHT = 630;

/** Public pages have no trailing slash, the same as the site's links, sitemap and RSS. */
export function withoutTrailingSlash(url: string): string {
  const parsed = new URL(url);
  if (parsed.pathname !== '/') parsed.pathname = parsed.pathname.replace(/\/+$/, '');
  return parsed.href;
}

export function canonicalUrl(pathname: string, site: URL): string {
  return withoutTrailingSlash(new URL(pathname, site).href);
}

/** Pages with their own card; every other page uses the generic one. */
const CARD_PAGE = /^\/(blog|projects)\/([^/]+)\/?$/;

/** Site path of the card for a page, such as /og/blog/evals.png or /og/default.png. */
export function ogImagePath(pathname: string): string {
  const match = CARD_PAGE.exec(pathname);
  return match ? `/og/${match[1]}/${match[2]}.png` : '/og/default.png';
}

/**
 * JSON for a <script type="application/ld+json"> block. "<" is escaped so text such as
 * "</script>" in a title cannot end the block; ">", "&" and the two JS line separators too.
 */
export function jsonLd(data: Record<string, unknown>): string {
  return JSON.stringify(data)
    .replace(/</g, '\\u003c')
    .replace(/>/g, '\\u003e')
    .replace(/&/g, '\\u0026')
    .replace(/\u2028/g, '\\u2028')
    .replace(/\u2029/g, '\\u2029');
}

/** tokens.css, read from disk: astro build and vitest both run in web/. */
export function readTokens(): Promise<string> {
  return readFile(join(process.cwd(), 'src/styles/tokens.css'), 'utf8');
}

/** schema.org Person for Skylar: the About page's JSON-LD and every post's author. */
export function person(site: URL): Record<string, unknown> {
  return { '@type': 'Person', name: AUTHOR_NAME, url: canonicalUrl('/', site), sameAs: [LINKEDIN_URL] };
}

export interface Post {
  pathname: string;
  title: string;
  summary: string;
  date: Date;
  updated: Date;
}

/** schema.org BlogPosting for an article page (Google Article: no required fields). */
export function blogPosting(post: Post, site: URL): Record<string, unknown> {
  const url = canonicalUrl(post.pathname, site);
  return {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline: post.title,
    description: post.summary,
    datePublished: post.date.toISOString().slice(0, 10),
    dateModified: post.updated.toISOString().slice(0, 10),
    author: person(site),
    image: new URL(ogImagePath(post.pathname), site).href,
    url,
    mainEntityOfPage: url,
  };
}

/** Light colours from tokens.css (cards have one palette, docs/design.md "颜色"). */
export function lightColors(css: string): Record<string, string> {
  const root = /:root\s*\{([^}]*)\}/.exec(css)?.[1] ?? '';
  const colors: Record<string, string> = {};
  for (const [, name, value] of root.matchAll(/--color-([\w-]+):\s*([^;]+);/g)) colors[name!] = value!.trim();
  for (const name of ['background', 'text', 'muted', 'accent', 'glow-teal']) {
    if (!colors[name]) throw new Error(`tokens.css has no --color-${name}`);
  }
  return colors;
}

export interface Card {
  title: string;
  /** Small label above the title, such as "Blog"; empty for the generic card. */
  label: string;
  /** Line after the name at the bottom. */
  tagline: string;
}

/** Static weights: satori renders a variable font at its default weight only (no bold). */
const FONT_FILES = { 400: 'instrument-sans-latin-400-normal.woff', 700: 'instrument-sans-latin-700-normal.woff' };

export async function loadFonts() {
  const require = createRequire(import.meta.url);
  const directory = join(dirname(require.resolve('@fontsource/instrument-sans/package.json')), 'files');
  return Promise.all(Object.entries(FONT_FILES).map(async ([weight, file]) => ({
    name: 'Instrument Sans',
    data: await readFile(join(directory, file)),
    weight: Number(weight) as 400 | 700,
    style: 'normal' as const,
  })));
}

type Node = { type: string; props: { style: Record<string, unknown>; children?: unknown } };
const h = (type: string, style: Record<string, unknown>, children?: unknown): Node =>
  ({ type, props: { style, children } });

export function cardTree(card: Card, colors: Record<string, string>): Node {
  const titleSize = card.title.length > 60 ? 56 : 68;
  return h('div', {
    width: OG_WIDTH,
    height: OG_HEIGHT,
    boxSizing: 'border-box',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    padding: '72px 80px',
    fontFamily: 'Instrument Sans',
    color: colors.text,
    backgroundColor: colors.background,
    backgroundImage: `linear-gradient(180deg, ${colors['glow-teal']} 0%, ${colors.background} 45%)`,
    borderTop: `12px solid ${colors.accent}`,
  }, [
    h('div', { display: 'flex', fontSize: 28, fontWeight: 700, color: colors.accent }, card.label || ' '),
    h('div', {
      display: 'block',
      width: OG_WIDTH - 160,
      fontSize: titleSize,
      fontWeight: 700,
      lineHeight: 1.15,
      letterSpacing: '-0.01em',
      lineClamp: 3,
    }, card.title),
    h('div', { display: 'flex', alignItems: 'flex-end' }, [
      h('div', { display: 'flex', fontSize: 34, fontWeight: 700, color: colors.text }, AUTHOR_NAME),
      h('div', { display: 'flex', fontSize: 28, fontWeight: 400, color: colors.muted, marginLeft: 20 }, card.tagline),
    ]),
  ]);
}

/** SVG of the card with text as paths (satori), so resvg needs no fonts at all. */
export async function renderOgSvg(card: Card): Promise<string> {
  const { default: satori } = await import('satori');
  // satori's types want a React element; the plain { type, props } tree is what it reads.
  return satori(cardTree(card, lightColors(await readTokens())) as unknown as Parameters<typeof satori>[0], {
    width: OG_WIDTH,
    height: OG_HEIGHT,
    fonts: await loadFonts(),
  });
}

export async function renderOgPng(card: Card): Promise<Uint8Array> {
  const { Resvg } = await import('@resvg/resvg-js');
  const svg = await renderOgSvg(card);
  return new Resvg(svg, { fitTo: { mode: 'original' }, font: { loadSystemFonts: false } }).render().asPng();
}
