// Share preview cards (docs/blog.md "网站部件"): /og/blog/<slug>.png and /og/projects/<slug>.png
// for each published article and project page, /og/default.png for every other page.
// Built at build time from local fonts (src/lib/og.ts); Base.astro picks the card per page.
import type { APIRoute, GetStaticPaths } from 'astro';
import { getCollection, getEntry } from 'astro:content';
import { renderOgPng, type Card } from '../../lib/og';

export const getStaticPaths = (async () => {
  const home = await getEntry('pages', 'home');
  if (!home?.data.kicker) throw new Error('Missing front matter: home kicker');
  // "AI and full-stack developer · Sydney": the role goes under the name on page cards; the
  // generic card shows the role large and the city under the name.
  const [role, place = ''] = home.data.kicker.split(' · ');
  if (!role) throw new Error('Missing front matter: home kicker role');
  const tagline = role;
  const pages = await getCollection('pages', ({ id, data }) =>
    id.startsWith('projects/') || (id.startsWith('blog/') && !data.draft));
  return [
    ...pages.map((page) => ({
      params: { slug: page.id },
      props: { title: page.data.title, label: page.id.startsWith('blog/') ? 'Blog' : 'Project', tagline },
    })),
    { params: { slug: 'default' }, props: { title: role, label: '', tagline: place } },
  ];
}) satisfies GetStaticPaths;

export const GET: APIRoute<Card> = async ({ props }) =>
  new Response(Buffer.from(await renderOgPng(props)), { headers: { 'Content-Type': 'image/png' } });
