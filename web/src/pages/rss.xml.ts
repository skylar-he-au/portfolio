import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { getCollection } from 'astro:content';

export async function GET(context: APIContext) {
  if (!context.site) throw new Error('RSS requires site in astro.config.mjs');
  const origin = context.site.origin;
  const articles = (await getCollection('pages', ({ id, data }) => id.startsWith('blog/') && !data.draft))
    .map((entry) => {
      if (!entry.data.date) throw new Error(`Missing front matter: ${entry.id} date`);
      if (!entry.rendered) throw new Error(`Missing rendered content: ${entry.id}`);
      return {
        title: entry.data.title,
        description: entry.data.summary,
        pubDate: entry.data.date,
        link: `/${entry.id}`,
        // Feed readers need absolute local resources; preserve the rendered HTML otherwise.
        content: entry.rendered.html.replace(/<!--[\s\S]*?-->|<\/?[a-z][^>]*>/gi, (tag) =>
          tag.startsWith('<!--') ? tag : tag.replace(
            /(\s(?:href|src)\s*=\s*)(["'])(\/(?!\/)[^"']*)\2/gi,
            (_match: string, attribute: string, quote: string, path: string) =>
              `${attribute}${quote}${origin}${path}${quote}`,
          )),
      };
    })
    .sort((a, b) => b.pubDate.getTime() - a.pubDate.getTime());
  return rss({
    title: 'Skylar He — Blog',
    description: 'Notes on building projects and the decisions behind them.',
    site: context.site,
    trailingSlash: false,
    items: articles,
  });
}
