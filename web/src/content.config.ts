import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

// The ingest reads front matter line by line as "key: value" (api/app/ingest.py read_page),
// so nested keys must never be title, rag, summary or updated.
const proof = z.object({
  /** A figure the page body states, such as "154 / 164". YAML reads a bare 62 as a number. */
  value: z.coerce.string().min(1),
  /** One line saying what the figure counts. */
  label: z.string().min(1),
}).strict();

const pages = defineCollection({
  loader: glob({ base: '../content', pattern: '**/*.md' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    updated: z.coerce.date(),
    rag: z.boolean(),
    /** Small label: above the home headline, on a project card and above a project page title. */
    kicker: z.string().min(1).optional(),
    /** Result figures: the home card shows the first, the project page all of them. */
    proof: z.array(proof).min(1).max(3).optional(),
  }),
});

export const collections = { pages };
