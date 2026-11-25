/**
 * Sätteri hast plugin (astro.config.mjs, `markdown.processor`): each Markdown table goes inside
 * <div class="table-wrap" tabindex="0" role="region" aria-label="Table: <heading>">.
 *
 * A table wider than the column scrolls sideways inside that div (global.css). Chrome and
 * Firefox let a keyboard reach a scrolling box on their own; Safari does not, so the div is
 * focusable, and as a focusable region it needs a name a screen reader can read: the nearest
 * heading above the table ("Table: How RoleRadar matches jobs"), numbered when one section has
 * several. The table itself keeps its own role.
 *
 * Only a type import from `satteri` (Astro's own dependency); at runtime the plugin is a plain
 * object: a `before` hook and a filtered `element` visitor.
 */

import type { HastPluginDefinition } from 'satteri';

interface HastNodeLike {
  type: string;
  tagName?: string;
  value?: string;
  children?: readonly HastNodeLike[];
}

const HEADING = /^h[1-6]$/;
const DATA_KEY = 'tableScrollLabels';

function textOf(node: HastNodeLike): string {
  if (node.type === 'text') return node.value ?? '';
  return (node.children ?? []).map(textOf).join('');
}

/** The heading text above each table, in document order ('' before any heading). */
export function tableHeadings(root: HastNodeLike): string[] {
  const headings: string[] = [];
  let current = '';
  function walk(node: HastNodeLike) {
    for (const child of node.children ?? []) {
      if (child.type !== 'element') continue;
      if (HEADING.test(child.tagName ?? '')) current = textOf(child).replace(/\s+/g, ' ').trim();
      else if (child.tagName === 'table') headings.push(current);
      else walk(child);
    }
  }
  walk(root);
  return headings;
}

export function tableLabels(headings: string[]): string[] {
  const totals = new Map<string, number>();
  for (const heading of headings) totals.set(heading, (totals.get(heading) ?? 0) + 1);
  const seen = new Map<string, number>();
  return headings.map((heading) => {
    const index = (seen.get(heading) ?? 0) + 1;
    seen.set(heading, index);
    const total = totals.get(heading)!;
    const base = heading ? `Table: ${heading}` : 'Table';
    return total > 1 ? `${base} (${index} of ${total})` : base;
  });
}

export const tableScroll: HastPluginDefinition = {
  name: 'table-scroll',
  before(root, ctx) {
    (ctx.data as Record<string, unknown>)[DATA_KEY] = tableLabels(tableHeadings(root as HastNodeLike));
  },
  element: {
    filter: ['table'],
    visit(node, ctx) {
      const labels = ((ctx.data as Record<string, unknown>)[DATA_KEY] ?? []) as string[];
      ctx.wrapNode(node, {
        type: 'element',
        tagName: 'div',
        properties: { className: ['table-wrap'], tabIndex: 0, role: 'region', ariaLabel: labels.shift() ?? 'Table' },
        children: [],
      });
    },
  },
};
