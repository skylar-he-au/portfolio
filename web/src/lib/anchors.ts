/**
 * Give every h2 with an id a "#" link to its own section.
 *
 * The heading is named by its own words only (`aria-labelledby` points at a span holding them),
 * so browsing by headings reads "In 30 seconds" without the link's name after it.
 * The link is named after the section ("Link to In 30 seconds"), so a list of links tells them apart.
 * The span id is the heading id plus ":title": heading ids are slugs, which never contain ":",
 * so it cannot clash with another heading's id on the page.
 */
export function addSectionAnchors(html: string): string {
  // Astro has already escaped the HTML and assigned ids; preserve both verbatim.
  return html.replace(
    /(<h2\b[^>]*\sid=(["'])(.*?)\2[^>]*>)([\s\S]*?)(<\/h2>)/g,
    (_match: string, opening: string, _quote: string, id: string, heading: string, closing: string) => {
      const titleId = `${id}:title`;
      return `${opening.slice(0, -1)} aria-labelledby="${titleId}"><span id="${titleId}">${heading}</span> `
        + `<a class="anchor" href="#${id}" aria-label="Link to ${attributeText(heading)}">#</a>${closing}`;
    },
  );
}

/** The heading's words for an attribute value: tags dropped, whitespace collapsed, entities left escaped. */
function attributeText(html: string): string {
  return html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim().replace(/"/g, '&quot;');
}
