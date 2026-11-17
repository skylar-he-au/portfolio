export function addSectionAnchors(html: string): string {
  // Astro has already escaped the HTML and assigned ids; preserve both verbatim.
  return html.replace(
    /(<h2\b[^>]*\sid=(["'])(.*?)\2[^>]*>)([\s\S]*?)(<\/h2>)/g,
    (_match: string, opening: string, _quote: string, id: string, heading: string, closing: string) =>
      `${opening}${heading} <a class="anchor" href="#${id}" aria-label="Link to this section">#</a>${closing}`,
  );
}
