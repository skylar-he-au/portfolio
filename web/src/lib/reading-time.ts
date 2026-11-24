export function readingMinutes(markdown: string): number {
  let fenceChar = '';
  let fenceLength = 0;
  const prose = markdown.replace(/<!--[\s\S]*?(?:-->|$)/g, '').split(/\r?\n/)
    .filter((line) => {
      const fence = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
      if (fence) {
        const marker = fence[1]!;
        const rest = fence[2]!;
        if (fenceChar) {
          if (marker[0] === fenceChar && marker.length >= fenceLength && !rest.trim()) {
            fenceChar = '';
          }
        } else if (marker[0] !== '`' || !rest.includes('`')) {
          fenceChar = marker[0]!;
          fenceLength = marker.length;
        } else {
          return true;
        }
        return false;
      }
      return !fenceChar;
    }).join('\n')
    // Keep link labels, but destinations and markup do not add reading time.
    .replace(/^ {0,3}\[[^\]]+\]:.*$/gm, '')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/!?\[([^\]]*)\]\[[^\]]*\]/g, '$1')
    .replace(/<[^>]*>/g, '')
    .replace(/[`*_{}\[\]()#+.!>|~\\-]/g, '');
  const words = prose.match(/\S+/g)?.length ?? 0;
  return Math.max(1, Math.ceil(words / 200));
}
