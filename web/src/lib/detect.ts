export const ASK_MAX_CHARS = 500;
export type Detected =
  | { kind: 'ask' }
  | { kind: 'match'; source: 'link' | 'text' }
  | { kind: 'bad-link' };
export const BAD_LINK_MESSAGE =
  'Only Greenhouse and Lever job links can be read — paste the job ad text instead.';

const board = '[A-Za-z0-9_-]+';
const leverId = '[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}';
const suffix = '(?:\\?[^\\s#]*)?(?:#[^\\s]*)?';
const greenhousePath = new RegExp(`^/${board}/jobs/[0-9]+/?${suffix}$`, 'u');
const leverPath = new RegExp(`^/${board}/${leverId}(?:/apply)?/?${suffix}$`, 'u');

export function detectInput(raw: string): Detected {
  const text = raw.trim();
  if (/^https?:\/\/\S*$/.test(text)) {
    // Lowercase only the host: the scheme and path are case-sensitive in the backend.
    const parts = /^https:\/\/([^/]+)(\/.*)$/.exec(text);
    if (parts) {
      const host = parts[1]!.toLowerCase();
      const path = parts[2]!;
      if ((['boards.greenhouse.io', 'job-boards.greenhouse.io', 'job-boards.anz.greenhouse.io'].includes(host) && greenhousePath.test(path))
        || (['jobs.lever.co', 'jobs.eu.lever.co'].includes(host) && leverPath.test(path))) {
        return { kind: 'match', source: 'link' };
      }
    }
    return { kind: 'bad-link' };
  }
  return [...text].length > ASK_MAX_CHARS ? { kind: 'match', source: 'text' } : { kind: 'ask' };
}
