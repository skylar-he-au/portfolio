import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { expect, it } from 'vitest';
import { emailAddress } from '../../src/lib/email';

const dist = join(process.cwd(), 'dist');
if (!existsSync(dist)) throw new Error('Missing dist/: run `npm run build` first');

function filesWithin(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesWithin(path) : [path];
  });
}

it('exposes neither the email address nor a mailto URL in any built file', () => {
  const files = filesWithin(dist);
  expect(files.length).toBeGreaterThan(0);
  for (const path of files) {
    const source = readFileSync(path, 'utf8');
    expect(source, relative(dist, path)).not.toContain(emailAddress());
    expect(source, relative(dist, path)).not.toContain('mailto:');
  }
});
