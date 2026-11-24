import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = process.cwd();

function filesWithin(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesWithin(path) : [path];
  });
}

describe('source constraints', () => {
  it('keeps literal colors in tokens.css', () => {
    const violations: string[] = [];
    for (const path of filesWithin(join(root, 'src'))) {
      const name = relative(root, path);
      if (name === 'src/styles/tokens.css') continue;
      const source = readFileSync(path, 'utf8');
      // Scan CSS and Astro style contexts so fragment links and heading anchors
      // cannot be mistaken for colors. Boundaries reject partial hex matches.
      const styles = path.endsWith('.css') ? [source] : path.endsWith('.astro')
        ? [
          ...Array.from(source.matchAll(/<style\b[^>]*>([\s\S]*?)<\/style>/gi), (match) => match[1]!),
          ...Array.from(source.matchAll(/\bstyle\s*=\s*(["'])([\s\S]*?)\1/gi), (match) => match[2]!),
        ] : [];
      for (const style of styles) {
        const colors = style.match(/#(?:[\da-f]{8}|[\da-f]{6}|[\da-f]{4}|[\da-f]{3})(?![\w-])|\b(?:rgba?|hsl)\s*\(/gi);
        if (colors) violations.push(`${name}: ${colors.join(', ')}`);
      }
    }
    expect(violations).toEqual([]);
  });

  it('configures the API proxy and preserves the React integration', () => {
    const config = readFileSync(join(root, 'astro.config.mjs'), 'utf8');
    expect(config).toMatch(/integrations:\s*\[react\(\),\s*sitemap\(/);
    expect(config).toMatch(/vite:\s*\{\s*server:\s*\{\s*proxy:\s*\{\s*['"]\/api['"]:\s*\{/);
    expect(config).toMatch(/target:\s*process\.env\.API_ORIGIN\s*\?\?\s*['"]http:\/\/localhost:8000['"]/);
    expect(config).toMatch(/changeOrigin:\s*true/);
    expect(config).not.toMatch(/\b(?:rewrite|host)\s*:/);
  });
});
