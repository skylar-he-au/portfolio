import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const dist = join(process.cwd(), 'dist');
if (!existsSync(dist)) throw new Error('Missing dist/: run `npm run build` first');

function htmlFiles(directory: string): string[] {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? htmlFiles(path) : path.endsWith('.html') ? [path] : [];
  });
}

const pages = htmlFiles(dist);
it('finds built HTML pages to validate', () => { expect(pages.length).toBeGreaterThan(0); });

describe.each(pages.map((path) => [relative(dist, path), path]))('CSP in %s', (_name, path) => {
  it('allows only the required sources and hashes every inline script and style element', () => {
    const document = new DOMParser().parseFromString(readFileSync(path!, 'utf8'), 'text/html');
    const metas = [...document.querySelectorAll('meta[http-equiv]')]
      .filter((meta) => meta.getAttribute('http-equiv')?.toLowerCase() === 'content-security-policy');
    expect(metas).toHaveLength(1);
    const policy = metas[0]!.getAttribute('content')!;
    const directives = new Map(policy.split(';').filter((part) => part.trim()).map((part) => {
      const [name, ...sources] = part.trim().split(/\s+/);
      return [name!, sources];
    }));
    expect(directives.get('default-src')).toEqual(["'self'"]);
    expect(directives.get('frame-src')).toEqual(['https://challenges.cloudflare.com']);
    expect(directives.get('connect-src')).toEqual(["'self'"]);
    expect(directives.get('object-src')).toEqual(["'none'"]);
    expect(directives.get('base-uri')).toEqual(["'self'"]);
    expect(directives.get('form-action')).toEqual(["'self'"]);
    expect(directives.get('style-src-attr')).toEqual(["'unsafe-inline'"]);

    const scripts = directives.get('script-src')!;
    expect(scripts).toContain("'self'");
    expect(scripts).toContain('https://challenges.cloudflare.com');
    for (const [name, sources] of directives) {
      if (!/^(?:script|style)-src(?:-elem|-attr)?$/.test(name)) continue;
      if (name.startsWith('script-')) {
        expect(sources).not.toContain("'unsafe-inline'");
        expect(sources).not.toContain("'unsafe-eval'");
      }
      if (name.endsWith('-attr')) continue;
      for (const source of sources) {
        expect(source).toMatch(/^(?:'self'|https:\/\/challenges\.cloudflare\.com|'sha256-[A-Za-z0-9+/]+=*')$/);
      }
    }
    const styles = directives.get('style-src')!;
    expect(styles).toContain("'self'");
    for (const script of document.querySelectorAll('script:not([src])')) {
      const hash = createHash('sha256').update(script.textContent ?? '').digest('base64');
      expect(scripts).toContain(`'sha256-${hash}'`);
    }
    for (const style of document.querySelectorAll('style')) {
      const hash = createHash('sha256').update(style.textContent ?? '').digest('base64');
      expect(directives.get('style-src-elem') ?? styles).toContain(`'sha256-${hash}'`);
    }
    for (const element of document.querySelectorAll('*')) {
      expect(element.getAttributeNames().filter((name) => /^on/i.test(name)), element.outerHTML).toEqual([]);
    }
  });
});
