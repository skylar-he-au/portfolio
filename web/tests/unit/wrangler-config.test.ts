import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const root = process.cwd();

describe('Cloudflare deployment configuration', () => {
  it('uses the API Worker and static assets with only a local origin default', () => {
    const config = JSON.parse(readFileSync(join(root, 'wrangler.jsonc'), 'utf8'));
    expect(config).toEqual({
      name: 'portfolio',
      main: 'worker/index.ts',
      compatibility_date: '2026-11-18',
      assets: {
        directory: './dist',
        run_worker_first: ['/api/*'],
        not_found_handling: '404-page',
        html_handling: 'auto-trailing-slash',
      },
      observability: { enabled: false },
      vars: { ORIGIN_URL: 'http://localhost:8000' },
    });
    expect(config).not.toHaveProperty('routes');
    expect(config).not.toHaveProperty('workers_dev');
    expect(existsSync(join(root, config.main))).toBe(true);
  });

  it('sets the static security headers in one rule within Cloudflare line limits', () => {
    const source = readFileSync(join(root, 'public/_headers'), 'utf8');
    const lines = source.trimEnd().split(/\r?\n/);
    expect(lines.filter((line) => !/^\s/.test(line))).toEqual(['/*']);
    expect(lines.slice(1).map((line) => line.trim())).toEqual([
      'Strict-Transport-Security: max-age=31536000; includeSubDomains',
      'X-Content-Type-Options: nosniff',
      'Referrer-Policy: strict-origin-when-cross-origin',
      'Permissions-Policy: camera=(), microphone=()',
      "Content-Security-Policy: frame-ancestors 'none'; base-uri 'self'; object-src 'none'; form-action 'self'",
    ]);
    expect(source).not.toContain('preload');
    for (const line of lines) expect(line.length).toBeLessThanOrEqual(2000);
  });
});
