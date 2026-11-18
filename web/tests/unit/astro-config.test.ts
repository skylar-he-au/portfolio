import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const loadEnv = vi.hoisted(() => vi.fn(() => ({ ORIGIN_SECRET: 'dotenv-secret' })));
vi.mock('vite', () => ({ loadEnv }));
vi.mock('astro/config', () => ({ defineConfig: (config: unknown) => config }));
vi.mock('@astrojs/react', () => ({ default: () => 'react' }));

interface Config {
  vite: {
    envDir: string;
    server: { proxy: { '/api': { target: string; changeOrigin: boolean; headers?: Record<string, string> } } };
    plugins: {
      apply: string;
      config: (config: unknown, env: { mode: string; command: string }) => {
        server: { proxy: { '/api': { headers: Record<string, string> } } };
      };
    }[];
  };
}

async function config() {
  vi.resetModules();
  // The production module is JavaScript; the shape here describes the tested hooks.
  return (await import('../../astro.config.mjs')).default as unknown as Config;
}

beforeEach(() => { vi.stubEnv('API_ORIGIN', undefined); });
afterEach(() => { vi.unstubAllEnvs(); vi.clearAllMocks(); });

describe('development API proxy environment', () => {
  it('loads public values from the repository root and limits secret loading to serve', async () => {
    const { vite } = await config();
    expect(vite.envDir).toBe('..');
    expect(vite.server.proxy['/api']).toMatchObject({ target: 'http://localhost:8000', changeOrigin: true });
    expect(vite.server.proxy['/api'].headers).toBeUndefined();
    expect(loadEnv).not.toHaveBeenCalled();
    expect(vite.plugins[0].apply).toBe('serve');
  });

  it.each(['host-secret', undefined, ''])('uses process.env first for %j and never sends undefined', async (secret) => {
    vi.stubEnv('ORIGIN_SECRET', secret);
    vi.stubEnv('API_ORIGIN', 'http://api:8000');
    const { vite } = await config();
    const result = vite.plugins[0].config({}, { mode: 'development', command: 'serve' });
    expect(result.server.proxy['/api'].headers).toEqual({
      'X-Client-IP': '127.0.0.1',
      ...(secret === '' ? {} : { 'X-Origin-Secret': secret ?? 'dotenv-secret' }),
    });
    expect(vite.server.proxy['/api'].target).toBe('http://api:8000');
    if (secret === undefined) expect(loadEnv).toHaveBeenCalledWith('development', '..', '');
    else expect(loadEnv).not.toHaveBeenCalled();
  });

  it('omits the origin header when both sources are unset', async () => {
    vi.stubEnv('ORIGIN_SECRET', undefined);
    loadEnv.mockReturnValueOnce({ ORIGIN_SECRET: '' });
    const { vite } = await config();
    const result = vite.plugins[0].config({}, { mode: 'development', command: 'serve' });
    expect(result.server.proxy['/api'].headers).toEqual({ 'X-Client-IP': '127.0.0.1' });
  });

  it('passes the two host values to only the compose web service', () => {
    const compose = readFileSync(join(process.cwd(), '..', 'docker-compose.yml'), 'utf8');
    const web = compose.split('\n  web:')[1];
    expect(web).toContain('ORIGIN_SECRET: ${ORIGIN_SECRET:-}');
    expect(web).toContain('PUBLIC_TURNSTILE_SITEKEY: ${PUBLIC_TURNSTILE_SITEKEY:-}');
  });
});
