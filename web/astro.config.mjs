// @ts-check
import { defineConfig } from 'astro/config';
import { loadEnv } from 'vite';

import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  // Placeholder: step 8 changes this together with EMAIL_DOMAIN_PARTS in web/src/lib/email.ts
  site: 'https://example.com',
  integrations: [react()],
  security: {
    csp: {
      directives: [
        "default-src 'self'",
        'frame-src https://challenges.cloudflare.com',
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
      ],
      scriptDirective: {
        resources: ["'self'", 'https://challenges.cloudflare.com'],
      },
      styleDirective: {
        resources: ["'self'", { resource: "'unsafe-inline'", kind: 'attribute' }],
      },
    },
  },
  vite: {
    server: {
      proxy: {
        '/api': {
          target: process.env.API_ORIGIN ?? 'http://localhost:8000',
          changeOrigin: true,
        },
      },
    },
    build: {
      // Fonts stay files: CSP default-src 'self' blocks data: fonts (Vite inlines files under
      // 4 kB, such as the JetBrains Mono Cyrillic-ext subset). Other assets use the default.
      assetsInlineLimit: (filePath) => (/\.(?:woff2?|ttf|otf|eot)$/i.test(filePath) ? false : undefined),
    },
    envDir: '..',
    plugins: [{
      name: 'dev-api-headers',
      apply: 'serve',
      config(_config, { mode }) {
        const secret = process.env.ORIGIN_SECRET ?? loadEnv(mode, '..', '').ORIGIN_SECRET;
        return {
          server: {
            proxy: {
              '/api': {
                headers: {
                  'X-Client-IP': '127.0.0.1',
                  ...(secret ? { 'X-Origin-Secret': secret } : {}),
                },
              },
            },
          },
        };
      },
    }],
  },
});
