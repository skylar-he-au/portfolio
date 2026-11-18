// @ts-check
import { defineConfig } from 'astro/config';
import { loadEnv } from 'vite';

import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  integrations: [react()],
  vite: {
    server: {
      proxy: {
        '/api': {
          target: process.env.API_ORIGIN ?? 'http://localhost:8000',
          changeOrigin: true,
        },
      },
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
