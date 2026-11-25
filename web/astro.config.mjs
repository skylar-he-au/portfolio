// @ts-check
import { defineConfig } from 'astro/config';
import { loadEnv } from 'vite';

import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import { bundledThemes } from 'shiki/themes';

import { withoutTrailingSlash } from './src/lib/og.ts';
import { satteri } from '@astrojs/markdown-satteri';
import { tableScroll } from './src/lib/table-scroll.ts';

// Code blocks sit on the site's --color-code-background (#edf1f4 light, #1e252e dark), so the
// themes are checked against those (tests/unit/code-theme.test.ts, WCAG AA 4.5:1). GitHub's
// light theme has two colours under 4.5:1 there (comments 4.00, purple 4.44); they are
// darkened to 5.38 and 6.47. GitHub's dark theme passes as it is (lowest 5.02).
const githubLight = (await bundledThemes['github-light-default']()).default;
/** @type {{ light: import('shiki').ThemeRegistrationRaw, dark: import('shiki').BundledTheme }} */
export const codeThemes = {
  light: {
    ...JSON.parse(JSON.stringify(githubLight).replace(/#6e7781/gi, '#59636e').replace(/#8250df/gi, '#6639ba')),
    name: 'github-light-site',
  },
  dark: 'github-dark-default',
};

// https://astro.build/config
export default defineConfig({
  // Placeholder: step 8 changes this together with EMAIL_DOMAIN_PARTS in web/src/lib/email.ts
  // and the Sitemap line in web/public/robots.txt (tests/dist/site.test.ts checks robots.txt).
  site: 'https://example.com',
  integrations: [react(), sitemap({
    // 404 is left out by the integration; endpoints (rss.xml, og/*.png) are not pages.
    serialize: (item) => ({ ...item, url: withoutTrailingSlash(item.url) }),
  })],
  markdown: {
    // Astro's default processor (Sätteri) plus one hast plugin: each table in a focusable,
    // named box that scrolls sideways when the table is wide.
    processor: satteri({ hastPlugins: [tableScroll] }),
    shikiConfig: {
      themes: codeThemes,
      // Each token gets color: light-dark(light, dark); tokens.css sets color-scheme per mode.
      defaultColor: 'light-dark()',
      transformers: [{
        name: 'site-code-background',
        // Drop the theme's background so global.css gives <pre> --color-code-background.
        pre(node) {
          const style = String(node.properties.style ?? '');
          node.properties.style = style.split(';').filter((rule) => !/^\s*background(?:-color)?\s*:/i.test(rule)).join(';');
        },
      }],
    },
  },
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
