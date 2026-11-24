// @vitest-environment node
// Code blocks: one light and one dark Shiki theme, every token at WCAG AA (4.5:1) on the
// site's --color-code-background (docs/blog.md "页面", Shiki). Uses the real astro.config.mjs.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { codeToHtml, codeToTokens } from 'shiki';
import { describe, expect, it, vi } from 'vitest';

vi.mock('astro/config', () => ({ defineConfig: (config: unknown) => config }));
vi.mock('@astrojs/react', () => ({ default: () => 'react' }));

const tokens = readFileSync(join(process.cwd(), 'src/styles/tokens.css'), 'utf8');
const darkBlock = /@media \(prefers-color-scheme: dark\)\s*\{\s*:root\s*\{([^}]*)\}/.exec(tokens)?.[1] ?? '';
const lightBlock = /:root\s*\{([^}]*)\}/.exec(tokens)?.[1] ?? '';
const codeBackground = (block: string) => /--color-code-background:\s*(#[\da-f]{6});/i.exec(block)?.[1];

function luminance(hex: string): number {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r! + 0.7152 * g! + 0.0722 * b!;
}

function contrast(a: string, b: string): number {
  const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (high! + 0.05) / (low! + 0.05);
}

/** #rrggbbaa over the background, as the browser paints it. */
function blend(color: string, background: string): string {
  if (color.length !== 9) return color.slice(0, 7);
  const alpha = parseInt(color.slice(7, 9), 16) / 255;
  return `#${[1, 3, 5].map((i) => Math.round(
    parseInt(color.slice(i, i + 2), 16) * alpha + parseInt(background.slice(i, i + 2), 16) * (1 - alpha),
  ).toString(16).padStart(2, '0')).join('')}`;
}

const samples: Record<string, string> = {
  python: 'from typing import Any\n\n@dataclass\nclass Run:\n    """Docstring."""\n    score: float = 0.9  # comment\n\ndef grade(x: dict[str, Any]) -> bool:\n    if x["n"] > 3 and not None:\n        return f"ok {x!r}" == \'b\'\n    raise ValueError(r"\\d+")\n',
  ts: '// comment\nimport { a } from \'./b\';\nexport async function f<T>(x: T, y = 1): Promise<number> {\n  const re = /a+/g; let s = `t ${x}`;\n  if (x === null || y > 2) return 0x1f;\n  class A extends B { #p = true; static m() { this.p?.(); } }\n  return JSON.parse(\'{"a":1}\') as number;\n}\n',
  bash: '#!/usr/bin/env bash\n# comment\nexport FOO="bar $HOME"\nnpm run build && echo \'done\' | grep -E "^d" > /tmp/x 2>&1\nfor i in $(seq 1 3); do echo $i; done\n',
  json: '{"name": "web", "n": 1, "ok": true, "x": null}',
  yaml: '# c\nname: ci\non:\n  push:\n    branches: [main]\njobs:\n  a: { runs-on: ubuntu-latest, n: 3, b: true }\n',
  sql: "-- c\nSELECT id, count(*) AS n FROM docs WHERE title = 'x' AND n > 3 GROUP BY 1;\n",
  diff: '--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n',
  markdown: '# Title\n\n**bold** _em_ `code` [link](https://x.y)\n- item\n',
};

async function shikiConfig() {
  const config = (await import('../../astro.config.mjs')).default as unknown as {
    markdown: { shikiConfig: { themes: { light: object; dark: string }; defaultColor: string; transformers: object[] } };
  };
  return config.markdown.shikiConfig;
}

describe('code highlighting', () => {
  it.each([['light', lightBlock], ['dark', darkBlock]] as const)(
    'keeps every %s token at 4.5:1 or more on the code background',
    async (mode, block) => {
      const background = codeBackground(block);
      expect(background, mode).toBeDefined();
      const { themes } = await shikiConfig();
      const failures: string[] = [];
      for (const [lang, code] of Object.entries(samples)) {
        const { tokens: lines } = await codeToTokens(code, { lang: lang as never, theme: themes[mode] as never });
        for (const token of lines.flat()) {
          if (!token.content.trim() || !token.color) continue;
          const color = blend(token.color.toLowerCase(), background!);
          const ratio = contrast(color, background!);
          if (ratio < 4.5) failures.push(`${lang} ${JSON.stringify(token.content)} ${color} ${ratio.toFixed(2)}`);
        }
      }
      expect(failures).toEqual([]);
    },
  );

  it('colours tokens with light-dark() and leaves the background to global.css', async () => {
    const { themes, defaultColor, transformers } = await shikiConfig();
    const html = await codeToHtml('const a = 1; // c', {
      lang: 'ts', themes: themes as never, defaultColor: defaultColor as never, transformers: transformers as never,
    });
    const pre = /<pre[^>]*style="([^"]*)"/.exec(html)?.[1] ?? '';
    expect(pre).not.toMatch(/(?:^|;)\s*background(?:-color)?\s*:/);
    expect(html).toMatch(/<span style="color:light-dark\(#[\da-f]{6}, #[\da-f]{6}\)/i);
  });
});
