import { mkdirSync } from 'node:fs';
import { dirname, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

export function parseShotArgs(argv) {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    strict: true,
    options: {
      width: { type: 'string', default: '1280' },
      height: { type: 'string', default: '800' },
      scheme: { type: 'string', default: 'light' },
      full: { type: 'boolean', default: false },
      out: { type: 'string' },
    },
  });

  if (positionals.length === 0) throw new Error('URL is required');
  if (positionals.length !== 1) throw new Error('Expected exactly one URL');
  const url = positionals[0];
  let parsedUrl;
  try {
    parsedUrl = new URL(url);
  } catch {
    throw new Error('URL must be a valid http: or https: URL');
  }
  if (!['http:', 'https:'].includes(parsedUrl.protocol)) {
    throw new Error('URL must use http: or https:');
  }

  const dimensions = {};
  for (const name of ['width', 'height']) {
    const value = Number(values[name]);
    if (!/^\d+$/.test(values[name]) || !Number.isInteger(value) || value < 200 || value > 4000) {
      throw new Error(`--${name} must be an integer from 200 to 4000`);
    }
    dimensions[name] = value;
  }

  const scheme = values.scheme;
  if (scheme !== 'light' && scheme !== 'dark') {
    throw new Error('--scheme must be light or dark');
  }
  const out = values.out;
  if (!out) throw new Error('--out is required');
  if (!['.png', '.jpg', '.jpeg'].includes(extname(out).toLowerCase())) {
    throw new Error('--out must have a .png, .jpg or .jpeg extension');
  }

  return { url, width: dimensions.width, height: dimensions.height, scheme, full: values.full, out };
}

async function takeShot({ url, width, height, scheme, full, out }) {
  mkdirSync(dirname(out), { recursive: true });
  const { chromium } = await import('playwright-core');
  const browser = await chromium.launch(process.env.SHOT_BROWSER_PATH
    ? { executablePath: process.env.SHOT_BROWSER_PATH }
    : { channel: 'chrome' });
  try {
    const context = await browser.newContext({
      viewport: { width, height },
      colorScheme: scheme,
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();
    await page.goto(url, { waitUntil: 'networkidle', timeout: 30_000 });
    await page.evaluate(() => document.fonts.ready);
    const jpeg = ['.jpg', '.jpeg'].includes(extname(out).toLowerCase());
    await page.screenshot({
      path: out,
      type: jpeg ? 'jpeg' : 'png',
      ...(jpeg ? { quality: 85 } : {}),
      fullPage: full,
    });
    const { scrollWidth, horizontalOverflow } = await page.evaluate(() => ({
      scrollWidth: document.documentElement.scrollWidth,
      horizontalOverflow: document.documentElement.scrollWidth > innerWidth,
    }));
    console.log(`saved ${out} (${width}x${height}, ${scheme}) scrollWidth=${scrollWidth} horizontalOverflow=${horizontalOverflow}`);
  } finally {
    await browser.close();
  }
}

function reportError(error) {
  console.error((error instanceof Error ? error.message : String(error)).replace(/\s+/g, ' '));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  let options;
  try {
    options = parseShotArgs(process.argv.slice(2));
  } catch (error) {
    reportError(error);
    process.exit(2);
  }
  try {
    await takeShot(options);
  } catch (error) {
    reportError(error);
    process.exitCode = 1;
  }
}
