import { vi } from 'vitest';

export interface WidgetOptions {
  sitekey: string;
  execution: string;
  action: string;
  appearance: string;
  callback: (token: string) => void;
  'error-callback': () => void;
  'expired-callback': () => void;
}

// The real api.js throws this from turnstile.ready() when its script tag is async or deferred.
export const READY_ASYNC_ERROR = '[Cloudflare Turnstile] Remove async/defer from the Turnstile api.js script tag before using turnstile.ready().';

export function turnstileScripts() {
  return document.querySelectorAll<HTMLScriptElement>('script[src^="https://challenges.cloudflare.com/"]');
}

// Behaves like the real api.js: `window.turnstile` appears only once the script runs, the
// `onload` query parameter names a global callback it calls, then the script's load event fires.
export function mockTurnstile(autoLoad = true) {
  let options: WidgetOptions;
  let auto = autoLoad;
  const api = {
    ready: vi.fn((callback: () => void) => {
      const script = turnstileScripts()[0];
      if (!script || script.async || script.defer) throw new Error(READY_ASYNC_ERROR);
      callback();
    }),
    render: vi.fn((_container: HTMLElement, value: WidgetOptions) => { options = value; return 'widget-1'; }),
    execute: vi.fn((_id: string) => { options.callback('turnstile-token'); }),
    reset: vi.fn((_id: string) => {}),
    remove: vi.fn((_id: string) => {}),
  };

  function load(script = turnstileScripts()[0]) {
    vi.stubGlobal('turnstile', api);
    const onload = new URL(script.src).searchParams.get('onload');
    const callback = onload ? (window as unknown as Record<string, unknown>)[onload] : undefined;
    if (typeof callback === 'function') callback();
    script.dispatchEvent(new Event('load'));
  }

  const append = document.head.appendChild.bind(document.head);
  vi.spyOn(document.head, 'appendChild').mockImplementation(<T extends Node>(node: T): T => {
    append(node);
    if (auto && node instanceof HTMLScriptElement) queueMicrotask(() => load(node));
    return node;
  });
  return {
    api,
    options: () => options,
    // Keep the script loading until `load()` is called.
    holdLoad: () => { auto = false; },
    load,
  };
}
