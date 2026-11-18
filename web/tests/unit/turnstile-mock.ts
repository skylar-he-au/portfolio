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

export function mockTurnstile(autoLoad = true) {
  let options: WidgetOptions;
  const api = {
    ready: vi.fn((callback: () => void) => callback()),
    render: vi.fn((_container: HTMLElement, value: WidgetOptions) => { options = value; return 'widget-1'; }),
    execute: vi.fn((_id: string) => { options.callback('turnstile-token'); }),
    reset: vi.fn((_id: string) => {}),
    remove: vi.fn((_id: string) => {}),
  };
  vi.stubGlobal('turnstile', api);
  const append = document.head.appendChild.bind(document.head);
  vi.spyOn(document.head, 'appendChild').mockImplementation(<T extends Node>(node: T): T => {
    append(node);
    if (autoLoad && node instanceof HTMLScriptElement) queueMicrotask(() => node.dispatchEvent(new Event('load')));
    return node;
  });
  return { api, options: () => options };
}

export function turnstileScripts() {
  return document.querySelectorAll<HTMLScriptElement>('script[src^="https://challenges.cloudflare.com/"]');
}
