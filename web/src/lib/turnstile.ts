interface TurnstileOptions {
  sitekey: string;
  execution: 'execute';
  action: 'ask';
  appearance: 'execute';
  callback: (token: string) => void;
  'error-callback': () => void;
  'expired-callback': () => void;
}

interface Turnstile {
  ready: (callback: () => void) => void;
  render: (container: HTMLElement, options: TurnstileOptions) => string;
  execute: (widgetId: string) => void;
  reset: (widgetId: string) => void;
  remove: (widgetId: string) => void;
}

declare global {
  interface Window {
    turnstile?: Turnstile;
  }
}

export interface WidgetController {
  execute: () => Promise<string>;
  reset: () => void;
  remove: () => void;
}

let scriptPromise: Promise<Turnstile> | undefined;

export function loadTurnstile(): Promise<Turnstile> {
  if (!scriptPromise) {
    scriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
      script.async = true;
      script.addEventListener('load', () => {
        if (window.turnstile) resolve(window.turnstile);
        else reject(new Error('Turnstile is unavailable'));
      }, { once: true });
      script.addEventListener('error', () => reject(new Error('Turnstile failed to load')), { once: true });
      document.head.appendChild(script);
    });
  }
  return scriptPromise;
}

export function renderWidget(container: HTMLElement, sitekey: string): WidgetController {
  let api: Turnstile | undefined;
  let widgetId: string | undefined;
  let removed = false;
  let resetOnReady = false;
  let pending: { resolve: (token: string) => void; reject: (error: unknown) => void; timer: ReturnType<typeof setTimeout> } | undefined;
  let resolveReady!: () => void;
  let rejectReady!: (error: unknown) => void;
  const ready = new Promise<void>((resolve, reject) => { resolveReady = resolve; rejectReady = reject; });
  // A focus may load the widget without ever executing it.
  void ready.catch(() => {});

  function fail(error: unknown) {
    if (!pending) return;
    clearTimeout(pending.timer);
    pending.reject(error);
    pending = undefined;
  }

  void loadTurnstile().then((loaded) => {
    if (removed) return;
    api = loaded;
    api.ready(() => {
      if (removed) return;
      try {
        widgetId = loaded.render(container, {
          sitekey,
          execution: 'execute',
          action: 'ask',
          appearance: 'execute',
          callback: (token) => {
            if (!pending) return;
            clearTimeout(pending.timer);
            pending.resolve(token);
            pending = undefined;
          },
          'error-callback': () => fail(new Error('Turnstile failed')),
          'expired-callback': () => fail(new Error('Turnstile expired')),
        });
        if (resetOnReady) loaded.reset(widgetId);
        resolveReady();
      } catch (error) {
        rejectReady(error);
      }
    });
  }).catch(rejectReady);

  return {
    execute() {
      if (removed) return Promise.reject(new Error('Turnstile was removed'));
      if (pending) return Promise.reject(new Error('Turnstile is already executing'));
      return new Promise<string>((resolve, reject) => {
        const execution = { resolve, reject, timer: setTimeout(() => fail(new Error('Turnstile timed out')), 20_000) };
        pending = execution;
        void ready.then(() => {
          if (pending === execution && !removed && widgetId !== undefined) api?.execute(widgetId);
        }).catch((error: unknown) => {
          if (pending === execution) fail(error);
        });
      });
    },
    reset() {
      fail(new Error('Turnstile was reset'));
      if (removed) return;
      if (widgetId !== undefined) api?.reset(widgetId);
      else resetOnReady = true;
    },
    remove() {
      if (removed) return;
      removed = true;
      const error = new DOMException('Turnstile was removed', 'AbortError');
      rejectReady(error);
      fail(error);
      if (widgetId !== undefined) api?.remove(widgetId);
    },
  };
}
