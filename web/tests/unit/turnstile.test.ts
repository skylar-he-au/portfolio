import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mockTurnstile, turnstileScripts } from './turnstile-mock';

let turnstile: typeof import('../../src/lib/turnstile');
let widget: ReturnType<typeof mockTurnstile>;

beforeEach(async () => {
  vi.resetModules();
  widget = mockTurnstile(false);
  turnstile = await import('../../src/lib/turnstile');
});

afterEach(() => {
  for (const script of turnstileScripts()) script.remove();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

function startWidget() {
  return turnstile.renderWidget(document.createElement('div'), 'sitekey');
}

async function loadScript() {
  widget.load();
  await Promise.resolve();
  await Promise.resolve();
}

async function renderWidget() {
  const controller = startWidget();
  await loadScript();
  return controller;
}

describe('Turnstile controller', () => {
  it('loads the explicit script lazily and shares one promise and script', async () => {
    expect(turnstileScripts()).toHaveLength(0);
    const first = turnstile.loadTurnstile();
    expect(turnstile.loadTurnstile()).toBe(first);
    expect(turnstileScripts()).toHaveLength(1);
    const url = new URL(turnstileScripts()[0].src);
    expect(url.origin + url.pathname).toBe('https://challenges.cloudflare.com/turnstile/v0/api.js');
    expect(url.searchParams.get('render')).toBe('explicit');
    expect(url.searchParams.get('onload')).toMatch(/^\w+$/);
    expect(turnstileScripts()[0].async).toBe(true);
    widget.load();
    await first;
    await turnstile.loadTurnstile();
    expect(turnstileScripts()).toHaveLength(1);
  });

  it('renders from the async script without turnstile.ready(), which the real api.js rejects', async () => {
    const controller = await renderWidget();
    expect(await controller.execute()).toBe('turnstile-token');
    expect(widget.api.ready).not.toHaveBeenCalled();
    expect(widget.api.render).toHaveBeenCalledOnce();
    controller.remove();
  });

  it('waits for the script, executes by widget id, and resets/removes by id', async () => {
    const controller = startWidget();
    await Promise.resolve();
    const execution = controller.execute();
    expect(widget.api.render).not.toHaveBeenCalled();
    await loadScript();
    expect(await execution).toBe('turnstile-token');
    expect(widget.api.execute).toHaveBeenCalledWith('widget-1');
    controller.reset();
    controller.remove();
    expect(widget.api.reset).toHaveBeenCalledWith('widget-1');
    expect(widget.api.remove).toHaveBeenCalledWith('widget-1');
  });

  it.each(['error-callback', 'expired-callback'] as const)('rejects %s and can execute after reset', async (callback) => {
    const controller = await renderWidget();
    widget.api.execute.mockImplementationOnce(() => widget.options()[callback]());
    await expect(controller.execute()).rejects.toThrow();
    controller.reset();
    expect(await controller.execute()).toBe('turnstile-token');
    controller.remove();
  });

  it('rejects execution after 20 seconds', async () => {
    vi.useFakeTimers();
    const controller = await renderWidget();
    widget.api.execute.mockImplementation(() => {});
    const rejected = expect(controller.execute()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(20_000);
    await rejected;
    controller.remove();
  });

  it('queues a reset before the script loads, then resets before executing', async () => {
    const controller = startWidget();
    controller.reset();
    const token = controller.execute();
    await loadScript();
    expect(await token).toBe('turnstile-token');
    expect(widget.api.reset).toHaveBeenCalledWith('widget-1');
    expect(widget.api.reset.mock.invocationCallOrder[0]).toBeLessThan(widget.api.execute.mock.invocationCallOrder[0]);
    controller.remove();
  });

  it('executes only the current attempt if the script loads after a timeout', async () => {
    vi.useFakeTimers();
    const controller = startWidget();
    const expired = expect(controller.execute()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(20_000);
    await expired;
    controller.reset();
    widget.api.execute.mockImplementation(() => {});
    const token = controller.execute();
    await loadScript();
    expect(widget.api.execute).toHaveBeenCalledOnce();
    widget.options().callback('turnstile-token');
    expect(await token).toBe('turnstile-token');
    controller.remove();
  });

  it('rejects script failure without adding another script', async () => {
    const loaded = turnstile.loadTurnstile();
    const rejected = expect(loaded).rejects.toThrow();
    turnstileScripts()[0].dispatchEvent(new Event('error'));
    await rejected;
    await expect(turnstile.loadTurnstile()).rejects.toThrow();
    expect(turnstileScripts()).toHaveLength(1);
  });

  it('cancels a pending execution on remove and ignores a late script load', async () => {
    const controller = startWidget();
    const rejected = expect(controller.execute()).rejects.toThrow();
    controller.remove();
    await loadScript();
    await rejected;
    expect(widget.api.render).not.toHaveBeenCalled();
  });
});
