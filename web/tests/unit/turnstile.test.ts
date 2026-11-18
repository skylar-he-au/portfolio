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

async function renderWidget() {
  const controller = turnstile.renderWidget(document.createElement('div'), 'sitekey');
  turnstileScripts()[0].dispatchEvent(new Event('load'));
  await Promise.resolve();
  return controller;
}

describe('Turnstile controller', () => {
  it('loads the explicit script lazily and shares one promise and script', async () => {
    expect(turnstileScripts()).toHaveLength(0);
    const first = turnstile.loadTurnstile();
    expect(turnstile.loadTurnstile()).toBe(first);
    expect(turnstileScripts()).toHaveLength(1);
    expect(turnstileScripts()[0].src).toBe('https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit');
    turnstileScripts()[0].dispatchEvent(new Event('load'));
    await first;
    await turnstile.loadTurnstile();
    expect(turnstileScripts()).toHaveLength(1);
  });

  it('waits for ready, executes by widget id, and resets/removes by id', async () => {
    widget.api.ready.mockImplementation(() => {});
    const controller = await renderWidget();
    expect(widget.api.render).not.toHaveBeenCalled();
    const execution = controller.execute();
    widget.api.ready.mock.calls[0][0]();
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

  it('queues a reset before readiness, then resets before executing', async () => {
    widget.api.ready.mockImplementation(() => {});
    const controller = await renderWidget();
    controller.reset();
    const token = controller.execute();
    widget.api.ready.mock.calls[0][0]();
    expect(await token).toBe('turnstile-token');
    expect(widget.api.reset).toHaveBeenCalledWith('widget-1');
    expect(widget.api.reset.mock.invocationCallOrder[0]).toBeLessThan(widget.api.execute.mock.invocationCallOrder[0]);
    controller.remove();
  });

  it('executes only the current attempt if readiness arrives after a timeout', async () => {
    vi.useFakeTimers();
    widget.api.ready.mockImplementation(() => {});
    const controller = await renderWidget();
    const expired = expect(controller.execute()).rejects.toThrow();
    await vi.advanceTimersByTimeAsync(20_000);
    await expired;
    controller.reset();
    widget.api.execute.mockImplementation(() => {});
    const token = controller.execute();
    widget.api.ready.mock.calls[0][0]();
    await Promise.resolve();
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

  it('cancels a pending execution on remove and ignores late readiness', async () => {
    widget.api.ready.mockImplementation(() => {});
    const controller = await renderWidget();
    const rejected = expect(controller.execute()).rejects.toThrow();
    controller.remove();
    widget.api.ready.mock.calls[0][0]();
    await rejected;
    expect(widget.api.render).not.toHaveBeenCalled();
  });
});
