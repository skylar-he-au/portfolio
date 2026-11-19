import { describe, expect, it } from 'vitest';
import { parseShotArgs } from '../../scripts/shot.mjs';

describe('screenshot arguments', () => {
  it('uses desktop defaults', () => {
    expect(parseShotArgs(['http://localhost:4321', '--out', 'shot.png'])).toEqual({
      url: 'http://localhost:4321',
      width: 1280,
      height: 800,
      scheme: 'light',
      full: false,
      out: 'shot.png',
    });
  });

  it('accepts every option', () => {
    expect(parseShotArgs([
      'https://example.com/page', '--width', '375', '--height', '900',
      '--scheme', 'dark', '--full', '--out', 'screenshots/mobile.jpg',
    ])).toEqual({
      url: 'https://example.com/page',
      width: 375,
      height: 900,
      scheme: 'dark',
      full: true,
      out: 'screenshots/mobile.jpg',
    });
  });

  it.each(['png', 'jpg', 'jpeg'])('accepts .%s output', (extension) => {
    expect(parseShotArgs(['https://example.com', '--out', `shot.${extension}`]).out)
      .toBe(`shot.${extension}`);
  });

  it.each(['200', '4000'])('accepts the dimension boundary %s', (dimension) => {
    expect(parseShotArgs([
      'https://example.com', '--out', 'shot.png', '--width', dimension, '--height', dimension,
    ])).toMatchObject({ width: Number(dimension), height: Number(dimension) });
  });

  it('requires a URL', () => {
    expect(() => parseShotArgs(['--out', 'shot.png'])).toThrow(/url.*required/i);
  });

  it('requires an output path', () => {
    expect(() => parseShotArgs(['https://example.com'])).toThrow(/--out.*required/i);
  });

  it.each(['--width', '--height'])('validates %s', (option) => {
    for (const value of ['199', '4001', '375.5', 'wide', 'NaN', 'Infinity', '']) {
      expect(() => parseShotArgs(['https://example.com', '--out', 'shot.png', `${option}=${value}`]))
        .toThrow(/integer.*200.*4000/i);
    }
  });

  it('rejects an unknown color scheme', () => {
    expect(() => parseShotArgs(['https://example.com', '--out', 'shot.png', '--scheme', 'auto']))
      .toThrow(/scheme.*light.*dark/i);
  });

  it.each(['file:///etc/hosts', 'ftp://example.com', 'not a url'])('rejects URL %s', (url) => {
    expect(() => parseShotArgs([url, '--out', 'shot.png'])).toThrow(/https?/i);
  });

  it.each(['shot.webp', 'shot'])('rejects unsupported output %s', (out) => {
    expect(() => parseShotArgs(['https://example.com', '--out', out])).toThrow(/png.*jpg.*jpeg/i);
  });

  it('rejects extra positional arguments', () => {
    expect(() => parseShotArgs(['https://example.com', 'extra', '--out', 'shot.png']))
      .toThrow(/one url/i);
  });

  it('rejects unknown options and missing option values', () => {
    expect(() => parseShotArgs(['https://example.com', '--out', 'shot.png', '--mobile'])).toThrow();
    expect(() => parseShotArgs(['https://example.com', '--out'])).toThrow();
  });
});
