import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    include: process.env.TEST_DIST === '1'
      ? ['tests/dist/**/*.test.ts']
      : ['tests/unit/**/*.test.{ts,tsx}'],
  },
});
