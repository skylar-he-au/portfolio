export function parseShotArgs(argv: string[]): {
  url: string;
  width: number;
  height: number;
  scheme: 'light' | 'dark';
  full: boolean;
  out: string;
};
