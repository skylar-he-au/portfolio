export const METRIC_LABELS = {
  faithfulness: 'Faithfulness',
  response_relevancy: 'Response relevancy',
  context_recall: 'Context recall',
  factual_correctness: 'Factual correctness',
} as const;

export const CHECK_LABELS = {
  project: 'Project',
  zh: 'Chinese',
  not_on_site: 'Not on site',
  private_answerable: 'Private (answerable)',
  private_fixed: 'Private (fixed reply)',
  off_topic: 'Off topic',
  injection: 'Injection',
} as const;

// Job-match scores, in table order. Pass lines are SPEC "评测先于上线" (job-match items);
// precision is recorded only.
export const MATCH_SCORE_LABELS = {
  recall: 'Section recall',
  not_covered_accuracy: 'Not-covered accuracy',
  faithfulness: 'Faithfulness',
  precision: 'Precision',
} as const;

export const MATCH_PASS_LINES: Partial<Record<keyof typeof MATCH_SCORE_LABELS, number>> = {
  recall: .8,
  not_covered_accuracy: .9,
  faithfulness: .9,
};

export const MATCH_FAITH_FLOOR = .7;

export type MetricName = keyof typeof METRIC_LABELS;
export type MatchScoreName = keyof typeof MATCH_SCORE_LABELS;
export type CheckName = keyof typeof CHECK_LABELS;

export interface Metric {
  mean?: number | null;
  min?: number | null;
  max?: number | null;
  scored?: number;
  below_floor?: number | null;
}

export interface Checks {
  passed: number;
  total: number;
}

export interface PrivateRepeat {
  runs: number;
  fixed: number;
  answered: number;
  failed: number;
}

type Metrics = Partial<Record<MetricName, Metric>>;

interface Run {
  run_id: string;
  date: string;
  repeats?: number;
}

export interface QaRun extends Run {
  kind: 'qa';
  prompt_version?: string;
  route?: string;
  split?: string;
  questions?: number;
  samples?: number;
  errors?: number;
  ragas_nulls?: number;
  models: Partial<Record<'answer' | 'check' | 'judge' | 'embed', string>>;
  ragas: { tune: Metrics; held_out?: Metrics; zh?: Metrics };
  code_checks: Partial<Record<CheckName, Checks>>;
  private_repeat?: PrivateRepeat;
  cost_usd?: number;
}

export interface CotArm {
  name: string;
  label: string;
  thinking_level?: string;
  accuracy?: Metric;
  unsupported_precision?: Metric;
  unsupported_recall?: Metric;
  invalid?: number;
  ms_median?: number;
  cost_usd?: number;
}

export interface CotRun extends Run {
  kind: 'cot';
  model?: string;
  sentences?: number;
  arms: CotArm[];
}

export interface MatchRun extends Run {
  kind: 'match';
  prompt_version?: string;
  route?: string;
  split?: string;
  questions?: number;
  samples?: number;
  errors?: number;
  fallback_samples?: number;
  models: Partial<Record<'answer' | 'check' | 'embed' | 'judge_align' | 'judge_faithfulness', string>>;
  scores: { tune?: Partial<Record<MatchScoreName, Metric>>; held_out?: Partial<Record<MatchScoreName, Metric>> };
  code_checks?: Checks;
  cost_usd?: number;
}

function object(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown> : undefined;
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function count(value: unknown): number | undefined {
  const result = number(value);
  return result !== undefined && Number.isInteger(result) ? result : undefined;
}

function nullableNumber(value: unknown): number | null | undefined {
  return value === null ? null : number(value);
}

function parseMetric(value: unknown): Metric | undefined {
  const raw = object(value);
  if (!raw) return undefined;
  return {
    mean: nullableNumber(raw.mean),
    min: nullableNumber(raw.min),
    max: nullableNumber(raw.max),
    scored: count(raw.scored),
    below_floor: raw.below_floor === null ? null : count(raw.below_floor),
  };
}

function parseMetrics(value: unknown): Metrics | undefined {
  const raw = object(value);
  if (!raw) return undefined;
  const result: Metrics = {};
  for (const key of Object.keys(METRIC_LABELS) as MetricName[]) {
    const metric = parseMetric(raw[key]);
    if (metric) result[key] = metric;
  }
  return result;
}

function parseChecks(value: unknown): Checks | undefined {
  const raw = object(value);
  const passed = count(raw?.passed);
  const total = count(raw?.total);
  return passed !== undefined && total !== undefined ? { passed, total } : undefined;
}

function parsePrivateRepeat(value: unknown): PrivateRepeat | undefined {
  const raw = object(value);
  const runs = count(raw?.runs);
  const fixed = count(raw?.fixed);
  const answered = count(raw?.answered);
  const failed = count(raw?.failed);
  return runs !== undefined && fixed !== undefined && answered !== undefined && failed !== undefined
    ? { runs, fixed, answered, failed } : undefined;
}

function parseQa(raw: Record<string, unknown>): QaRun {
  const models: QaRun['models'] = {};
  const rawModels = object(raw.models);
  for (const key of ['answer', 'check', 'judge', 'embed'] as const) {
    const model = text(rawModels?.[key]);
    if (model) models[key] = model;
  }
  const code_checks: QaRun['code_checks'] = {};
  const rawChecks = object(raw.code_checks);
  for (const key of Object.keys(CHECK_LABELS) as CheckName[]) {
    const checks = parseChecks(rawChecks?.[key]);
    if (checks) code_checks[key] = checks;
  }
  const ragas = object(raw.ragas);
  return {
    kind: 'qa',
    run_id: text(raw.run_id) ?? '',
    date: text(raw.date) ?? '',
    prompt_version: text(raw.prompt_version),
    route: text(raw.route),
    split: text(raw.split),
    repeats: count(raw.repeats),
    questions: count(raw.questions),
    samples: count(raw.samples),
    errors: count(raw.errors),
    ragas_nulls: count(raw.ragas_nulls),
    models,
    ragas: {
      tune: parseMetrics(ragas?.tune) ?? {},
      held_out: parseMetrics(ragas?.held_out),
      zh: parseMetrics(ragas?.zh),
    },
    code_checks,
    private_repeat: parsePrivateRepeat(raw.private_repeat),
    cost_usd: number(raw.cost_usd),
  };
}

function parseArm(value: unknown): CotArm | undefined {
  const raw = object(value);
  const name = text(raw?.name);
  const label = text(raw?.label);
  if (!raw || !name || !label) return undefined;
  return {
    name,
    label,
    thinking_level: text(raw.thinking_level),
    accuracy: parseMetric(raw.accuracy),
    unsupported_precision: parseMetric(raw.unsupported_precision),
    unsupported_recall: parseMetric(raw.unsupported_recall),
    invalid: count(raw.invalid),
    ms_median: number(raw.ms_median),
    cost_usd: number(raw.cost_usd),
  };
}

function parseMatchScores(value: unknown): Partial<Record<MatchScoreName, Metric>> | undefined {
  const raw = object(value);
  if (!raw) return undefined;
  const result: Partial<Record<MatchScoreName, Metric>> = {};
  for (const key of Object.keys(MATCH_SCORE_LABELS) as MatchScoreName[]) {
    const metric = parseMetric(raw[key]);
    if (metric) result[key] = metric;
  }
  return result;
}

// Only the fields below are kept: a summary file never carries a job ad, a requirement or
// a guide, and anything else in the file is dropped here too.
function parseMatch(raw: Record<string, unknown>): MatchRun {
  const models: MatchRun['models'] = {};
  const rawModels = object(raw.models);
  for (const key of ['answer', 'check', 'embed', 'judge_align', 'judge_faithfulness'] as const) {
    const model = text(rawModels?.[key]);
    if (model) models[key] = model;
  }
  const scores = object(raw.scores);
  return {
    kind: 'match',
    run_id: text(raw.run_id) ?? '',
    date: text(raw.date) ?? '',
    prompt_version: text(raw.prompt_version),
    route: text(raw.route),
    split: text(raw.split),
    repeats: count(raw.repeats),
    questions: count(raw.questions),
    samples: count(raw.samples),
    errors: count(raw.errors),
    fallback_samples: count(raw.fallback_samples),
    models,
    scores: { tune: parseMatchScores(scores?.tune), held_out: parseMatchScores(scores?.held_out) },
    code_checks: parseChecks(object(raw.code_checks)?.all),
    cost_usd: number(raw.cost_usd),
  };
}

function newestFirst(a: Run, b: Run): number {
  const date = a.date === b.date ? 0 : a.date > b.date ? -1 : 1;
  return date || (a.run_id === b.run_id ? 0 : a.run_id > b.run_id ? -1 : 1);
}

export function parseEvalFiles(raw: unknown[]): { qa: QaRun[]; cot: CotRun[]; match: MatchRun[] } {
  const qa: QaRun[] = [];
  const cot: CotRun[] = [];
  const match: MatchRun[] = [];
  for (const value of raw) {
    const run = object(value);
    if (run?.kind === 'qa') qa.push(parseQa(run));
    if (run?.kind === 'match') match.push(parseMatch(run));
    if (run?.kind === 'cot') {
      cot.push({
        kind: 'cot',
        run_id: text(run.run_id) ?? '',
        date: text(run.date) ?? '',
        repeats: count(run.repeats),
        model: text(run.model),
        sentences: count(run.sentences),
        arms: Array.isArray(run.arms)
          ? run.arms.map(parseArm).filter((arm): arm is CotArm => arm !== undefined) : [],
      });
    }
  }
  return { qa: qa.sort(newestFirst), cot: cot.sort(newestFirst), match: match.sort(newestFirst) };
}

export function formatScore(value: number | null | undefined): string {
  return number(value)?.toFixed(2) ?? '–';
}

export function formatMetric(
  metric: Metric | undefined,
  { repeats, floor }: { repeats?: number; floor?: number },
): string {
  const mean = number(metric?.mean);
  let result = formatScore(mean);
  const min = number(metric?.min);
  const max = number(metric?.max);
  if (mean !== undefined && repeats !== undefined && repeats > 1
    && min !== undefined && max !== undefined && !(min === mean && max === mean)) {
    result += ` (${formatScore(min)}–${formatScore(max)})`;
  }
  const below = count(metric?.below_floor);
  if (floor !== undefined && below !== undefined) result += `, ${below} below ${floor}`;
  return result;
}

export function formatCost(usd: number | undefined): string {
  const value = number(usd);
  if (value === undefined) return '–';
  return value > 0 && value < .01 ? '<$0.01' : `$${value.toFixed(2)}`;
}

export function formatChecks(value: Checks | undefined): string {
  const checks = parseChecks(value);
  return checks ? `${checks.passed}/${checks.total}` : '–';
}

export function buildRagasRows(run: QaRun): { label: string; cells: string[] }[] {
  const splits = [
    ['Tuning questions', run.ragas.tune],
    ['Held-out questions', run.ragas.held_out],
    ['Chinese questions', run.ragas.zh],
  ] as const;
  return splits.flatMap(([label, metrics]) => metrics ? [{
    label,
    cells: (Object.keys(METRIC_LABELS) as MetricName[]).map((key) => formatMetric(metrics[key], {
      repeats: run.repeats,
      floor: key === 'faithfulness' ? .7 : undefined,
    })),
  }] : []);
}

export function buildMatchRows(run: MatchRun): { label: string; cells: string[] }[] {
  const splits = [
    ['Tuning ads', run.scores.tune],
    ['Held-out ads', run.scores.held_out],
  ] as const;
  return splits.flatMap(([label, scores]) => scores ? [{
    label,
    cells: (Object.keys(MATCH_SCORE_LABELS) as MatchScoreName[]).map((key) => formatMetric(scores[key], {
      repeats: run.repeats,
      floor: key === 'faithfulness' ? MATCH_FAITH_FLOOR : undefined,
    })),
  }] : []);
}

// "Section recall ≥ 0.80" etc., from MATCH_PASS_LINES, for the caption.
export function matchPassLines(): string {
  const lines = (Object.keys(MATCH_SCORE_LABELS) as MatchScoreName[]).flatMap((key) => {
    const line = MATCH_PASS_LINES[key];
    return line === undefined ? [] : [`${MATCH_SCORE_LABELS[key]} ≥ ${formatScore(line)}`];
  });
  return `${lines.join(', ')}, every ad in every pass Faithfulness ≥ ${formatScore(MATCH_FAITH_FLOOR)}`;
}
