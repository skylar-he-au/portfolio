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

// What a run cost from each spending pool: real money, and the provider's granted credit.
export interface CostPools {
  real: number;
  credit: number;
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
  cost_pools_usd?: CostPools;
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
  cost_pools_usd?: CostPools;
  // Round 2 on: who graded (the subscription grader) and whether every needed grade came back.
  grader?: string;
  grading_status?: 'complete' | 'incomplete';
}

// The subscription grader of step 6 round 2 (docs/evals/GRADING.md in the docs repo).
export const SUBSCRIPTION_GRADER = 'claude-sonnet-5-5 (subscription, GRADING.md)';
export const SUBSCRIPTION_GRADER_LABEL = 'Claude Sonnet 5.5 (subscription, following GRADING.md), not RAGAS';

// "Judge: …" for a match run: the subscription grader by its label, API judges by model id.
export function matchJudgeLabel(run: MatchRun): string {
  const judges = [...new Set([run.models.judge_align, run.models.judge_faithfulness]
    .filter((m): m is string => !!m))];
  if (run.grader === SUBSCRIPTION_GRADER || (judges.length === 1 && judges[0] === SUBSCRIPTION_GRADER)) {
    return SUBSCRIPTION_GRADER_LABEL;
  }
  return judges.join(', ') || '–';
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

function parseCostPools(value: unknown): CostPools | undefined {
  const raw = object(value);
  const real = number(raw?.real);
  const credit = number(raw?.credit);
  return real !== undefined && credit !== undefined ? { real, credit } : undefined;
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
    cost_pools_usd: parseCostPools(raw.cost_pools_usd),
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
    cost_pools_usd: parseCostPools(raw.cost_pools_usd),
    grader: text(raw.grader),
    grading_status: gradingStatus(object(raw.grading)?.status),
  };
}

function gradingStatus(value: unknown): 'complete' | 'incomplete' | undefined {
  return value === 'complete' || value === 'incomplete' ? value : undefined;
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

// "$0.22 paid + $3.24 provider credit" when the summary splits the pools; older summaries
// only have the total, and say so.
export function formatRunCost(run: { cost_usd?: number; cost_pools_usd?: CostPools }): string {
  const pools = parseCostPools(run.cost_pools_usd);
  if (pools) return `${formatCost(pools.real)} paid + ${formatCost(pools.credit)} provider credit`;
  const total = formatCost(run.cost_usd);
  return total === '–' ? total : `${total} total`;
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

// --- step 6 round 2: one model per step, compared on replays of a stored run ----------------

export const COMPARE_STEPS = ['read', 'match', 'check', 'write'] as const;
export type CompareStep = typeof COMPARE_STEPS[number];

export const COMPARE_STEP_LABELS: Record<CompareStep, { step: string; metric: string }> = {
  read: { step: 'Read the ad', metric: 'Gold requirements read' },
  match: { step: 'Match each requirement', metric: 'Expected section picked (of those retrieved)' },
  check: { step: 'Check each link', metric: 'Pairs judged right' },
  write: { step: 'Write the guide', metric: 'Faithfulness' },
};

const COMPARE_METRICS: Record<CompareStep, string> = {
  read: 'read_share', match: 'match_share', check: 'pairs_right', write: 'faithfulness',
};

export interface CompareScore {
  value?: number;
  num?: number;
  den?: number;
  n?: number;
  wrongly_removed?: number;
  wrongly_kept?: number;
  status?: 'ok' | 'awaiting_grades' | 'incomplete';
}

export interface SplitScores {
  tune?: CompareScore;
  held_out?: CompareScore;
}

export interface CompareOption {
  run_id: string;
  order: string;
  max_requirements?: number;
  rows?: number;
  errors?: number;
  score: SplitScores;
  cost_per_match?: number;
  latency_per_match_ms?: number;
  over_cap: boolean;
}

export interface CompareStepTable {
  step: CompareStep;
  old: SplitScores;
  options: CompareOption[]; // ranking order: the first is the one chosen
}

export interface CompareTable {
  source_run: string;
  items_tune?: number;
  items_held_out?: number;
  cap_per_match_usd?: number;
  tie?: number;
  steps: CompareStepTable[];
}

function parseCompareScore(value: unknown): CompareScore | undefined {
  const raw = object(value);
  if (!raw) return undefined;
  const status = raw.status === 'ok' || raw.status === 'awaiting_grades' || raw.status === 'incomplete'
    ? raw.status : undefined;
  return {
    value: number(raw.value),
    num: number(raw.num),
    den: count(raw.den),
    n: count(raw.n),
    wrongly_removed: count(raw.wrongly_removed),
    wrongly_kept: count(raw.wrongly_kept),
    status,
  };
}

function splitScores(scores: unknown, metric: string): SplitScores {
  const raw = object(scores);
  return {
    tune: parseCompareScore(object(raw?.tune)?.[metric]),
    held_out: parseCompareScore(object(raw?.held_out)?.[metric]),
  };
}

function parseOption(value: unknown, metric: string): CompareOption[] {
  const option = object(value);
  const run_id = text(option?.run_id);
  const order = text(option?.order);
  if (!option || !run_id || !order) return [];
  const cost = object(option.cost_usd);
  return [{
    run_id,
    order,
    max_requirements: count(option.max_requirements),
    rows: count(option.rows),
    errors: count(option.errors),
    score: splitScores(option.scores, metric),
    cost_per_match: number(cost?.per_match),
    latency_per_match_ms: number(object(option.latency_ms)?.per_match),
    over_cap: cost?.over_cap === true,
  }];
}

// Only the fields below are kept: names and numbers (the file has no ad, requirement or guide).
export function parseCompare(value: unknown): CompareTable | undefined {
  const raw = object(value);
  const steps = object(raw?.steps);
  const source = text(raw?.source_run);
  if (!raw || !steps || !source) return undefined;
  const old = object(object(raw.old_prompt)?.scores);
  const tables: CompareStepTable[] = [];
  for (const step of COMPARE_STEPS) {
    const entry = object(steps[step]);
    if (!entry) continue;
    const metric = COMPARE_METRICS[step];
    const options = (Array.isArray(entry.options) ? entry.options : [])
      .flatMap((item) => parseOption(item, metric));
    const ranking = (Array.isArray(entry.ranking) ? entry.ranking : [])
      .filter((id): id is string => typeof id === 'string');
    const rank = (option: CompareOption) => {
      const at = ranking.indexOf(option.run_id);
      return at === -1 ? ranking.length : at;
    };
    tables.push({ step, old: splitScores(old, metric), options: [...options].sort((a, b) => rank(a) - rank(b)) });
  }
  const items = object(raw.items);
  return {
    source_run: source,
    items_tune: count(items?.tune),
    items_held_out: count(items?.held_out),
    cap_per_match_usd: number(raw.cap_per_match_usd),
    tie: number(raw.tie),
    steps: tables,
  };
}

const EFFORT_LABELS: Record<string, string> = {
  off: 'thinking off',
  low: 'low thinking',
  minimal: 'minimal thinking',
  none: 'reasoning none',
};

// "claude-haiku-5-5, thinking off, up to 12 requirements": the option's first model.
export function compareOptionLabel(
  option: Pick<CompareOption, 'order' | 'max_requirements'>,
  step: CompareStep,
): string {
  const first = (option.order.split(' -> ')[0] ?? '').replace(/ \(max \d+\).*$/, '');
  const [, model, effort] = first.split(':');
  const parts = [model ?? first, effort ? EFFORT_LABELS[effort] ?? effort : undefined]
    .filter((part): part is string => !!part);
  if (step === 'read' && option.max_requirements !== undefined) {
    parts.push(`up to ${option.max_requirements} requirements`);
  }
  return parts.join(', ');
}

// "0.83 (246/295)"; the check adds what it removed or kept wrongly.
export function formatCompareScore(score: CompareScore | undefined, step: CompareStep): string {
  if (!score || score.status === 'awaiting_grades') return '–';
  let result = formatScore(score.value);
  if (score.num !== undefined && score.den !== undefined) result += ` (${Math.round(score.num)}/${score.den})`;
  if (step === 'check' && score.wrongly_removed !== undefined && score.wrongly_kept !== undefined) {
    result += `, ${score.wrongly_removed} removed and ${score.wrongly_kept} kept wrongly`;
  }
  if (score.status === 'incomplete') result += ', grading incomplete';
  return result;
}

export function formatMatchCost(usd: number | undefined): string {
  const value = number(usd);
  return value === undefined ? '–' : `$${value.toFixed(4)}`;
}

export function formatSeconds(ms: number | undefined): string {
  const value = number(ms);
  return value === undefined ? '–' : `${(value / 1000).toFixed(1)} s`;
}
