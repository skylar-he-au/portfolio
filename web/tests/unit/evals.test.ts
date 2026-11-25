import { describe, expect, it } from 'vitest';
import threePasses from '../fixtures/evals/qa-3-passes.json';
import onePass from '../fixtures/evals/qa-1-pass.json';
import older from '../fixtures/evals/qa-older.json';
import cot from '../fixtures/evals/cot.json';
import match from '../fixtures/evals/match.json';
import compare from '../fixtures/evals/compare.json';
import step10 from '../fixtures/evals/qa-step10.json';
import {
  buildMatchRows,
  buildRagasRows,
  CHECK_LABELS,
  compareOptionLabel,
  formatCompareScore,
  formatMatchCost,
  formatSeconds,
  matchJudgeLabel,
  parseCompare,
  SUBSCRIPTION_GRADER,
  SUBSCRIPTION_GRADER_LABEL,
  formatChecks,
  formatCost,
  formatRunCost,
  formatMetric,
  formatQuoteMatch,
  formatScore,
  NOT_JUDGED_LABEL,
  qaGapsLabel,
  qaScoresCaption,
  MATCH_SCORE_LABELS,
  matchPassLines,
  METRIC_LABELS,
  newestFullQaRun,
  parseEvalFiles,
  pickHeadlineCards,
} from '../../src/lib/evals';

describe('headline cards', () => {
  const latestQa = {
    ...threePasses,
    run_id: '20261118T031419-v7-default',
    date: '2026-11-18',
    prompt_version: 'v7',
    ragas: { ...threePasses.ragas, held_out: { faithfulness: { mean: .974 } } },
    code_checks: { ...threePasses.code_checks, project: { passed: 59, total: 60 } },
  };
  // A later date must not outweigh the greater run_id.
  const lesserQa = { ...threePasses, run_id: '20261117T031419-v6-default', date: '2026-11-21' };
  const qaCards = [
    { key: 'qa-faithfulness', label: 'Faithfulness, held-out questions', value: '0.97', note: 'prompt v7 · 2026-11-18' },
    { key: 'qa-project', label: 'Project questions, code checks', value: '59/60', note: 'prompt v7 · 2026-11-18' },
    { key: 'qa-zh', label: 'Chinese questions, code checks', value: '18/18', note: 'prompt v7 · 2026-11-18' },
  ];

  it.each([
    [latestQa, lesserQa],
    [lesserQa, latestQa],
  ])('picks QA by greatest run_id in either input order and formats values and notes (%#)', (first, last) => {
    expect(pickHeadlineCards([first, last])).toEqual(qaCards);
  });

  it('ignores a greater QA run_id on another route', () => {
    expect(pickHeadlineCards([latestQa, {
      ...lesserQa, run_id: '20261119T031419-v7-gemini-answer', route: 'gemini-answer',
    }])).toEqual(qaCards);
  });

  it('ignores a greater QA run_id on another split', () => {
    expect(pickHeadlineCards([{
      ...lesserQa, run_id: '20261119T031419-v7-default', split: 'tune',
    }, latestQa])).toEqual(qaCards);
  });

  it('picks match by greatest run_id regardless of date, route, split or input order', () => {
    const latestMatch = {
      ...match, run_id: `${match.run_id}-z`, date: '2026-11-18',
      route: 'other', split: 'tune', code_checks: { all: { passed: 77, total: 78 } },
    };
    const expected = {
      key: 'match-checks', label: 'Job match, code checks', value: '77/78',
      note: 'prompt match-v9 · 2026-11-18',
    };
    for (const runs of [[latestMatch, match], [match, latestMatch]]) {
      expect(pickHeadlineCards(runs)).toEqual([expected]);
      expect(pickHeadlineCards([latestQa, ...runs])).toEqual([...qaCards, expected]);
    }
  });

  it('returns no cards for no qualifying runs', () => {
    expect(pickHeadlineCards([])).toEqual([]);
    expect(pickHeadlineCards([onePass, cot, { ...latestQa, route: 'other' }])).toEqual([]);
  });

  it('omits missing values without falling back to an older run', () => {
    expect(pickHeadlineCards([lesserQa, {
      ...latestQa, ragas: { held_out: { faithfulness: { mean: null } } },
      code_checks: { zh: { passed: 0, total: 0 } },
    }, { ...match, code_checks: {} }])).toEqual([
      { ...qaCards[2], value: '0/0' },
    ]);
    expect(pickHeadlineCards([{ ...latestQa, ragas: {}, code_checks: {} }])).toEqual([]);
  });
});

describe('eval summaries', () => {
  it('sorts several files by descending date, then descending run id for both kinds', () => {
    const laterQa = { ...threePasses, run_id: `${threePasses.run_id}-z` };
    const earlierCot = { ...cot, date: '2026-11-18' };
    const laterCot = { ...cot, run_id: `${cot.run_id}-z` };
    const result = parseEvalFiles([older, threePasses, earlierCot, cot, onePass, laterQa, laterCot]);
    expect(result.qa.map((run) => run.run_id))
      .toEqual([onePass.run_id, laterQa.run_id, threePasses.run_id, older.run_id]);
    expect(result.cot.map((run) => [run.date, run.run_id]))
      .toEqual([laterCot, cot, earlierCot].map((run) => [run.date, run.run_id]));
  });

  it('produces no range with one pass, even if different bounds are supplied', () => {
    const run = parseEvalFiles([onePass]).qa[0]!;
    expect(buildRagasRows(run)[0]?.cells[0]).toBe('0.91, 0 below 0.7');
    expect(formatMetric(threePasses.ragas.tune.faithfulness, { repeats: 1 })).toBe('0.91');
  });

  it('omits the held-out row when the run has no held-out split', () => {
    const run = parseEvalFiles([onePass]).qa[0]!;
    expect(buildRagasRows(run).map((row) => row.label))
      .toEqual(['Tuning questions', 'Chinese questions']);
  });

  it('returns empty lists for empty input', () => {
    expect(parseEvalFiles([])).toEqual({ qa: [], cot: [], match: [] });
  });

  it('preserves all three CoT arms in file order', () => {
    const run = parseEvalFiles([cot]).cot[0]!;
    expect(run.arms).toHaveLength(3);
    expect(run.arms.map((arm) => arm.label)).toEqual(cot.arms.map((arm) => arm.label));
  });

  it('drops unknown kinds and non-objects', () => {
    expect(parseEvalFiles([{ kind: 'unknown' }, null, 42, 'qa', [], true]))
      .toEqual({ qa: [], cot: [], match: [] });
  });

  it('formats null metric values as an en dash', () => {
    expect(formatMetric({ mean: null, min: null, max: null }, { repeats: 3 })).toBe('–');
    expect(formatScore(null)).toBe('–');
    expect(formatMetric(undefined, { repeats: 3 })).toBe('–');
  });

  it('formats costs at normal, sub-cent, and zero amounts', () => {
    expect(formatCost(1.23)).toBe('$1.23');
    expect(formatCost(0.004)).toBe('<$0.01');
    expect(formatCost(0)).toBe('$0.00');
    expect(formatCost(undefined)).toBe('–');
  });

  it('splits a run cost into paid and provider credit, or labels a lone total', () => {
    const pooled = parseEvalFiles([{ ...match, cost_usd: 3.458717,
      cost_pools_usd: { real: 0.222603, credit: 3.236114 } }]).match[0]!;
    expect(formatRunCost(pooled)).toBe('$0.22 paid + $3.24 provider credit');
    const qa = parseEvalFiles([{ ...onePass, cost_usd: 1.2, cost_pools_usd: { real: 0.004, credit: 0 } }]).qa[0]!;
    expect(formatRunCost(qa)).toBe('<$0.01 paid + $0.00 provider credit');
    const broken = parseEvalFiles([{ ...onePass, cost_usd: 1.2, cost_pools_usd: { real: -1, credit: 1 } }]).qa[0]!;
    expect(broken.cost_pools_usd).toBeUndefined();
    expect(formatRunCost(broken)).toBe('$1.20 total');
    expect(formatRunCost({})).toBe('–');
  });

  it('shows below-floor counts only for faithfulness, including zero but excluding null', () => {
    const run = parseEvalFiles([threePasses]).qa[0]!;
    run.ragas.tune.response_relevancy!.below_floor = 9;
    const cells = buildRagasRows(run)[0]!.cells;
    expect(cells[0]).toBe('0.91 (0.88–0.93), 2 below 0.7');
    expect(cells[1]).toBe('0.80 (0.78–0.82)');
    expect(formatMetric({ mean: .9, below_floor: 0 }, { repeats: 3, floor: .7 }))
      .toBe('0.90, 0 below 0.7');
    expect(formatMetric({ mean: .9, below_floor: null }, { repeats: 3, floor: .7 }))
      .toBe('0.90');
  });

  it('hides identical ranges and formats scores and checks', () => {
    expect(formatMetric({ mean: .9, min: .9, max: .9 }, { repeats: 3 })).toBe('0.90');
    expect(formatScore(.8)).toBe('0.80');
    expect(formatChecks({ passed: 58, total: 60 })).toBe('58/60');
    expect(formatChecks(undefined)).toBe('–');
  });

  it('keeps display names in metric order and labels all known check categories', () => {
    expect(Object.values(METRIC_LABELS)).toEqual([
      'Faithfulness', 'Response relevancy', 'Context recall', 'Factual correctness',
    ]);
    expect(CHECK_LABELS).toEqual({
      project: 'Project', zh: 'Chinese', not_on_site: 'Not on site',
      private_answerable: 'Private (answerable)', private_fixed: 'Private (fixed reply)',
      off_topic: 'Off topic', injection: 'Injection',
    });
  });

  it('treats malformed optional fields as absent and discards fields outside the contract', () => {
    const result = parseEvalFiles([
      { ...threePasses, models: [], private_repeat: { runs: 10 }, code_checks: {
        project: { passed: '58', total: 60 }, injection: null, secret: { passed: 1, total: 1 },
      }, ragas: { tune: { faithfulness: { mean: 'oops', min: {}, max: null } }, held_out: [], zh: null },
      question: 'must be discarded', errors: {}, cost_usd: '1.23' },
      { kind: 'qa' },
      { ...cot, arms: [null, 'bad', { ...cot.arms[0], accuracy: [], invalid: {} }, {}] },
      { kind: 'cot', arms: {} },
    ]);
    const run = result.qa[0]!;
    expect(run.models).toEqual({});
    expect(run.private_repeat).toBeUndefined();
    expect(run.code_checks).toEqual({});
    expect(run.errors).toBeUndefined();
    expect(run.cost_usd).toBeUndefined();
    expect(run).not.toHaveProperty('question');
    expect(buildRagasRows(run)).toEqual([{ label: 'Tuning questions', cells: ['–', '–', '–', '–'] }]);
    expect(buildRagasRows(result.qa[1]!)).toHaveLength(1);
    expect(result.cot[0]!.arms).toHaveLength(1);
    expect(result.cot[0]!.arms[0]!.accuracy).toBeUndefined();
    expect(result.cot[1]!.arms).toEqual([]);
  });
});

describe('Q&A summaries from step 10', () => {
  it('shows the two metrics the grader no longer judges as "Not judged", not as scores or gaps', () => {
    const run = parseEvalFiles([step10]).qa[0]!;
    expect(run.not_judged).toEqual(['response_relevancy', 'factual_correctness']);
    expect(buildRagasRows(run)).toEqual([
      { label: 'Tuning questions', cells: ['0.99 (0.99–1.00), 0 below 0.7', NOT_JUDGED_LABEL, '0.99 (0.98–1.00)', NOT_JUDGED_LABEL] },
      { label: 'Held-out questions', cells: ['1.00, 0 below 0.7', NOT_JUDGED_LABEL, '1.00', NOT_JUDGED_LABEL] },
    ]);
    expect(qaScoresCaption(run)).toBe(`Scores graded by ${SUBSCRIPTION_GRADER_LABEL}: mean and, for repeated passes, lowest–highest pass average. Response relevancy and Factual correctness not judged.`);
    expect(qaGapsLabel(run)).toBe('Grading gaps');
  });

  it('keeps the RAGAS caption and four scored columns for older runs', () => {
    const run = parseEvalFiles([threePasses]).qa[0]!;
    expect(run.not_judged).toBeUndefined();
    expect(buildRagasRows(run)[0]!.cells).not.toContain(NOT_JUDGED_LABEL);
    expect(qaScoresCaption(run)).toBe('RAGAS scores: mean and, for repeated passes, lowest–highest pass average.');
    expect(qaGapsLabel(run)).toBe('RAGAS gaps');
  });

  it('formats the share of quoted sentences found, with the lowest–highest pass', () => {
    const run = parseEvalFiles([step10]).qa[0]!;
    expect(formatQuoteMatch(run)).toBe('0.86 (335 of 389 quoted sentences found; 132 of 135 sources with at least one), passes 0.80–0.95');
    const onePassRun = parseEvalFiles([{ ...step10, quote_match: { ...step10.quote_match, passes: { 1: 0.86 } } }]).qa[0]!;
    expect(formatQuoteMatch(onePassRun)).toBe('0.86 (335 of 389 quoted sentences found; 132 of 135 sources with at least one)');
  });

  it('shows no quote line for older summaries, runs without quoted sentences, or malformed ones', () => {
    for (const raw of [
      threePasses,
      { ...step10, quote_match: { samples: 0, given: 0, kept: 0, citations: 0, citations_quoted: 0, rate: null, passes: {} } },
      { ...step10, quote_match: { samples: 3, given: 2, kept: 5, rate: 2.5, passes: {} } },
      { ...step10, quote_match: 'oops' },
    ]) {
      expect(formatQuoteMatch(parseEvalFiles([raw]).qa[0]!)).toBeUndefined();
    }
  });
});

describe('job match summaries', () => {
  it('parses a match run with both splits and builds one row per split', () => {
    const run = parseEvalFiles([match]).match[0]!;
    expect(run.prompt_version).toBe('match-v9');
    expect(run.code_checks).toEqual({ passed: 63, total: 63 });
    expect(run.models.judge_faithfulness).toBe('claude-sonnet-5-5');
    expect(buildMatchRows(run)).toEqual([
      { label: 'Tuning ads', cells: ['0.74 (0.73–0.75)', '0.85 (0.83–0.87)', '0.89 (0.87–0.91), 4 below 0.7', '0.51 (0.51–0.52)'] },
      { label: 'Held-out ads', cells: ['0.85 (0.80–0.90)', '0.95', '0.93 (0.91–0.95), 0 below 0.7', '0.50'] },
    ]);
    expect(Object.values(MATCH_SCORE_LABELS))
      .toEqual(['Section recall', 'Not-covered accuracy', 'Faithfulness', 'Precision']);
    expect(matchPassLines()).toBe(
      'Section recall ≥ 0.80, Not-covered accuracy ≥ 0.90, Faithfulness ≥ 0.90, every ad in every pass Faithfulness ≥ 0.70',
    );
  });

  it('keeps only contract fields, so no ad, requirement or guide text reaches the page', () => {
    const run = parseEvalFiles([{
      ...match,
      jd: 'Senior engineer wanted', requirements: [{ text: 'Python' }], items: [{ text: 'guide' }],
      scores: { tune: { recall: { mean: 0.8 }, guide: 'x' }, held_out: null },
      code_checks: { all: { passed: '1', total: 2 } },
    }]).match[0]!;
    const json = JSON.stringify(run);
    for (const leak of ['Senior engineer', 'Python', 'guide']) expect(json).not.toContain(leak);
    expect(run.code_checks).toBeUndefined();
    expect(buildMatchRows(run).map((row) => row.label)).toEqual(['Tuning ads']);
  });

  it('sorts match runs newest first', () => {
    const older = { ...match, date: '2026-11-19' };
    expect(parseEvalFiles([older, match]).match.map((run) => run.date)).toEqual(['2026-11-20', '2026-11-19']);
  });

  it('names the subscription grader, not RAGAS, and keeps API judges by model id', () => {
    const graded = parseEvalFiles([{
      ...match,
      models: { ...match.models, judge_align: SUBSCRIPTION_GRADER, judge_faithfulness: SUBSCRIPTION_GRADER },
      grader: SUBSCRIPTION_GRADER,
      grading: { status: 'incomplete', kinds: {}, warnings: 0 },
    }]).match[0]!;
    expect(matchJudgeLabel(graded)).toBe(SUBSCRIPTION_GRADER_LABEL);
    expect(SUBSCRIPTION_GRADER_LABEL).toBe('Claude Sonnet 5.5 (subscription, following GRADING.md), not RAGAS');
    expect(graded.grading_status).toBe('incomplete');
    const api = parseEvalFiles([match]).match[0]!;
    expect(matchJudgeLabel(api)).toBe('claude-sonnet-5-5');
    expect(api.grading_status).toBeUndefined();
  });
});

describe('model comparison table', () => {
  it('orders each step by its ranking, keeps the v4 row and labels the options', () => {
    const table = parseCompare(compare)!;
    expect(table.steps.map((step) => step.step)).toEqual(['read', 'check']);
    const [read, check] = table.steps;
    expect(read!.options.map((o) => o.run_id)).toEqual(['r-max12', 'r-haiku']);
    expect(read!.options.map((o) => compareOptionLabel(o, 'read'))).toEqual([
      'claude-haiku-5-5, thinking off, up to 12 requirements',
      'claude-haiku-5-5, thinking off, up to 8 requirements',
    ]);
    expect(formatCompareScore(read!.options[0]!.score.tune, 'read')).toBe('0.94 (149/158)');
    expect(formatCompareScore(read!.old.tune, 'read')).toBe('0.86 (136/158)');
    expect(formatCompareScore(read!.old.held_out, 'read')).toBe('0.85 (34/40)');
    expect(check!.options.map((o) => o.run_id)).toEqual(['c-minimal', 'c-luna']);
    expect(compareOptionLabel(check!.options[1]!, 'check')).toBe('gpt-6-luna, reasoning none');
    expect(compareOptionLabel({ order: 'anthropic:claude-sonnet-5-5:off (max 1024)' }, 'write'))
      .toBe('claude-sonnet-5-5, lowest thinking setting');
    expect(compareOptionLabel({ order: 'anthropic:claude-haiku-5-5:off (max 1024)' }, 'write'))
      .toBe('claude-haiku-5-5, thinking off');
    expect(formatCompareScore(check!.options[1]!.score.tune, 'check'))
      .toBe('0.81 (240/295), 46 removed and 9 kept wrongly');
    expect(formatCompareScore(check!.options[0]!.score.tune, 'check')).toBe('–');
    expect(check!.options[1]!.over_cap).toBe(true);
  });

  it('keeps names and numbers only and formats cost and latency', () => {
    const json = JSON.stringify(parseCompare(compare));
    expect(json).not.toContain('TypeScript');
    expect(parseCompare({ steps: {} })).toBeUndefined();
    expect(formatMatchCost(0.00823)).toBe('$0.0082');
    expect(formatMatchCost(undefined)).toBe('–');
    expect(formatSeconds(9862)).toBe('9.9 s');
  });
});

describe('newest full final run', () => {
  const full = (run_id: string, extra: Record<string, unknown> = {}) =>
    ({ ...threePasses, run_id, route: 'default', split: 'all', repeats: 3, ...extra });

  it('picks the greatest run_id among default-route, all-question runs with three or more passes', () => {
    const { qa } = parseEvalFiles([
      full('20261118T031419-v7-default', { prompt_version: 'v7' }),
      full('20261125T010000-v11-default', { prompt_version: 'v11' }),
      full('20261126T010000-v12-gemini-answer', { route: 'gemini-answer' }),
      full('20261126T020000-v12-default', { repeats: 1 }),
      full('20261126T030000-v12-default', { split: 'tune' }),
    ]);
    expect(newestFullQaRun(qa)?.prompt_version).toBe('v11');
  });

  it('returns nothing when no run is a full final run', () => {
    const { qa } = parseEvalFiles([full('20261126T020000-v12-default', { repeats: 1 })]);
    expect(newestFullQaRun(qa)).toBeUndefined();
    expect(newestFullQaRun([])).toBeUndefined();
  });
});
