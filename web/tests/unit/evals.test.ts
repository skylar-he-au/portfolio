import { describe, expect, it } from 'vitest';
import threePasses from '../fixtures/evals/qa-3-passes.json';
import onePass from '../fixtures/evals/qa-1-pass.json';
import older from '../fixtures/evals/qa-older.json';
import cot from '../fixtures/evals/cot.json';
import match from '../fixtures/evals/match.json';
import {
  buildMatchRows,
  buildRagasRows,
  CHECK_LABELS,
  formatChecks,
  formatCost,
  formatMetric,
  formatScore,
  MATCH_SCORE_LABELS,
  matchPassLines,
  METRIC_LABELS,
  parseEvalFiles,
} from '../../src/lib/evals';

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
});
