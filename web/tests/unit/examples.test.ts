import { describe, expect, it } from 'vitest';
import { parseExamples } from '../../src/lib/examples';

const body = `The 3 example questions shown under the Ask box are 1, 2 and 4 (chosen 2026-11-17); 3 and 5 are spares. The topic in brackets is the page section that should answer it; section names match the project pages as of 2026-11-17.

1. How does RoleRadar decide which jobs match a user? (RoleRadar: How RoleRadar matches jobs)
2. How do you stop the AI from making things up when it rewrites a resume? (RoleRadar: How RoleRadar tailors a resume without inventing experience)
3. How do you test RoleRadar's AI features? (RoleRadar: How RoleRadar tests its AI: evals and prompt versions)
4. What did you build in the team project, and what was someone else's work? (OnCallOps: The team and my role)
5. What happens in your dashboard when one of the data sources fails? (OnCallOps: When part of a page fails to load)`;

describe('parseExamples', () => {
  it('selects items 1, 2 and 4 from the examples body and removes their trailing topics', () => {
    expect(parseExamples(body)).toEqual([
      'How does RoleRadar decide which jobs match a user?',
      'How do you stop the AI from making things up when it rewrites a resume?',
      "What did you build in the team project, and what was someone else's work?",
    ]);
  });

  it('rejects a list with only three numbered items', () => {
    expect(() => parseExamples('1. First (Page: Section)\n2. Second\n3. Third'))
      .toThrow(/at least 4 numbered items/i);
  });

  it.each([1, 2, 4])('rejects four numbered items when item %i is missing', (missing) => {
    const markdown = [1, 2, 3, 4, 5]
      .filter((number) => number !== missing)
      .map((number) => `${number}. Question ${number} (Page: Section)`)
      .join('\n');
    expect(() => parseExamples(markdown)).toThrow(/items 1, 2 and 4/i);
  });

  it('picks the RoleRadar page examples 1, 2 and 3 by the page name in brackets', () => {
    expect(parseExamples(body, 'roleradar')).toEqual([
      'How does RoleRadar decide which jobs match a user?',
      'How do you stop the AI from making things up when it rewrites a resume?',
      "How do you test RoleRadar's AI features?",
    ]);
  });

  it('picks the OnCallOps page examples 4 and 5 by the page name in brackets', () => {
    expect(parseExamples(body, 'oncallops')).toEqual([
      "What did you build in the team project, and what was someone else's work?",
      'What happens in your dashboard when one of the data sources fails?',
    ]);
  });

  it('matches page names case-insensitively, in numeric order, and ignores colons in the section', () => {
    const markdown = '5. Fifth? (roleradar: A: B)\n4. Fourth? (Other: X)\n2. Second? (RoleRadar)\n1. First? (RoleRadar: Y)\n3. Third?';
    expect(parseExamples(markdown, 'RoleRadar')).toEqual(['First?', 'Second?', 'Fifth?']);
  });

  it('returns no examples for a page no item names', () => {
    expect(parseExamples(body, 'about')).toEqual([]);
  });

  it('uses numeric order and preserves question text apart from the trailing topic', () => {
    expect(parseExamples('Intro\r\n  4. Fourth (detail) question?   (Page: Section)\r\n3. Spare\r\n2. Second?\r\n1. First  question? (Page: Section)\r\nOutro'))
      .toEqual(['First  question?', 'Second?', 'Fourth (detail) question?']);
  });
});
