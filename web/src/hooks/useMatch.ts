import { useEffect, useRef, useState } from 'react';
import {
  MATCH_DROPPED_MESSAGE, MATCH_MAX_CHARS, MATCH_MESSAGES, matchErrorMessage, matchHttpMessage,
  matchInputLength, runMatch, sendMatchFeedback,
} from '../lib/match';
import type { MatchEvent, MatchNotes, MatchResult } from '../lib/match';
import { acquireWidget, HttpError, prepareWidget, VerificationError } from '../lib/session';

export type MatchStepName = 'Fetch job' | 'Read job' | 'Match' | 'Check' | 'Write';
export interface MatchStep { name: MatchStepName; state: 'active' | 'done' }

const stepNames: Record<Extract<MatchEvent, { event: 'stage' }>['data']['stage'], MatchStepName> = {
  fetching: 'Fetch job', reading: 'Read job', matching: 'Match', checking: 'Check', writing: 'Write',
};

export function useMatch(): {
  inProgress: boolean;
  stage: { stage: 'fetching' | 'reading' | 'matching' | 'checking' | 'writing'; count?: number } | undefined;
  steps: MatchStep[];
  notes?: MatchNotes;
  result?: MatchResult;
  error?: string;
  done: boolean;
  feedback: 'idle' | 'sending' | 'sent';
  prepare: () => void;
  match: (input: string) => { started: boolean; finished: Promise<void> };
  recordFeedback: () => Promise<void>;
} {
  const [inProgress, setInProgress] = useState(false);
  const [stage, setStage] = useState<Extract<MatchEvent, { event: 'stage' }>['data']>();
  const [steps, setSteps] = useState<MatchStep[]>([]);
  const [notes, setNotes] = useState<MatchNotes>();
  const [result, setResult] = useState<MatchResult>();
  const [error, setError] = useState<string>();
  const [done, setDone] = useState(false);
  const [feedback, setFeedback] = useState<'idle' | 'sending' | 'sent'>('idle');
  const mounted = useRef(true);
  const requestRef = useRef<AbortController | null>(null);
  const feedbackRef = useRef<AbortController | null>(null);

  useEffect(() => {
    mounted.current = true;
    const releaseWidget = acquireWidget();
    return () => {
      mounted.current = false;
      requestRef.current?.abort();
      feedbackRef.current?.abort();
      releaseWidget();
    };
  }, []);

  async function recordFeedback() {
    if (!mounted.current || !done || error || !result?.match_id || feedback !== 'idle' || feedbackRef.current) return;
    const controller = new AbortController();
    feedbackRef.current = controller;
    setFeedback('sending');
    const sent = await sendMatchFeedback(result.match_id, controller.signal);
    if (!mounted.current || controller.signal.aborted) return;
    feedbackRef.current = null;
    setFeedback(sent ? 'sent' : 'idle');
  }

  function match(input: string): { started: boolean; finished: Promise<void> } {
    const trimmed = input.trim();
    if (!trimmed || requestRef.current) return { started: false, finished: Promise.resolve() };
    feedbackRef.current?.abort();
    feedbackRef.current = null;
    setResult(undefined);
    setNotes(undefined);
    setError(undefined);
    setStage(undefined);
    setSteps([]);
    setDone(false);
    setFeedback('idle');
    if (matchInputLength(trimmed) > MATCH_MAX_CHARS) {
      setError(MATCH_MESSAGES.too_long);
      return { started: false, finished: Promise.resolve() };
    }

    const controller = new AbortController();
    requestRef.current = controller;
    setInProgress(true);
    return { started: true, finished: run() };

    async function run() {
      try {
        await runMatch(trimmed, {
          signal: controller.signal,
          onEvent(event) {
            if (!mounted.current || controller.signal.aborted) return;
            switch (event.event) {
              case 'stage': {
                setStage(event.data);
                const name = stepNames[event.data.stage];
                setSteps((previous) => {
                  const steps: MatchStep[] = previous.map((step) => ({ ...step, state: 'done' }));
                  const existing = steps.find((step) => step.name === name);
                  if (existing) existing.state = 'active';
                  else steps.push({ name, state: 'active' });
                  return steps;
                });
                break;
              }
              case 'notes': setNotes(event.data); break;
              case 'result': setResult(event.data); break;
              case 'done':
                setDone(true);
                setSteps((previous) => previous.map((step) => ({ ...step, state: 'done' })));
                break;
              case 'error': setError(matchErrorMessage(event.data.code, event.data.message)); break;
            }
          },
        });
      } catch (error) {
        if (!mounted.current || controller.signal.aborted) return;
        setError(error instanceof VerificationError ? MATCH_MESSAGES.unavailable
          : error instanceof HttpError ? matchHttpMessage(error) : MATCH_DROPPED_MESSAGE);
      } finally {
        if (requestRef.current === controller) requestRef.current = null;
        if (mounted.current && !controller.signal.aborted) {
          setInProgress(false);
          setStage(undefined);
        }
      }
    }
  }

  return { inProgress, stage, steps, notes, result, error, done, feedback, prepare: prepareWidget, match, recordFeedback };
}
