import type { Citation } from './answer';
import type { AnswerTrace, CheckResult } from '../components/HowItAnswered';

export const MAX_STORED_TURNS = 10;

export interface StoredTurn {
  turnId?: string;
  question: string;
  text: string;
  citations: Citation[];
  unsupported: string[];
  checkResult?: CheckResult;
  trace?: AnswerTrace;
  feedback: 'idle' | 'sent';
  private?: boolean;
}

export interface StoredConversation { v: 1; id: string; turns: StoredTurn[]; forgotten?: true }

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function trace(value: unknown): value is AnswerTrace {
  return record(value)
    && Array.isArray(value.nodes) && value.nodes.every((node: unknown) => record(node)
      && typeof node.name === 'string' && (node.provider === null || typeof node.provider === 'string')
      && (node.model === null || typeof node.model === 'string') && finite(node.ms))
    && Array.isArray(value.sections) && value.sections.every((section: unknown) => record(section)
      && typeof section.url === 'string' && typeof section.heading === 'string' && finite(section.score))
    && Array.isArray(value.memory) && value.memory.every((memory: unknown) => record(memory)
      && typeof memory.turn_id === 'string' && typeof memory.question === 'string' && finite(memory.score));
}

function storedTurn(value: unknown): StoredTurn | null {
  if (!record(value) || typeof value.question !== 'string' || typeof value.text !== 'string'
    || (value.turnId !== undefined && typeof value.turnId !== 'string')
    || (value.private !== undefined && typeof value.private !== 'boolean')
    || !Array.isArray(value.citations) || !value.citations.every((citation: unknown) => record(citation)
      && finite(citation.n) && typeof citation.url === 'string' && typeof citation.heading === 'string'
      // Quotes arrived in step 10: turns stored before that have none and stay valid.
      && (citation.quotes === undefined
        || (Array.isArray(citation.quotes) && citation.quotes.every((quote: unknown) => typeof quote === 'string'))))
    || !Array.isArray(value.unsupported) || !value.unsupported.every((span: unknown) => typeof span === 'string')
    || (value.checkResult !== undefined && (!record(value.checkResult)
      || typeof value.checkResult.passed !== 'boolean'
      || (value.checkResult.reason !== null && typeof value.checkResult.reason !== 'string')))
    || (value.trace !== undefined && !trace(value.trace))) return null;
  return {
    turnId: value.turnId as string | undefined,
    question: value.question, text: value.text, citations: value.citations as Citation[],
    unsupported: value.unsupported as string[],
    checkResult: value.checkResult as CheckResult | undefined,
    trace: value.trace as AnswerTrace | undefined,
    feedback: value.feedback === 'sent' ? 'sent' : 'idle',
    private: value.private as boolean | undefined,
  };
}

export function loadConversation(): StoredConversation | null {
  try {
    const value: unknown = JSON.parse(sessionStorage.getItem('conversation') ?? 'null');
    if (!record(value) || value.v !== 1 || typeof value.id !== 'string' || !value.id.trim()
      || !Array.isArray(value.turns) || (value.forgotten !== undefined && value.forgotten !== true)) return null;
    const turns: StoredTurn[] = [];
    for (const item of value.turns) {
      const turn = storedTurn(item);
      if (!turn) return null;
      turns.push(turn);
    }
    return { v: 1, id: value.id, turns: turns.slice(-MAX_STORED_TURNS), ...(value.forgotten ? { forgotten: true } : {}) };
  } catch {
    return null;
  }
}

export function saveConversation(conversation: StoredConversation): void {
  try {
    sessionStorage.setItem('conversation', JSON.stringify({ ...conversation, turns: conversation.turns.slice(-MAX_STORED_TURNS) }));
  } catch {
    // Storage can be blocked; the conversation still works in memory.
  }
}

export function clearConversation(): void {
  try {
    sessionStorage.removeItem('conversation');
  } catch {
    // Storage can be blocked.
  }
}

export function adoptConversationId(
  stored: StoredConversation | null, id: string,
): { conversation: StoredConversation; replaced: boolean } {
  if (!stored) return { conversation: { v: 1, id, turns: [] }, replaced: false };
  if (stored.id === id) return { conversation: stored, replaced: false };
  return { conversation: { v: 1, id, turns: stored.turns, forgotten: true }, replaced: true };
}
