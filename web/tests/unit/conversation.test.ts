import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { adoptConversationId, clearConversation, loadConversation, MAX_STORED_TURNS, saveConversation } from '../../src/lib/conversation';
import type { StoredConversation, StoredTurn } from '../../src/lib/conversation';

const turn: StoredTurn = {
  turnId: 'turn-1', question: 'Question?', text: 'Answer [1]',
  citations: [{ n: 1, url: '/about', heading: 'About' }], unsupported: ['Answer'],
  checkResult: { passed: true, reason: null },
  trace: {
    nodes: [{ name: 'answer', provider: null, model: 'model', ms: 1 }],
    sections: [{ url: '/about', heading: 'About', score: 0.5 }],
    memory: [{ turn_id: 'old', question: 'Earlier?', score: 0.2 }],
  },
  feedback: 'sent', private: true,
};
const conversation: StoredConversation = { v: 1, id: 'conversation-1', turns: [turn], forgotten: true };

beforeEach(() => sessionStorage.clear());
afterEach(() => vi.restoreAllMocks());

describe('conversation storage', () => {
  it('round-trips every field', () => {
    saveConversation(conversation);
    expect(loadConversation()).toEqual(conversation);
  });

  it('trims to the last ten on save and load', () => {
    expect(MAX_STORED_TURNS).toBe(10);
    const turns = Array.from({ length: 12 }, (_, index) => ({ ...turn, question: String(index) }));
    saveConversation({ ...conversation, turns });
    expect(JSON.parse(sessionStorage.getItem('conversation')!).turns).toEqual(turns.slice(-10));
    sessionStorage.setItem('conversation', JSON.stringify({ ...conversation, turns }));
    expect(loadConversation()?.turns).toEqual(turns.slice(-10));
  });

  it.each([
    null, '{', JSON.stringify({ ...conversation, v: 2 }), JSON.stringify({ turns: [] }),
    JSON.stringify({ ...conversation, id: '' }), JSON.stringify({ ...conversation, id: '  ' }),
    JSON.stringify({ ...conversation, turns: {} }), JSON.stringify({ ...conversation, turns: [{}] }),
    JSON.stringify({ ...conversation, forgotten: false }),
  ])('rejects malformed storage %s', (raw) => {
    if (raw !== null) sessionStorage.setItem('conversation', raw);
    expect(loadConversation()).toBeNull();
  });

  it.each([
    { turnId: 1 }, { question: null }, { text: [] }, { citations: [{ n: '1', url: '/', heading: 'h' }] },
    { citations: [{ n: 1, url: false, heading: 'h' }] }, { citations: [{ n: 1, url: '/', heading: 1 }] },
    { unsupported: [1] }, { private: 'yes' }, { checkResult: { passed: 'yes', reason: null } },
    { checkResult: { passed: true, reason: 1 } }, { trace: { nodes: [], sections: [] } },
    { trace: { nodes: [{ name: 'a', provider: 1, model: null, ms: 1 }], sections: [], memory: [] } },
    { trace: { nodes: [], sections: [{ url: '/', heading: 'h', score: '1' }], memory: [] } },
    { trace: { nodes: [], sections: [], memory: [{ turn_id: 't', question: 1, score: 1 }] } },
  ])('rejects the whole conversation for a malformed field %j', (change) => {
    sessionStorage.setItem('conversation', JSON.stringify({ ...conversation, turns: [turn, { ...turn, ...change }] }));
    expect(loadConversation()).toBeNull();
  });

  it.each(['sending', undefined, 1])('normalizes feedback %s to idle', (feedback) => {
    sessionStorage.setItem('conversation', JSON.stringify({ ...conversation, turns: [{ ...turn, feedback }] }));
    expect(loadConversation()?.turns[0]?.feedback).toBe('idle');
  });

  it('ignores storage read, write, and remove failures', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('Blocked'); });
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('Blocked'); });
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => { throw new Error('Blocked'); });
    expect(loadConversation()).toBeNull();
    expect(() => saveConversation(conversation)).not.toThrow();
    expect(() => clearConversation()).not.toThrow();
  });

  it('clears the key', () => {
    saveConversation(conversation);
    clearConversation();
    expect(sessionStorage.getItem('conversation')).toBeNull();
  });

  it('adopts a replacement id, keeping the display history', () => {
    expect(adoptConversationId(conversation, 'new')).toEqual({
      conversation: { ...conversation, id: 'new', forgotten: true }, replaced: true,
    });
  });

  it('keeps the same id and forgotten flag', () => {
    expect(adoptConversationId(conversation, conversation.id)).toEqual({ conversation, replaced: false });
    const fresh = { v: 1 as const, id: 'same', turns: [] };
    expect(adoptConversationId(fresh, 'same')).toEqual({ conversation: fresh, replaced: false });
  });

  it('creates an empty conversation for a first id', () => {
    expect(adoptConversationId(null, 'first')).toEqual({ conversation: { v: 1, id: 'first', turns: [] }, replaced: false });
  });
});
