import { useEffect, useId, useState, type CSSProperties } from 'react';

const controlStyle: CSSProperties = {
  border: 'var(--border-width) solid var(--color-border)',
  background: 'var(--color-surface)',
  color: 'var(--color-text)',
  padding: 'var(--space-3)',
  borderRadius: 'var(--radius-small)',
  font: 'inherit',
};

export default function AskPlaceholder({ examples }: { examples: string[] }) {
  const inputId = useId();
  const [question, setQuestion] = useState('');
  const [status, setStatus] = useState('Checking assistant…');
  const [healthStatus, setHealthStatus] = useState('Checking database…');

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;

    async function checkPing() {
      try {
        const response = await fetch('/api/ping', { signal });
        if (!signal.aborted) {
          setStatus(response.status === 200 ? 'Assistant coming soon' : 'Assistant is offline');
        }
      } catch {
        if (!signal.aborted) setStatus('Assistant is offline');
      }
    }
    void checkPing();

    if (import.meta.env.DEV) {
      async function checkHealth() {
        try {
          const response = await fetch('/api/health', { signal });
          if (signal.aborted) return;
          const body: unknown = await response.json();
          if (signal.aborted) return;

          let message = 'db: down';
          if (typeof body === 'object' && body !== null) {
            if (response.status === 200 && 'pgvector' in body && typeof body.pgvector === 'string') {
              message = `db: ok · pgvector ${body.pgvector}`;
            } else if (response.status === 503 && 'db' in body && body.db === 'ok') {
              message = 'db: ok · pgvector missing';
            }
          }
          setHealthStatus(message);
        } catch {
          if (!signal.aborted) setHealthStatus('db: down');
        }
      }
      void checkHealth();
    }

    return () => controller.abort();
  }, []);

  return (
    <div style={{
      display: 'grid',
      gap: 'var(--space-4)',
      padding: 'var(--space-5)',
      border: 'var(--border-width) solid var(--color-border)',
      borderRadius: 'var(--radius-card)',
      background: 'var(--color-surface)',
      minInlineSize: 'var(--space-0)',
    }}>
      <form onSubmit={(event) => event.preventDefault()} style={{ display: 'grid', gap: 'var(--space-3)' }}>
        <label htmlFor={inputId} style={{ fontWeight: 'var(--font-weight-semibold)' }}>
          Ask about my projects
        </label>
        <textarea
          id={inputId}
          maxLength={500}
          rows={4}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          style={{ ...controlStyle, inlineSize: 'var(--size-full)', minInlineSize: 'var(--space-0)', resize: 'vertical' }}
        />
        <button type="submit" disabled style={{
          ...controlStyle,
          justifySelf: 'start',
          cursor: 'not-allowed',
          color: 'var(--color-muted)',
        }}>
          Send
        </button>
      </form>
      <div role="group" aria-label="Example questions" style={{ display: 'grid', gap: 'var(--space-2)' }}>
        <span style={{ fontWeight: 'var(--font-weight-semibold)' }}>Example questions</span>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)' }}>
          {examples.map((example) => (
            <button key={example} type="button" onClick={() => setQuestion(example)} style={{
              ...controlStyle,
              cursor: 'pointer',
              textAlign: 'start',
              maxInlineSize: 'var(--size-full)',
              minInlineSize: 'var(--space-0)',
              overflowWrap: 'anywhere',
            }}>
              {example}
            </button>
          ))}
        </div>
      </div>
      <p style={{ margin: 'var(--space-0)', fontSize: 'var(--font-size-small)', color: 'var(--color-muted)' }}>
        Your question is processed by AI providers. <a href="/privacy">See Privacy.</a>
      </p>
      <p role="status" style={{ margin: 'var(--space-0)', color: 'var(--color-muted)' }}>{status}</p>
      {import.meta.env.DEV && (
        <p style={{ margin: 'var(--space-0)', fontSize: 'var(--font-size-small)', color: 'var(--color-muted)' }}>
          {healthStatus}
        </p>
      )}
    </div>
  );
}
