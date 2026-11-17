import { useEffect, useId, useState } from 'react';

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

  // Styles live in src/styles/global.css (.ask-*); the inline token keeps grid children shrinkable.
  return (
    <div className="ask" style={{ minInlineSize: 'var(--space-0)' }}>
      <form className="ask-form" onSubmit={(event) => event.preventDefault()}>
        <label htmlFor={inputId} className="ask-label">
          Ask about my projects
        </label>
        <textarea
          id={inputId}
          className="ask-control ask-input"
          maxLength={500}
          rows={4}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button type="submit" disabled className="ask-control ask-send">
          Send
        </button>
      </form>
      <div role="group" aria-label="Example questions" className="ask-examples">
        <span className="ask-label">Example questions</span>
        <div className="ask-example-list">
          {examples.map((example) => (
            <button key={example} type="button" onClick={() => setQuestion(example)} className="ask-control ask-example">
              {example}
            </button>
          ))}
        </div>
      </div>
      <p className="ask-note">
        Your question is processed by AI providers. <a href="/privacy">See Privacy.</a>
      </p>
      <p role="status" className="ask-status">{status}</p>
      {import.meta.env.DEV && (
        <p className="ask-note">
          {healthStatus}
        </p>
      )}
    </div>
  );
}
