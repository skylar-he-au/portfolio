import { safeSiteUrl } from '../lib/answer';

export interface AnswerTrace {
  nodes: { name: string; provider: string | null; model: string | null; ms: number }[];
  sections: { url: string; heading: string; score: number }[];
  memory: { turn_id: string; question: string; score: number }[];
}

export default function HowItAnswered({ trace }: { trace?: AnswerTrace }) {
  if (!trace) return null;

  return (
    <details className="ask-panel">
      <summary>How it answered</summary>
      <div className="ask-panel-block">
        <h3>Steps</h3>
        <ol>
          {trace.nodes.map((node, index) => (
            <li key={index}>
              {node.name} · {node.provider ?? 'no model'}{node.model ? ` · ${node.model}` : ''} · {node.ms} ms
            </li>
          ))}
        </ol>
      </div>
      <div className="ask-panel-block">
        <h3>Passages found</h3>
        {trace.sections.length === 0 ? <p>None</p> : (
          <ul>
            {trace.sections.map((section, index) => {
              const href = safeSiteUrl(section.url);
              return (
                <li key={index}>
                  {href ? <a href={href}>{section.heading}</a> : section.heading} · {section.score.toFixed(3)}
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <div className="ask-panel-block">
        <h3>Memory recalled</h3>
        {trace.memory.length === 0 ? <p>None</p> : (
          <ul>
            {trace.memory.map((memory, index) => (
              <li key={index}>{memory.question} · {memory.score.toFixed(2)}</li>
            ))}
          </ul>
        )}
      </div>
    </details>
  );
}
