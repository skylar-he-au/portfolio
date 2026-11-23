import { safeSiteUrl } from '../lib/answer';

export interface AnswerTrace {
  nodes: { name: string; provider: string | null; model: string | null; ms: number }[];
  sections: { url: string; heading: string; score: number }[];
  memory: { turn_id: string; question: string; score: number }[];
}

export interface CheckResult {
  passed: boolean;
  reason: string | null;
}

export default function HowItAnswered({ trace, checkResult }: { trace?: AnswerTrace; checkResult?: CheckResult }) {
  if (!trace) return null;

  let checkReason: string | undefined;
  if (checkResult && !checkResult.passed) {
    switch (checkResult.reason) {
      case 'unsupported': checkReason = "Some sentences weren't found in the project pages."; break;
      case 'not_in_about': checkReason = "The About page doesn't answer this."; break;
      case 'inferred': checkReason = 'The draft said more than the About page does.'; break;
    }
  }

  return (
    <details className="ask-panel">
      <summary>How it answered</summary>
      {checkReason && (
        <div className="ask-panel-block">
          <h3>Check</h3>
          <p>{checkReason}</p>
        </div>
      )}
      <div className="ask-panel-block">
        <h3>Steps</h3>
        <ol className="ask-trace-steps">
          {trace.nodes.map((node, index) => (
            <li key={index}>
              <span className="ask-trace-icon" aria-hidden="true">✓</span>
              <span>{node.name} · {node.provider ?? 'no model'}{node.model ? ` · ${node.model}` : ''} · {node.ms} ms</span>
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
