export interface SseEvent {
  event: string;
  data: unknown;
}

export async function* parseSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let name = 'message';
  let data: string[] = [];
  let ended = false;

  function lineEvent(line: string): SseEvent | undefined {
    if (line === '') {
      const event = name;
      const json = data.join('\n');
      name = 'message';
      data = [];
      if (!json) return;
      try {
        return { event, data: JSON.parse(json) as unknown };
      } catch {
        return;
      }
    }
    if (line.startsWith(':')) return;
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') name = value || 'message';
    if (field === 'data') data.push(value);
  }

  try {
    while (!ended) {
      const chunk = await reader.read();
      ended = chunk.done;
      buffer += ended ? decoder.decode() : decoder.decode(chunk.value, { stream: true });
      let newline = buffer.indexOf('\n');
      while (newline >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, '');
        buffer = buffer.slice(newline + 1);
        const event = lineEvent(line);
        if (event) yield event;
        newline = buffer.indexOf('\n');
      }
    }
  } finally {
    try {
      if (!ended) await reader.cancel();
    } finally {
      reader.releaseLock();
    }
  }
}
