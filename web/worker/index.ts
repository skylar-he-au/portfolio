interface Env {
  ORIGIN_URL: string;
  ORIGIN_SECRET?: string;
}

function unavailable(): Response {
  return new Response('{"error":"unavailable"}', {
    status: 503,
    headers: {
      'Content-Type': 'application/json',
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    let target: string;
    try {
      if (typeof env.ORIGIN_URL !== 'string') return unavailable();
      const origin = new URL(env.ORIGIN_URL);
      if (origin.protocol !== 'http:' && origin.protocol !== 'https:') return unavailable();
      const url = new URL(request.url);
      target = env.ORIGIN_URL.replace(/\/+$/, '') + url.pathname + url.search;
    } catch {
      return unavailable();
    }

    const clientIp = request.headers.get('CF-Connecting-IP');
    const headers = new Headers(request.headers);
    for (const name of ['X-Client-IP', 'X-Origin-Secret', 'CF-Connecting-IP', 'X-Forwarded-For', 'X-Real-IP']) {
      headers.delete(name);
    }
    if (clientIp !== null) headers.set('X-Client-IP', clientIp);
    if (typeof env.ORIGIN_SECRET === 'string' && env.ORIGIN_SECRET.length > 0) {
      headers.set('X-Origin-Secret', env.ORIGIN_SECRET);
    }

    // The browser RequestInit type omits duplex, which streaming fetch requires.
    const options: RequestInit & { duplex: 'half' } = {
      method: request.method,
      headers,
      body: request.body,
      redirect: 'manual',
      duplex: 'half',
    };
    let upstream: Response;
    try {
      upstream = await fetch(target, options);
    } catch {
      return unavailable();
    }
    if ([502, 503, 504].includes(upstream.status)) {
      // Cancel without delaying the fallback if the upstream cleanup stalls.
      void upstream.body?.cancel().catch(() => {});
      return unavailable();
    }

    const response = new Response(upstream.body, upstream);
    response.headers.set('X-Content-Type-Options', 'nosniff');
    return response;
  },
};
