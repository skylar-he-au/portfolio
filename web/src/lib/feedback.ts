import { getToken } from './session';

export async function sendFeedback(turnId: string, signal?: AbortSignal): Promise<boolean> {
  try {
    const token = getToken();
    if (token === null) return false;
    const response = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
      body: JSON.stringify({ turn_id: turnId }),
      signal,
    });
    return response.ok;
  } catch {
    return false;
  }
}
