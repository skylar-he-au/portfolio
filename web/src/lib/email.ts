export const EMAIL_USER = 'hello';
export const EMAIL_DOMAIN_PARTS = ['example', 'com'];

export function emailAddress(): string {
  return [EMAIL_USER, EMAIL_DOMAIN_PARTS.join('.')].join('@');
}

export function renderEmailPlaceholder(html: string, fallback: string): string {
  const escaped = fallback
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
  return html.replace(/\{\{email\}\}/g, () => `<span data-email-link>${escaped}</span>`);
}

export function mountEmailLinks(root: ParentNode = document): void {
  for (const span of root.querySelectorAll('span[data-email-link]')) {
    const link = span.ownerDocument.createElement('a');
    link.href = ['mailto', emailAddress()].join(':');
    link.textContent = emailAddress();
    link.className = 'email-link';
    span.replaceWith(link);
  }
}
