// One field takes a link, a claim or a whole article. This decides which it is, and the desk shows
// the answer under the field ("Checking this link" / "Checking this text") so it is never a surprise.

export type DetectedInput =
  | { kind: 'empty' }
  | { kind: 'link'; url: string }
  | { kind: 'text'; text: string }

const HAS_SCHEME = /^[a-z][a-z0-9+.-]*:/i
// A host is two or more dot-separated labels ending in a top-level domain of letters.
const HOST = /^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$/i

function asLink(token: string): string | null {
  if (token.includes('@')) return null // an e-mail address, not a page
  const isWeb = /^https?:\/\//i.test(token)
  if (HAS_SCHEME.test(token) && !isWeb) return null // javascript:, ftp:, mailto: and the like
  const candidate = isWeb ? token : `https://${token}`
  try {
    const { hostname } = new URL(candidate)
    return HOST.test(hostname) ? candidate : null
  } catch {
    return null
  }
}

export function detectInput(raw: string): DetectedInput {
  const text = raw.trim()
  if (!text) return { kind: 'empty' }
  // A link is the whole input and nothing else; a sentence that mentions a link is still text.
  if (!/\s/.test(text) && !text.endsWith('.')) {
    const url = asLink(text)
    if (url) return { kind: 'link', url }
  }
  return { kind: 'text', text }
}
