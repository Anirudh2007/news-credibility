// Small wording and formatting helpers shared by the translation layer.

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/** "2024-03-12" or an ISO timestamp -> "12 Mar 2024"; anything else -> null. */
export function formatDate(iso: string | null | undefined): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso ?? '')
  if (!m) return null
  const [year, month, day] = [Number(m[1]), Number(m[2]), Number(m[3])]
  if (month < 1 || month > 12 || day < 1 || day > 31) return null
  return `${day} ${MONTHS[month - 1]} ${year}`
}

/** Evidence links come from the open web: only http and https addresses may become links. */
export function safeUrl(url: string | null | undefined): string | null {
  const trimmed = (url ?? '').trim()
  return /^https?:\/\//i.test(trimmed) ? trimmed : null
}

function host(url: string | null | undefined): string | null {
  const safe = safeUrl(url)
  if (!safe) return null
  try {
    return new URL(safe).hostname
  } catch {
    return null
  }
}

/** The name shown for a source: its domain without "www.", taken from the link if need be. */
export function publication(source: string | null | undefined, url: string | null | undefined): string {
  const name = (source ?? '').trim() || host(url) || ''
  return name.replace(/^www\./i, '') || 'Unnamed source'
}

// News passages are stored as "<site> reported (<date>): <text>". The site and date are shown under
// the quote, so the prefix is dropped.
const REPORTED_PREFIX = /^\S+ reported \(\d{4}-\d{2}-\d{2}\):\s*/

/** Some evidence is stored as "Q: ... A: ...". The question is context; the answer is the quote. */
export function splitQuote(snippet: string): { question: string | null; quote: string } {
  const text = snippet.trim().replace(REPORTED_PREFIX, '')
  const m = /^Q:\s*(.+?)\s+A:\s*(.+)$/s.exec(text)
  return m ? { question: m[1].trim(), quote: m[2].trim() } : { question: null, quote: text }
}

/** Evidence passages are stored cut to a fixed length. Say so when one stops mid-sentence. */
export function endQuote(text: string): string {
  const trimmed = text.trimEnd()
  return /[.!?…"”’')\]]$/.test(trimmed) ? trimmed : `${trimmed}…`
}

export function plural(n: number, one: string, many: string): string {
  return n === 1 ? one : many
}

/** "BOOM", "BOOM and Alt News", "BOOM, Alt News and Factly". */
export function list(items: string[]): string {
  if (items.length <= 1) return items.join('')
  return `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`
}

// How fact-checkers word their own ratings. Tested in this order: "partly false" is mixed, and
// "not true" is false. Mirrors the backend's reading of the same ratings.
const PARTLY = /\b(partly|partially|half[- ]true|mixed|mixture|missing context|needs context|exaggerat\w*|cherry[- ]?pick\w*|unsubstantiated)\b/i
const FALSE = /\b(false|fake|misleading|incorrect|inaccurate|untrue|not true|wrong|fabricated|hoax|scam|doctored|altered|manipulated|out of context|pants on fire|debunked|myth|deepfake|ai[- ]generated)\b/i
const TRUE = /\b(true|correct|accurate|real|genuine|verified)\b/i

export function ratingSays(rating: string | null | undefined): 'false' | 'true' | null {
  const text = (rating ?? '').trim()
  if (!text || PARTLY.test(text)) return null
  if (FALSE.test(text)) return 'false'
  if (TRUE.test(text)) return 'true'
  return null
}
