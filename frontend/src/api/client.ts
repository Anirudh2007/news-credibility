// The only place the app talks to the backend. Every call goes to the page's own origin; the dev
// server (vite.config.ts) or Caddy (frontend/Caddyfile) forwards it to FastAPI.

import type { AnalyzeInput, FeedbackBody, Health, Report } from './types'

export type ApiErrorKind = 'refused' | 'not_found' | 'network' | 'server'

/** An error whose message is already fit to show to a reader. */
export class ApiError extends Error {
  readonly status: number | null
  readonly kind: ApiErrorKind

  constructor(message: string, kind: ApiErrorKind, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
  }
}

const UNREACHABLE = 'We could not reach the checking service. Nothing was lost: try again in a moment.'
const SERVER_PROBLEM = 'The checking service had a problem. Nothing was lost: try again in a moment.'
// The input was refused before anything was analysed; the API's reason is written for the reader.
const REFUSALS = new Set([400, 413, 415, 422])

interface ValidationIssue {
  loc?: unknown[]
  msg?: string
}

async function refusalMessage(response: Response): Promise<string> {
  try {
    const { detail } = (await response.json()) as { detail?: string | ValidationIssue[] }
    if (Array.isArray(detail)) {
      return detail.map((d) => `${(d.loc ?? []).slice(1).join('.')}: ${d.msg ?? 'not valid'}`).join('; ')
    }
    if (typeof detail === 'string' && detail.trim()) return detail
  } catch {
    // not JSON: fall through to the generic wording
  }
  return 'That could not be checked. Try different text, a different link or a different image.'
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch (cause) {
    // A check the reader cancelled is not a failure; let the caller see the abort itself.
    if ((cause as { name?: string } | null)?.name === 'AbortError') throw cause
    throw new ApiError(UNREACHABLE, 'network')
  }
  if (response.ok) return response
  if (response.status === 404) throw new ApiError('That check could not be found.', 'not_found', 404)
  if (REFUSALS.has(response.status)) {
    throw new ApiError(await refusalMessage(response), 'refused', response.status)
  }
  throw new ApiError(SERVER_PROBLEM, 'server', response.status)
}

export async function analyze(input: AnalyzeInput, signal?: AbortSignal): Promise<Report> {
  const form = new FormData()
  const fields: Record<string, string | undefined> = {
    text: input.text?.trim(),
    url: input.url?.trim(),
    claimed_date: input.claimedDate,
    claimed_location: input.claimedLocation?.trim(),
    publish_date: input.publishDate,
  }
  for (const [name, value] of Object.entries(fields)) if (value) form.set(name, value)
  if (input.image) form.set('image', input.image, input.image.name)
  form.set('explain', 'true')
  form.set('refresh', input.refresh ? 'true' : 'false')

  const response = await request('/api/v1/analyze', { method: 'POST', body: form, signal })
  return (await response.json()) as Report
}

export async function getAnalysis(id: string): Promise<Report> {
  const response = await request(`/api/v1/analyses/${encodeURIComponent(id)}`)
  return (await response.json()) as Report
}

export async function deleteAnalysis(id: string): Promise<void> {
  // keepalive: the request still goes out if the reader closes the page straight after deleting
  await request(`/api/v1/analyses/${encodeURIComponent(id)}`, { method: 'DELETE', keepalive: true })
}

export async function sendFeedback(body: FeedbackBody): Promise<void> {
  await request('/api/v1/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export async function getHealth(): Promise<Health> {
  const response = await request('/health')
  return (await response.json()) as Health
}
