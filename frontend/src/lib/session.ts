// Checks made since the page loaded, with what was submitted. Lets the result page show the input
// and offer "Check again" without another request, and tells it whether to play the reveal.

import type { AnalyzeInput, Report } from '../api/types'

export interface SessionEntry {
  report: Report
  input?: AnalyzeInput
  /** an object URL for the submitted image; the server keeps only a fingerprint of it */
  imageUrl?: string | null
  /** true until the result has been shown once */
  fresh: boolean
}

const entries = new Map<string, SessionEntry>()

export function put(id: string, entry: SessionEntry): void {
  entries.set(id, entry)
}

export function get(id: string): SessionEntry | undefined {
  return entries.get(id)
}

/** True exactly once per check: the verdict reveal is for a result the reader just asked for. */
export function takeFresh(id: string): boolean {
  const entry = entries.get(id)
  if (!entry?.fresh) return false
  entry.fresh = false
  return true
}

export function clear(): void {
  entries.clear()
}
