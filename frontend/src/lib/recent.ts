// "Recent checks": the checks made in this browser, kept in this browser. The server stores every
// analysis, but the site never lists other people's; a check is reachable only by its own link.

import type { Tone } from '../translate/types'

export interface RecentItem {
  id: string
  /** the start of what was submitted: text, a link, or "An image" */
  preview: string
  /** the verdict label shown when the check was made */
  label: string
  tone: Tone
  /** ISO time of the check */
  at: string
}

const KEY = 'newscred.recent.v1'
const LIMIT = 20

function isItem(value: unknown): value is RecentItem {
  const v = value as Partial<RecentItem> | null
  return (
    typeof v === 'object' &&
    v !== null &&
    typeof v.id === 'string' &&
    typeof v.preview === 'string' &&
    typeof v.label === 'string' &&
    typeof v.tone === 'string' &&
    typeof v.at === 'string'
  )
}

// Storage can be missing or refuse access (private windows, blocked site data). Every read and
// write is guarded: the app works without the list.
function read(): RecentItem[] {
  try {
    const parsed: unknown = JSON.parse(window.localStorage.getItem(KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed.filter(isItem) : []
  } catch {
    return []
  }
}

function write(items: RecentItem[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(items.slice(0, LIMIT)))
  } catch {
    // nothing to do: the list simply is not kept
  }
}

export function list(): RecentItem[] {
  return read()
}

export function add(item: RecentItem): void {
  write([item, ...read().filter((i) => i.id !== item.id)])
}

export function remove(id: string): void {
  write(read().filter((i) => i.id !== id))
}

export function find(id: string): RecentItem | undefined {
  return read().find((i) => i.id === id)
}

/** False when the browser will not keep anything, so the drawer can say why it is empty. */
export function available(): boolean {
  try {
    const probe = `${KEY}.probe`
    window.localStorage.setItem(probe, '1')
    window.localStorage.removeItem(probe)
    return true
  } catch {
    return false
  }
}
