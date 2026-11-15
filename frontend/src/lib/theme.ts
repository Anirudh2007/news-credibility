// The reader's theme: follow the system (the default), or always Light, or always Dark.
//
// The choice lives in two places. `data-theme` on <html> is what the stylesheet reads (theme.css);
// localStorage is what brings the choice back on the next visit. index.html applies the stored choice
// before the app loads, so the page never flashes the other theme.

import { useSyncExternalStore } from 'react'

export type ThemeChoice = 'system' | 'light' | 'dark'

/** Also written out in index.html: keep the two the same. */
export const THEME_KEY = 'newscred.theme'

const listeners = new Set<() => void>()

function isForced(value: unknown): value is 'light' | 'dark' {
  return value === 'light' || value === 'dark'
}

export function readTheme(): ThemeChoice {
  try {
    const stored = window.localStorage.getItem(THEME_KEY)
    if (isForced(stored)) return stored
    if (stored !== null) return 'system' // something unexpected was stored
  } catch {
    // storage refused (private window): fall through to what the page itself is showing
  }
  const applied = document.documentElement.dataset.theme
  return isForced(applied) ? applied : 'system'
}

/** Match the browser's own chrome (the address bar on a phone) to the page colour. */
export function syncThemeColor(): void {
  const paper = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim()
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
  if (paper && meta) meta.content = paper
}

export function setTheme(choice: ThemeChoice): void {
  const root = document.documentElement

  // Switch every colour on the same frame: without this, anything with a colour transition would
  // fade to the new theme a beat behind the rest of the page.
  root.classList.add('theme-switching')
  if (choice === 'system') root.removeAttribute('data-theme')
  else root.dataset.theme = choice
  const settle = () => root.classList.remove('theme-switching')
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => requestAnimationFrame(settle))
  else settle()

  try {
    if (choice === 'system') window.localStorage.removeItem(THEME_KEY)
    else window.localStorage.setItem(THEME_KEY, choice)
  } catch {
    // not remembered for next time; it still applies to this visit
  }
  syncThemeColor()
  listeners.forEach((notify) => notify())
}

function subscribe(onChange: () => void): () => void {
  listeners.add(onChange)
  return () => listeners.delete(onChange)
}

/** The current choice and a way to change it; every component using it stays in step. */
export function useTheme(): [ThemeChoice, (choice: ThemeChoice) => void] {
  return [useSyncExternalStore(subscribe, readTheme), setTheme]
}

/** Called once at start-up: keep the browser chrome colour right when the system theme flips. */
export function watchSystemTheme(): void {
  syncThemeColor()
  if (typeof window.matchMedia !== 'function') return
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', syncThemeColor)
}
