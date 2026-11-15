import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

// The colour tokens are checked where they are defined: text colours must reach WCAG AA (4.5:1) on
// both backgrounds, and colours used only for icons, dots and borders must reach 3:1.

const css = readFileSync(resolve(process.cwd(), 'src/styles/theme.css'), 'utf8')

function tokens(block: string): Record<string, string> {
  const out: Record<string, string> = {}
  for (const m of block.matchAll(/--([a-z-]+):\s*(#[0-9a-fA-F]{6})\s*;/g)) out[m[1]] = m[2]
  return out
}

const block = (selector: RegExp) => css.match(new RegExp(selector.source + /\s*\{([^}]*)\}/.source))?.[1] ?? ''

// Light is the base. Dark is written twice: once for "follow the system" (unless the reader chose
// Light) and once for a reader who chose Dark.
const light = tokens(block(/:root/))
const darkBySystem = tokens(block(/prefers-color-scheme:\s*dark\)\s*\{\s*:root:not\(\[data-theme="light"\]\)/))
const darkByChoice = tokens(block(/:root\[data-theme="dark"\]/))
const themes = { light, dark: darkByChoice }

function luminance(hex: string): number {
  const channel = (i: number) => {
    const v = parseInt(hex.slice(i, i + 2), 16) / 255
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4
  }
  return 0.2126 * channel(1) + 0.7152 * channel(3) + 0.0722 * channel(5)
}

function contrast(a: string, b: string): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x)
  return (hi + 0.05) / (lo + 0.05)
}

const TEXT = ['text', 'ink', 'muted', 'accent', 'rust']
const GRAPHIC = ['amber', 'sky', 'grey']
const BACKGROUNDS = ['paper', 'surface']

describe('theme choice', () => {
  it('is cream paper in the light theme', () => {
    expect(light.paper).toBe('#F6F3EC')
  })

  it('uses the same dark colours whether dark comes from the system or from the reader’s choice', () => {
    expect(Object.keys(darkByChoice).length).toBeGreaterThan(8)
    expect(darkBySystem).toEqual(darkByChoice)
  })

  it('lets a reader who chose Light keep it on a system that is in dark mode', () => {
    expect(css).toMatch(/prefers-color-scheme:\s*dark\)\s*\{\s*:root:not\(\[data-theme="light"\]\)/)
  })

  it('does not animate colours across the page when the theme changes', () => {
    expect(css).toMatch(/\.theme-switching \*[^{]*\{[^}]*transition:\s*none\s*!important/)
  })
})

describe.each(Object.entries(themes))('%s theme colours', (_name, theme) => {
  it('defines every token', () => {
    for (const name of [...TEXT, ...GRAPHIC, ...BACKGROUNDS, 'line']) expect(theme[name], name).toMatch(/^#/)
  })

  it.each(TEXT.flatMap((t) => BACKGROUNDS.map((b) => [t, b])))('%s on %s reads at 4.5:1 or better', (t, b) => {
    expect(contrast(theme[t], theme[b])).toBeGreaterThanOrEqual(4.5)
  })

  it.each(GRAPHIC.flatMap((t) => BACKGROUNDS.map((b) => [t, b])))('%s on %s is visible at 3:1 or better', (t, b) => {
    expect(contrast(theme[t], theme[b])).toBeGreaterThanOrEqual(3)
  })

  it('primary button label reads at 4.5:1 or better on the accent colour', () => {
    expect(contrast(theme['on-accent'], theme['accent'])).toBeGreaterThanOrEqual(4.5)
  })
})
