import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { THEME_KEY, readTheme, setTheme } from './theme'

const root = () => document.documentElement

describe('theme choice', () => {
  it('follows the system until the reader chooses otherwise', () => {
    expect(readTheme()).toBe('system')
    expect(root().hasAttribute('data-theme')).toBe(false)
  })

  it('applies and remembers Dark', () => {
    setTheme('dark')
    expect(root().dataset.theme).toBe('dark')
    expect(window.localStorage.getItem(THEME_KEY)).toBe('dark')
    expect(readTheme()).toBe('dark')
  })

  it('applies and remembers Light, even though light is also what a light system shows', () => {
    setTheme('light')
    expect(root().dataset.theme).toBe('light')
    expect(readTheme()).toBe('light')
  })

  it('goes back to following the system and forgets the earlier choice', () => {
    setTheme('dark')
    setTheme('system')
    expect(root().hasAttribute('data-theme')).toBe(false)
    expect(window.localStorage.getItem(THEME_KEY)).toBeNull()
    expect(readTheme()).toBe('system')
  })

  it('treats anything unexpected in storage as "follow the system"', () => {
    window.localStorage.setItem(THEME_KEY, 'sepia')
    expect(readTheme()).toBe('system')
  })

  it('still switches for this visit when the browser refuses storage', () => {
    const blocked = () => {
      throw new DOMException('denied', 'SecurityError')
    }
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(blocked)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(blocked)
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(blocked)

    expect(() => setTheme('dark')).not.toThrow()
    expect(root().dataset.theme).toBe('dark')
    expect(readTheme()).toBe('dark') // read back from the page itself
  })

  it('is applied by index.html before the app loads, under the same storage key', () => {
    const html = readFileSync(resolve(process.cwd(), 'index.html'), 'utf8')
    expect(html).toContain(`localStorage.getItem('${THEME_KEY}')`)
    expect(html).toMatch(/<script>[\s\S]*dataset\.theme[\s\S]*<\/script>[\s\S]*<\/head>/)
  })
})
