import { describe, expect, it } from 'vitest'
import { detectInput } from './detectInput'

describe('detectInput', () => {
  it('treats a lone web address as a link', () => {
    expect(detectInput('https://www.thehindu.com/news/x')).toEqual({
      kind: 'link',
      url: 'https://www.thehindu.com/news/x',
    })
  })

  it('accepts a link pasted without a scheme and with stray whitespace', () => {
    expect(detectInput('  thehindu.com/news/x \n')).toEqual({ kind: 'link', url: 'https://thehindu.com/news/x' })
  })

  it('accepts a bare domain, which the service uses to look up the outlet', () => {
    expect(detectInput('bbc.co.uk')).toEqual({ kind: 'link', url: 'https://bbc.co.uk' })
  })

  it('treats a sentence that contains a link as text', () => {
    const raw = 'Read this https://x.com/a now'
    expect(detectInput(raw)).toEqual({ kind: 'text', text: raw })
  })

  it.each(['Hello.', 'hello', 'U.S.', 'javascript:alert(1)', 'ftp://example.com/file', 'a@b.com'])(
    'treats "%s" as text, not as a link',
    (raw) => {
      expect(detectInput(raw).kind).toBe('text')
    },
  )

  it.each(['', '   ', '\n\t'])('reports nothing to check for blank input', (raw) => {
    expect(detectInput(raw)).toEqual({ kind: 'empty' })
  })

  it('trims text before sending it', () => {
    expect(detectInput('  A claim about something.  ')).toEqual({ kind: 'text', text: 'A claim about something.' })
  })
})
