import { describe, expect, it, vi } from 'vitest'
import * as recent from './recent'
import type { RecentItem } from './recent'

const item = (id: string, at = '2026-11-15T10:00:00Z'): RecentItem => ({
  id,
  preview: `Claim ${id}`,
  label: 'Needs context',
  tone: 'warn',
  at,
})

describe('recent checks (this browser only)', () => {
  it('lists the newest check first', () => {
    recent.add(item('a'))
    recent.add(item('b'))
    expect(recent.list().map((i) => i.id)).toEqual(['b', 'a'])
  })

  it('moves a repeated check to the top instead of listing it twice', () => {
    recent.add(item('a'))
    recent.add(item('b'))
    recent.add({ ...item('a'), label: 'Well corroborated', tone: 'good' })
    expect(recent.list().map((i) => i.id)).toEqual(['a', 'b'])
    expect(recent.list()[0].label).toBe('Well corroborated')
  })

  it('keeps at most 20 checks, dropping the oldest', () => {
    for (let i = 0; i < 25; i++) recent.add(item(`id${i}`))
    const ids = recent.list().map((i) => i.id)
    expect(ids).toHaveLength(20)
    expect(ids[0]).toBe('id24')
    expect(ids).not.toContain('id4')
  })

  it('removes one check and leaves the rest', () => {
    recent.add(item('a'))
    recent.add(item('b'))
    recent.remove('a')
    expect(recent.list().map((i) => i.id)).toEqual(['b'])
  })

  it('ignores anything in storage that is not a list of checks', () => {
    window.localStorage.setItem('newscred.recent.v1', '{"not":"a list"}')
    expect(recent.list()).toEqual([])
    window.localStorage.setItem('newscred.recent.v1', 'not json')
    expect(recent.list()).toEqual([])
    window.localStorage.setItem('newscred.recent.v1', JSON.stringify([{ id: 5 }, item('ok')]))
    expect(recent.list().map((i) => i.id)).toEqual(['ok'])
  })

  it('keeps working, with an empty list, when the browser refuses storage', () => {
    const blocked = () => {
      throw new DOMException('denied', 'SecurityError')
    }
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(blocked)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(blocked)

    expect(recent.available()).toBe(false)
    expect(() => recent.add(item('a'))).not.toThrow()
    expect(recent.list()).toEqual([])
    expect(() => recent.remove('a')).not.toThrow()
  })

  it('reports that storage works in a normal window', () => {
    expect(recent.available()).toBe(true)
  })
})
