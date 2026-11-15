import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import type { Report } from '../api/types'
import * as recent from './recent'
import { parseRoute } from './router'
import * as session from './session'
import { STATUS_LINES, TIMING } from './timing'
import { useAnalyze } from './useAnalyze'

vi.mock('../api/client', async (original) => ({
  ...(await original<typeof import('../api/client')>()),
  analyze: vi.fn(),
}))
const { analyze } = await import('../api/client')
const analyzeMock = vi.mocked(analyze)

const REPORT: Report = { analysis_id: 'abc123', band: 'High', score: 88, claims: [] }

/** An API answer that arrives `ms` after the request was made. */
function answerAfter(ms: number, result: Report | Error) {
  analyzeMock.mockImplementation(
    (_input, signal) =>
      new Promise((resolve, reject) => {
        const timer = setTimeout(() => (result instanceof Error ? reject(result) : resolve(result)), ms)
        signal?.addEventListener('abort', () => {
          clearTimeout(timer)
          reject(new DOMException('cancelled', 'AbortError'))
        })
      }),
  )
}

const advance = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms))

beforeEach(() => vi.useFakeTimers())
afterEach(() => {
  vi.useRealTimers()
  analyzeMock.mockReset()
  session.clear()
})

describe('useAnalyze', () => {
  it('shows the press at once and never flashes a busy state for a stored answer', async () => {
    answerAfter(100, REPORT)
    const seen: string[] = []
    const { result } = renderHook(() => {
      const state = useAnalyze()
      seen.push(state.phase)
      return state
    })

    act(() => result.current.run({ text: 'A claim.' }))
    expect(result.current.phase).toBe('pressed')

    await advance(100)
    expect(result.current.phase).toBe('leaving')
    await advance(TIMING.leave)

    expect(seen).not.toContain('pending')
    expect(parseRoute(window.location.pathname)).toEqual({ name: 'check', id: 'abc123' })
  })

  it('shows the busy state after 300ms and holds it for at least 500ms once shown', async () => {
    answerAfter(400, REPORT)
    const { result } = renderHook(() => useAnalyze())
    act(() => result.current.run({ text: 'A claim.' }))

    await advance(299)
    expect(result.current.phase).toBe('pressed')
    await advance(1)
    expect(result.current.phase).toBe('pending')

    await advance(499) // the answer came at 400ms, but the busy state appeared at 300ms
    expect(result.current.phase).toBe('pending')
    await advance(1)
    expect(result.current.phase).toBe('leaving')
    await advance(TIMING.leave)
    expect(parseRoute(window.location.pathname)).toEqual({ name: 'check', id: 'abc123' })
  })

  it('moves the status line on as time passes, each line true at that moment', async () => {
    answerAfter(120_000, REPORT)
    const { result } = renderHook(() => useAnalyze())
    act(() => result.current.run({ text: 'A claim.' }))

    await advance(TIMING.showPendingAfter)
    expect(result.current.status).toBe(STATUS_LINES[0].text)
    await advance(2_000)
    expect(result.current.status).toBe(STATUS_LINES[1].text)
    await advance(20_000)
    expect(result.current.status).toBe(STATUS_LINES[2].text)
    await advance(40_000)
    expect(result.current.status).toBe(STATUS_LINES[3].text)
  })

  it('remembers a finished check for the result page and the recent list', async () => {
    answerAfter(100, REPORT)
    const { result } = renderHook(() => useAnalyze())
    act(() => result.current.run({ text: 'ISRO launches Chandrayaan-3.' }))
    await advance(100 + TIMING.leave)

    expect(session.get('abc123')).toMatchObject({ report: REPORT, input: { text: 'ISRO launches Chandrayaan-3.' } })
    expect(session.takeFresh('abc123')).toBe(true)
    expect(session.takeFresh('abc123')).toBe(false) // the reveal plays once
    expect(recent.list()[0]).toMatchObject({
      id: 'abc123',
      preview: 'ISRO launches Chandrayaan-3.',
      label: 'Well corroborated',
      tone: 'good',
    })
  })

  it('ends in a usable error state when the check is refused, keeping the reason', async () => {
    answerAfter(50, new ApiError('This does not look like English text.', 'refused', 422))
    const { result } = renderHook(() => useAnalyze())
    act(() => result.current.run({ text: 'asdf' }))
    await advance(50)

    expect(result.current.phase).toBe('error')
    expect(result.current.error?.message).toBe('This does not look like English text.')
    expect(window.location.pathname).toBe('/')

    answerAfter(50, REPORT) // and the reader can simply try again
    act(() => result.current.run({ text: 'A real claim.' }))
    expect(result.current.phase).toBe('pressed')
    expect(result.current.error).toBeNull()
  })

  it('goes straight back to idle when the reader cancels', async () => {
    answerAfter(5_000, REPORT)
    const { result } = renderHook(() => useAnalyze())
    act(() => result.current.run({ text: 'A claim.' }))
    await advance(1_000)
    expect(result.current.phase).toBe('pending')

    await act(async () => result.current.cancel())
    expect(result.current.phase).toBe('idle')
    expect(result.current.error).toBeNull()
    await advance(10_000)
    expect(window.location.pathname).toBe('/')
  })

  it('ignores a second submit while a check is running', async () => {
    answerAfter(1_000, REPORT)
    const { result } = renderHook(() => useAnalyze())
    act(() => result.current.run({ text: 'A claim.' }))
    act(() => result.current.run({ text: 'A claim.' }))
    expect(analyzeMock).toHaveBeenCalledTimes(1)
  })
})

describe('parseRoute', () => {
  it.each([
    ['/', { name: 'desk' }],
    ['/check/abc123', { name: 'check', id: 'abc123' }],
    ['/check/abc123/', { name: 'check', id: 'abc123' }],
    ['/how-it-works', { name: 'how' }],
    ['/check/', { name: 'not_found' }],
    ['/nope', { name: 'not_found' }],
  ])('%s', (path, route) => {
    expect(parseRoute(path)).toEqual(route)
  })
})
