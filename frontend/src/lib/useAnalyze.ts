// The life of one check, from the press to the result page. Timings follow the storyboard in
// timing.ts: the press registers at once, a busy state appears only if the answer is slow, and the
// desk fades out before the verdict comes in.

import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, analyze } from '../api/client'
import type { AnalyzeInput, Report } from '../api/types'
import { translateReport } from '../translate'
import * as recent from './recent'
import { navigate } from './router'
import * as session from './session'
import { STATUS_LINES, TIMING, prefersReducedMotion } from './timing'

export type AnalyzePhase = 'idle' | 'pressed' | 'pending' | 'leaving' | 'error'

interface Run {
  controller: AbortController
  timers: number[]
}

function previewOf(input: AnalyzeInput): string {
  return (input.text ?? input.url ?? (input.image ? 'An image' : '')).trim().slice(0, 140)
}

function imagePreview(image: File | undefined): string | null {
  return image && typeof URL.createObjectURL === 'function' ? URL.createObjectURL(image) : null
}

function remember(report: Report, input: AnalyzeInput): void {
  const view = translateReport(report)
  session.put(report.analysis_id, { report, input, imageUrl: imagePreview(input.image), fresh: true })
  recent.add({
    id: report.analysis_id,
    preview: previewOf(input),
    label: view.verdict.label,
    tone: view.verdict.tone,
    at: report.created_at ?? new Date().toISOString(),
  })
}

function clearTimers(run: Run): void {
  run.timers.forEach((t) => window.clearTimeout(t))
  run.timers = []
}

export function useAnalyze() {
  const [phase, setPhase] = useState<AnalyzePhase>('idle')
  const [status, setStatus] = useState<string>(STATUS_LINES[0].text)
  const [error, setError] = useState<ApiError | null>(null)
  // true from the moment the busy state appears until the check ends: decides whether the skeleton shows
  const [progressShown, setProgressShown] = useState(false)
  const current = useRef<Run | null>(null)

  // Leaving the page mid-check abandons it quietly.
  useEffect(
    () => () => {
      const running = current.current
      if (!running) return
      clearTimers(running)
      current.current = null
      running.controller.abort()
    },
    [],
  )

  const run = useCallback((input: AnalyzeInput) => {
    if (current.current) return // a check is already running: a second press does nothing
    const thisRun: Run = { controller: new AbortController(), timers: [] }
    current.current = thisRun
    const later = (fn: () => void, ms: number) => {
      thisRun.timers.push(window.setTimeout(fn, ms))
    }

    let shownAt: number | null = null
    setError(null)
    setProgressShown(false)
    setStatus(STATUS_LINES[0].text)
    setPhase('pressed')
    later(() => {
      shownAt = Date.now()
      setProgressShown(true)
      setPhase('pending')
    }, TIMING.showPendingAfter)
    for (const line of STATUS_LINES.slice(1)) later(() => setStatus(line.text), line.after)

    /** Run `finish` once the busy state, if it was shown, has been visible long enough. */
    const settle = (finish: () => void) => {
      if (current.current !== thisRun) return // cancelled or unmounted meanwhile
      clearTimers(thisRun)
      const hold = shownAt === null ? 0 : Math.max(0, TIMING.pendingMinVisible - (Date.now() - shownAt))
      if (hold === 0) finish()
      else later(finish, hold)
    }

    analyze(input, thisRun.controller.signal).then(
      (report) =>
        settle(() => {
          remember(report, input)
          const open = () => {
            current.current = null
            setPhase('idle')
            navigate(`/check/${report.analysis_id}`)
          }
          const leave = prefersReducedMotion() ? 0 : TIMING.leave
          if (leave === 0) return open()
          setPhase('leaving')
          later(open, leave)
        }),
      (cause: unknown) => {
        if ((cause as { name?: string } | null)?.name === 'AbortError') return
        settle(() => {
          current.current = null
          setProgressShown(false)
          setError(
            cause instanceof ApiError
              ? cause
              : new ApiError('Something went wrong on this page. Nothing was lost: try again.', 'server'),
          )
          setPhase('error')
        })
      },
    )
  }, [])

  const cancel = useCallback(() => {
    const running = current.current
    if (!running) return
    clearTimers(running)
    current.current = null
    running.controller.abort()
    setProgressShown(false)
    setPhase('idle')
  }, [])

  return { phase, status, progressShown, error, run, cancel }
}
