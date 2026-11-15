import { useEffect, useMemo, useState } from 'react'
import { ApiError, getAnalysis } from '../api/client'
import type { AnalyzeInput, Report } from '../api/types'
import { clearDraft } from '../components/Desk'
import { ArrowLeft, Spinner } from '../components/Icons'
import { Result } from '../components/Result'
import { ResultSkeleton } from '../components/ResultSkeleton'
import * as recent from '../lib/recent'
import { Link } from '../lib/router'
import * as session from '../lib/session'
import { TIMING } from '../lib/timing'
import { useAnalyze } from '../lib/useAnalyze'
import { translateReport, type ResultView } from '../translate'
import { NotFound } from './NotFound'

const shell = 'mx-auto w-full max-w-[720px] px-4 pb-24 pt-6 sm:px-6 sm:pt-8'

/** What the reader submitted, as far as this browser knows; otherwise the closest stand-in. */
function askedAbout(id: string, input: AnalyzeInput | undefined, view: ResultView): string | null {
  const typed = input?.text ?? input?.url ?? (input?.image ? 'An image' : null)
  return typed ?? recent.find(id)?.preview ?? view.checked.headline ?? view.claims[0]?.text ?? null
}

function BackToDesk() {
  return (
    <Link
      href="/"
      onClick={clearDraft}
      className="hover-fade -ml-2 inline-flex h-11 items-center gap-2 rounded-lg px-2 text-sm font-medium text-muted hover:bg-surface hover:text-ink"
    >
      <ArrowLeft size={16} />
      Check something else
    </Link>
  )
}

/** One check, at its own address: /check/<id>. */
export function CheckPage({ id }: { id: string }) {
  const stored = session.get(id)
  // The reveal is for a result the reader just asked for. Reading the flag is pure; the effect
  // below uses it up, so opening the same result again shows it at rest.
  const [fresh] = useState(() => stored?.fresh ?? false)
  useEffect(() => {
    session.takeFresh(id)
  }, [id])

  const [fetched, setFetched] = useState<Report | null>(null)
  const [failure, setFailure] = useState<ApiError | null>(null)
  const [attempt, setAttempt] = useState(0)
  const [slow, setSlow] = useState(false)
  const report = stored?.report ?? fetched

  useEffect(() => {
    if (stored) return
    let live = true
    setFailure(null)
    setSlow(false)
    // Same rule as the desk: say "loading" only if it is not instant.
    const timer = window.setTimeout(() => live && setSlow(true), TIMING.showPendingAfter)
    getAnalysis(id).then(
      (r) => live && setFetched(r),
      (e: unknown) =>
        live && setFailure(e instanceof ApiError ? e : new ApiError('This check could not be opened.', 'server')),
    )
    return () => {
      live = false
      window.clearTimeout(timer)
    }
  }, [id, attempt, stored])

  const view = useMemo(() => (report ? translateReport(report) : null), [report])
  const recheck = useAnalyze()

  useEffect(() => {
    if (view) document.title = `${view.verdict.label} · News Credibility`
  }, [view])

  if (failure?.kind === 'not_found') return <NotFound what="check" />

  if (!view) {
    return (
      <main id="main" data-reveal="off" className={shell}>
        <BackToDesk />
        {failure ? (
          <div className="mt-8">
            <p role="alert" className="text-[17px] leading-7 text-ink">
              {failure.message}
            </p>
            <button
              type="button"
              onClick={() => setAttempt((n) => n + 1)}
              className="press mt-4 h-11 cursor-pointer rounded-lg bg-accent px-5 text-[15px] font-semibold text-on-accent"
            >
              Try again
            </button>
          </div>
        ) : (
          slow && (
            <>
              <p role="status" className="mt-8 flex items-center gap-3 text-[15px] leading-6 text-ink">
                <Spinner size={18} className="shrink-0 text-accent" />
                Opening this check.
              </p>
              <ResultSkeleton />
            </>
          )
        )}
      </main>
    )
  }

  const preview = askedAbout(id, stored?.input, view)
  const rechecking = recheck.phase === 'pressed' || recheck.phase === 'pending' || recheck.phase === 'leaving'

  return (
    <main
      id="main"
      data-reveal={fresh ? 'on' : 'off'}
      className={`${shell} ${recheck.phase === 'leaving' ? 'leaving' : ''}`}
    >
      <BackToDesk />

      <section aria-label="What was checked" className="mb-10 mt-4 border-b border-line pb-6">
        {preview && (
          <>
            <p className="text-[13px] font-semibold uppercase tracking-[0.12em] text-muted">You asked about</p>
            <div className="mt-2 flex items-start gap-3">
              {stored?.imageUrl && (
                <img src={stored.imageUrl} alt="The image you submitted" className="size-14 shrink-0 rounded-lg object-cover" />
              )}
              <p className="line-clamp-3 min-w-0 text-[17px] leading-7 text-ink">{preview}</p>
            </div>
          </>
        )}
        {view.checked.headline && (
          <p className="mt-2 text-sm leading-6 text-muted">
            We read “{view.checked.headline}”{view.checked.published ? `, published ${view.checked.published}` : ''}.
          </p>
        )}
        {view.checked.note && <p className="mt-2 text-sm leading-6 text-muted">{view.checked.note}</p>}
        {view.meta.fromCache && view.meta.checkedAt && (
          <p className="mt-2 flex flex-wrap items-center gap-x-2 text-sm leading-6 text-muted">
            Checked on {view.meta.checkedAt}.
            {stored?.input && (
              <button
                type="button"
                disabled={rechecking}
                onClick={() => recheck.run({ ...stored.input, refresh: true })}
                className="hover-fade inline-flex h-11 cursor-pointer items-center gap-2 rounded-lg px-2 font-medium text-accent underline decoration-1 underline-offset-[3px] hover:bg-surface disabled:cursor-default disabled:no-underline"
              >
                {rechecking && <Spinner size={16} />}
                {rechecking ? 'Checking again…' : 'Check again'}
              </button>
            )}
          </p>
        )}
        {recheck.phase === 'error' && recheck.error && (
          <p role="alert" className="mt-2 text-sm leading-6 text-rust">
            {recheck.error.message}
          </p>
        )}
      </section>

      <Result view={view} />
    </main>
  )
}
