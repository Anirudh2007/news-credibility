import { Desk } from '../components/Desk'
import { Guide } from '../components/Guide'
import { Spinner } from '../components/Icons'
import { ResultSkeleton } from '../components/ResultSkeleton'
import { useAnalyze } from '../lib/useAnalyze'

/** The home page: the desk, and under it either the three-step guide or a check in progress. */
export function DeskPage() {
  const { phase, status, progressShown, error, run, cancel } = useAnalyze()

  return (
    <main
      id="main"
      className={`mx-auto w-full max-w-[720px] px-4 pb-24 pt-10 sm:px-6 sm:pt-16 ${phase === 'leaving' ? 'leaving' : ''}`}
    >
      <h1 className="font-display text-[40px] font-semibold leading-[1.1] tracking-[-0.01em] text-ink sm:text-[56px]">
        Check a story before you share it.
      </h1>
      <p className="mt-4 max-w-[36rem] text-lg leading-7 text-muted">
        Paste a link, a claim or an image. We check what it says against published fact-checks and trusted
        reporting, then show you the evidence.
      </p>

      <Desk phase={phase} progressShown={progressShown} error={error} onSubmit={run} />

      {progressShown ? (
        <section aria-label="Check in progress" className="mt-8">
          <div className="flex items-center gap-3">
            <p role="status" className="flex min-w-0 items-center gap-3 text-[15px] leading-6 text-ink">
              <Spinner size={18} className="shrink-0 text-accent" />
              <span>{status}</span>
            </p>
            <button
              type="button"
              onClick={cancel}
              className="hover-fade ml-auto h-11 shrink-0 cursor-pointer rounded-lg px-3 text-sm font-medium text-muted hover:bg-surface hover:text-ink"
            >
              Cancel
            </button>
          </div>
          <ResultSkeleton />
        </section>
      ) : (
        <Guide />
      )}
    </main>
  )
}
