/**
 * The outline of a result, shown where the result will land while a slow check runs. It holds still
 * on purpose: the spinner beside the status line is the one thing that moves.
 */
export function ResultSkeleton() {
  const bar = 'rounded bg-line/60'
  return (
    <div aria-hidden="true" className="mt-8">
      <div className={`${bar} h-9 w-64 max-w-full`} />
      <div className={`${bar} mt-5 h-4 w-full`} />
      <div className={`${bar} mt-3 h-4 w-4/5`} />
      <div className="mt-8 grid gap-4 sm:grid-cols-3">
        <div className={`${bar} h-24`} />
        <div className={`${bar} h-24`} />
        <div className={`${bar} h-24`} />
      </div>
      <div className={`${bar} mt-8 h-28 w-full`} />
    </div>
  )
}
