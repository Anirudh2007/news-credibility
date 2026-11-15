import { TIMING, claimDelay } from '../lib/timing'
import type { ResultView } from '../translate/types'
import { ClaimCard } from './ClaimCard'
import { Feedback } from './Feedback'
import { ReasonCards } from './ReasonCards'
import { TechnicalDetails } from './TechnicalDetails'
import { Verdict, revealDelay } from './Verdict'

/**
 * One result, top to bottom: the verdict and bottom line, why we say it, the claims with their
 * sources, then feedback and the optional technical details. Each block carries the delay it
 * takes in the reveal storyboard (src/lib/timing.ts).
 */
export function Result({ view }: { view: ResultView }) {
  return (
    <article>
      <Verdict verdict={view.verdict} />

      <ReasonCards cards={view.cards} />

      {view.claims.length > 0 && (
        <section aria-labelledby="claims-heading" className="mt-12">
          <h2
            id="claims-heading"
            className="reveal font-display text-2xl font-semibold text-ink"
            style={revealDelay(TIMING.reveal.claims)}
          >
            Claim by claim
          </h2>
          <p className="reveal mt-1 text-[15px] leading-6 text-muted" style={revealDelay(TIMING.reveal.claims)}>
            Each statement of fact we found, with the reporting it was checked against. Open the sources and read
            them yourself.
          </p>
          <ul className="mt-4 space-y-4">
            {view.claims.map((claim, i) => (
              <li key={`${i}-${claim.text}`} className="reveal" style={revealDelay(claimDelay(i))}>
                <ClaimCard claim={claim} />
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="reveal" style={revealDelay(TIMING.reveal.rest)}>
        <Feedback analysisId={view.id} />
        <p className="mt-8 text-sm leading-6 text-muted">{view.disclaimer}</p>
        <TechnicalDetails technical={view.technical} />
      </div>
    </article>
  )
}
