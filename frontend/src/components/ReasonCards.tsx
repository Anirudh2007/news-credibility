import { TIMING } from '../lib/timing'
import type { ReasonCard } from '../translate/types'
import { ToneIcon } from './Icons'
import { revealDelay } from './Verdict'

/** Tier 2: three editorial checks side by side. A check that did not run is shown, quietly. */
export function ReasonCards({ cards }: { cards: ReasonCard[] }) {
  return (
    <section aria-labelledby="reasons-heading" className="mt-12">
      <h2 id="reasons-heading" className="reveal font-display text-2xl font-semibold text-ink" style={revealDelay(TIMING.reveal.cards)}>
        Why we say this
      </h2>
      <ul className="mt-4 grid gap-4 sm:grid-cols-3">
        {cards.map((card, i) => {
          const checked = card.state === 'checked'
          return (
            <li
              key={card.key}
              style={revealDelay(TIMING.reveal.cards + i * TIMING.reveal.cardStep)}
              className={`reveal tone-${card.tone} rounded-xl border p-4 ${checked ? 'border-line bg-surface' : 'border-dashed border-line'}`}
            >
              <h3 className="flex items-center gap-2 text-[13px] font-semibold uppercase tracking-[0.08em] text-muted">
                {checked && <ToneIcon tone={card.tone} size={16} className="shrink-0 text-tone" />}
                {card.title}
              </h3>
              <p className={`mt-2 text-[15px] leading-6 ${checked ? 'font-semibold text-ink' : 'text-muted'}`}>
                {card.headline}
              </p>
              {card.detail.length > 0 && (
                <ul className="mt-2 space-y-1 text-sm leading-5 text-muted">
                  {card.detail.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              )}
            </li>
          )
        })}
      </ul>
    </section>
  )
}
