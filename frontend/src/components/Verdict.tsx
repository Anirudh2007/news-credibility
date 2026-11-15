import { useEffect, useRef, type CSSProperties } from 'react'
import { TIMING } from '../lib/timing'
import type { VerdictView } from '../translate/types'
import { ToneIcon } from './Icons'

/** The CSS variable the reveal animation reads its delay from. */
export function revealDelay(ms: number): CSSProperties {
  return { '--reveal-delay': `${ms}ms` } as CSSProperties
}

// The four rating bands, in the proportions the model uses (30, 30, 20, 20 of 100).
const BANDS = [
  { name: 'Low', upTo: 30, share: 30, colour: 'bg-rust' },
  { name: 'Questionable', upTo: 60, share: 30, colour: 'bg-amber' },
  { name: 'Mostly credible', upTo: 80, share: 20, colour: 'bg-sky' },
  { name: 'High', upTo: 100, share: 20, colour: 'bg-accent' },
]

/**
 * A quiet secondary indicator: where the rating falls among the four bands. No digits; the band
 * the marker sits in is also named in words, so it does not depend on colour or position alone.
 */
function BandStrip({ position }: { position: number }) {
  const active = BANDS.find((b) => position <= b.upTo) ?? BANDS[BANDS.length - 1]
  return (
    <div role="img" aria-label={`Rated in the ${active.name.toLowerCase()} range.`} className="mt-6 max-w-[480px]">
      <div className="relative">
        <div className="flex gap-1">
          {BANDS.map((band) => (
            <span
              key={band.name}
              style={{ flexGrow: band.share }}
              className={`h-1.5 basis-0 rounded-full ${band.colour} ${band === active ? '' : 'opacity-35'}`}
            />
          ))}
        </div>
        <span
          style={{ left: `${position}%` }}
          className="absolute -top-1.5 h-[18px] w-1 -translate-x-1/2 rounded-full bg-ink ring-2 ring-paper"
        />
      </div>
      <div aria-hidden="true" className="mt-2 flex gap-1 text-[12px] leading-4">
        {BANDS.map((band) => (
          <span
            key={band.name}
            style={{ flexGrow: band.share }}
            className={`basis-0 ${band === active ? 'font-semibold text-ink' : 'text-muted'}`}
          >
            {band.name}
          </span>
        ))}
      </div>
    </div>
  )
}

/** Tier 1: the verdict in words, then the bottom line. */
export function Verdict({ verdict }: { verdict: VerdictView }) {
  const heading = useRef<HTMLHeadingElement>(null)

  // The page changed without a reload: move focus to the verdict so keyboard and screen-reader
  // users land on the answer, not on the top of the document.
  useEffect(() => {
    heading.current?.focus({ preventScroll: true })
  }, [])

  return (
    <header className={`tone-${verdict.tone}`}>
      <p className="reveal text-[13px] font-semibold uppercase tracking-[0.12em] text-muted" style={revealDelay(TIMING.reveal.verdict)}>
        Verdict
      </p>
      <h1
        ref={heading}
        tabIndex={-1}
        className="reveal mt-2 flex items-start gap-3 font-display text-[36px] font-semibold leading-[1.1] tracking-[-0.01em] text-ink focus:outline-none sm:text-[48px]"
        style={revealDelay(TIMING.reveal.verdict)}
      >
        <ToneIcon tone={verdict.tone} size={36} className="mt-1 shrink-0 text-tone sm:mt-1.5" />
        {verdict.label}
      </h1>
      <div className="reveal mt-5 space-y-2 text-[19px] leading-8 text-body" style={revealDelay(TIMING.reveal.bottomLine)}>
        {verdict.bottomLine.map((sentence) => (
          <p key={sentence}>{sentence}</p>
        ))}
      </div>
      {verdict.bandPosition !== null && (
        <div className="reveal" style={revealDelay(TIMING.reveal.band)}>
          <BandStrip position={verdict.bandPosition} />
        </div>
      )}
    </header>
  )
}
