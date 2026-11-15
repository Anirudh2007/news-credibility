import { useId, useState, type ReactNode } from 'react'
import type { ClaimStatus, ClaimView, SourceView } from '../translate/types'
import { Expander } from './Expander'
import { CheckCircle, ChevronDown, ExternalLink, HelpCircle, Opposed, XCircle } from './Icons'

const STATUS_ICON: Record<ClaimStatus, (p: { size?: number; className?: string }) => ReactNode> = {
  supported: CheckCircle,
  contradicted: XCircle,
  mixed: Opposed,
  unverified: HelpCircle,
  vague: HelpCircle,
}

/** A link out to a source. Opens in a new tab so the reader does not lose the result. */
function Outbound({ href, children }: { href: string | null; children: ReactNode }) {
  if (!href) return <span className="font-medium text-ink">{children}</span>
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      // the padding and matching negative margin make the tap area 44px tall without moving the text
      className="hover-fade -my-2.5 inline-flex items-center gap-1 py-2.5 font-medium text-accent underline decoration-1 underline-offset-[3px] hover:decoration-2"
    >
      {children}
      <ExternalLink size={14} className="shrink-0" />
      <span className="sr-only">(opens in a new tab)</span>
    </a>
  )
}

function Source({ source }: { source: SourceView }) {
  return (
    <figure className="border-l-2 border-line pl-4">
      {source.question && <p className="text-sm leading-5 text-muted">Asked: {source.question}</p>}
      <blockquote className="font-display text-[18px] leading-7 text-body">{source.quote}</blockquote>
      <figcaption className="mt-1 flex flex-wrap items-center gap-x-2 text-sm leading-6 text-muted">
        <Outbound href={source.url}>{source.publication}</Outbound>
        {source.date && <span>· {source.date}</span>}
        {source.uncountedReason && <span>· {source.uncountedReason}</span>}
      </figcaption>
    </figure>
  )
}

/** Tier 3: one claim, what the evidence says about it, and the sources to read. */
export function ClaimCard({ claim }: { claim: ClaimView }) {
  const [open, setOpen] = useState(false)
  const moreId = useId()
  const StatusIcon = STATUS_ICON[claim.status]
  const extra = claim.more.length

  return (
    <article className={`tone-${claim.tone} rounded-xl border border-line bg-surface p-5`}>
      <p className="inline-flex h-7 items-center gap-1.5 rounded-full border border-tone px-2.5 text-[13px] font-semibold text-ink">
        <StatusIcon size={15} className="shrink-0 text-tone" />
        {claim.statusLabel}
      </p>
      <h3 className="mt-3 text-[17px] font-semibold leading-7 text-ink">{claim.text}</h3>

      {claim.lead && (
        <div className="mt-4">
          <Source source={claim.lead} />
        </div>
      )}

      {claim.factChecks.length > 0 && (
        <ul className="mt-3 space-y-1 text-sm leading-6 text-muted">
          {claim.factChecks.map((check) => (
            <li key={`${check.publisher}-${check.rating}`}>
              <Outbound href={check.url}>{check.publisher}</Outbound>{' '}
              {check.claimReviewed
                ? `checked the claim “${check.claimReviewed}” and rated it “${check.rating}”`
                : `rated a matching claim “${check.rating}”`}
              {check.date ? ` on ${check.date}` : ''}.
            </li>
          ))}
        </ul>
      )}

      {claim.notes.map((note) => (
        <p key={note} className="mt-3 text-sm leading-6 text-muted">
          {note}
        </p>
      ))}

      {extra > 0 && (
        <>
          <button
            type="button"
            aria-expanded={open}
            aria-controls={moreId}
            onClick={() => setOpen(!open)}
            className="hover-fade -ml-2 mt-2 inline-flex h-11 cursor-pointer items-center gap-1.5 rounded-lg px-2 text-sm font-medium text-accent hover:bg-paper"
          >
            {open ? 'Hide' : 'Show'} {extra} {claim.lead ? 'more ' : ''}
            {extra === 1 ? 'source' : 'sources'}
            <ChevronDown
              size={16}
              className={`transition-transform duration-[var(--dur-expand)] ease-[var(--ease-in-out)] ${open ? 'rotate-180' : ''}`}
            />
          </button>
          <Expander id={moreId} open={open}>
            <div className="space-y-4 pb-1 pt-2">
              {claim.more.map((source) => (
                <Source key={`${source.publication}-${source.quote}`} source={source} />
              ))}
            </div>
          </Expander>
        </>
      )}
    </article>
  )
}
