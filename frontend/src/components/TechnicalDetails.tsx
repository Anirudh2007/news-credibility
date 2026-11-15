import { useState } from 'react'
import type { TechnicalView } from '../translate/types'
import { Expander } from './Expander'
import { ChevronDown } from './Icons'

const signed = (points: number) => `${points < 0 ? '−' : '+'}${Math.abs(points).toFixed(1)}`
const percent = (value: number | null) => (value === null ? '–' : `${Math.round(value * 100)}%`)

const th = 'whitespace-nowrap border-b border-line py-2 pr-4 text-left text-[13px] font-semibold text-muted'
const td = 'border-b border-line py-2 pr-4 align-top'

function Body({ t }: { t: TechnicalView }) {
  return (
    <div className="space-y-6 pb-2 pt-4 text-sm leading-6 text-body">
      <p className="text-muted">
        These are the model’s own numbers, as produced. The verdict above is a plain-language reading of them.
      </p>

      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1">
        <dt className="text-muted">Rating</dt>
        <dd className="tabular-nums">{t.score === null ? 'None given' : `${t.score} out of 100`}</dd>
        <dt className="text-muted">Band</dt>
        <dd>{t.band}</dd>
        {t.baseline !== null && (
          <>
            <dt className="text-muted">Starting point</dt>
            <dd className="tabular-nums">
              {t.baseline.toFixed(1)} (an average item with the same kind of input)
            </dd>
          </>
        )}
      </dl>

      {t.contributions.length > 0 && (
        <div>
          <h3 className="font-semibold text-ink">What moved the rating</h3>
          <table className="mt-1 w-full border-collapse">
            <thead>
              <tr>
                <th scope="col" className={th}>Check</th>
                <th scope="col" className={`${th} text-right`}>Points</th>
              </tr>
            </thead>
            <tbody>
              {t.contributions.map((c) => (
                <tr key={c.label}>
                  <td className={td}>{c.label}</td>
                  <td className={`${td} text-right tabular-nums`}>{signed(c.points)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[13px] leading-5 text-muted">
            The starting point plus these points gives the rating. Checks that did not run add nothing.
          </p>
        </div>
      )}

      {t.claims.length > 0 && (
        <div>
          <h3 className="font-semibold text-ink">Claims as the model read them</h3>
          <div className="overflow-x-auto">
            <table className="mt-1 w-full border-collapse">
              <thead>
                <tr>
                  <th scope="col" className={th}>Claim</th>
                  <th scope="col" className={th}>Model verdict</th>
                  <th scope="col" className={`${th} text-right`}>Confidence</th>
                  <th scope="col" className={`${th} text-right`}>Evidence match</th>
                </tr>
              </thead>
              <tbody>
                {t.claims.map((c) => (
                  <tr key={c.text}>
                    <td className={`${td} min-w-[12rem]`}>{c.text}</td>
                    <td className={td}>{c.verdict ?? 'Not verified'}</td>
                    <td className={`${td} text-right tabular-nums`}>{percent(c.confidence)}</td>
                    <td className={`${td} text-right tabular-nums`}>{percent(c.relevance)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {t.explanation && (
        <div>
          <h3 className="font-semibold text-ink">The system’s own explanation</h3>
          <p className="mt-1">{t.explanation.summary}</p>
          {t.explanation.text && <p className="mt-1">{t.explanation.text}</p>}
          <p className="mt-1 text-[13px] leading-5 text-muted">
            {t.explanation.writtenBy === 'model'
              ? 'Written by a language model and checked against the evidence before it was accepted.'
              : 'Generated from fixed templates.'}
          </p>
        </div>
      )}

      {t.findings.length > 0 && (
        <div>
          <h3 className="font-semibold text-ink">Findings as logged</h3>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {t.findings.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      {t.skipped.length > 0 && (
        <div>
          <h3 className="font-semibold text-ink">Checks that did not run</h3>
          <ul className="mt-1 list-disc space-y-1 pl-5">
            {t.skipped.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-[13px] leading-5 text-muted">
        Check {t.analysisId}
        {t.modelVersion ? ` · model version ${t.modelVersion}` : ''}
      </p>
    </div>
  )
}

/** The model's numbers, for anyone who wants them. Closed by default and not built until opened. */
export function TechnicalDetails({ technical }: { technical: TechnicalView }) {
  const [open, setOpen] = useState(false)
  return (
    <section className="mt-8 border-t border-line pt-2">
      <button
        type="button"
        aria-expanded={open}
        aria-controls="technical-details"
        onClick={() => setOpen(!open)}
        className="hover-fade -ml-2 inline-flex h-11 cursor-pointer items-center gap-2 rounded-lg px-2 text-sm font-medium text-muted hover:bg-surface hover:text-ink"
      >
        <ChevronDown
          size={16}
          className={`transition-transform duration-[var(--dur-expand)] ease-[var(--ease-in-out)] ${open ? 'rotate-180' : ''}`}
        />
        View technical verification details
      </button>
      <Expander id="technical-details" open={open} lazy>
        <Body t={technical} />
      </Expander>
    </section>
  )
}
