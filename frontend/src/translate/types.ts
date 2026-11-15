// What the result screen renders. Everything here is written for a reader: no model numbers except
// inside `technical`, which stays collapsed until someone asks for it.

export type Tone = 'good' | 'ok' | 'warn' | 'bad' | 'neutral'

export type VerdictKey =
  | 'well_corroborated'
  | 'mostly_credible'
  | 'needs_context'
  | 'contradicted'
  | 'no_trusted_reporting'
  | 'out_of_context'
  | 'low_credibility'
  | 'unverified'
  | 'not_checked'

export interface VerdictView {
  key: VerdictKey
  label: string
  tone: Tone
  /** One or two plain sentences: what the evidence did, then the main caveat. */
  bottomLine: string[]
  /** Where the marker sits on the band strip (0 to 100); null when there is no rating to place. */
  bandPosition: number | null
}

export interface ReasonCard {
  key: 'corroboration' | 'context' | 'source'
  title: string
  state: 'checked' | 'not_checked'
  tone: Tone
  headline: string
  detail: string[]
}

export type ClaimStatus = 'supported' | 'contradicted' | 'mixed' | 'unverified' | 'vague'

export interface SourceView {
  publication: string
  url: string | null
  date: string | null
  quote: string
  /** Set when the evidence is a question-and-answer passage: the question it answers. */
  question: string | null
  counted: boolean
  uncountedReason: string | null
}

export interface FactCheckView {
  publisher: string
  rating: string
  /** the claim the fact-checker reviewed, in their words; it may differ from the claim on this card */
  claimReviewed: string | null
  url: string | null
  date: string | null
}

export interface ClaimView {
  text: string
  status: ClaimStatus
  statusLabel: string
  tone: Tone
  notes: string[]
  lead: SourceView | null
  factChecks: FactCheckView[]
  more: SourceView[]
}

export interface TechnicalView {
  score: number | null
  band: string
  baseline: number | null
  contributions: { label: string; points: number }[]
  claims: { text: string; verdict: string | null; confidence: number | null; relevance: number | null }[]
  explanation: { summary: string; text: string; writtenBy: 'model' | 'template' } | null
  skipped: string[]
  findings: string[]
  modelVersion: string | null
  analysisId: string
}

export interface ResultView {
  id: string
  verdict: VerdictView
  /** Always three, in this order: corroboration, context, source. */
  cards: ReasonCard[]
  claims: ClaimView[]
  /** What was read from a link, when the reader gave a link and no text. */
  checked: { headline: string | null; published: string | null; note: string | null }
  meta: { checkedAt: string | null; fromCache: boolean }
  technical: TechnicalView
  disclaimer: string
}
