// Tier 3: each claim the system found, what the evidence says about it, and the sources to read.

import type { ClaimReport, Evidence, Report } from '../api/types'
import { endQuote, formatDate, publication, safeUrl, splitQuote } from './format'
import type { ClaimStatus, ClaimView, SourceView, Tone } from './types'

/** Below this the best passage found is about a different claim (the re-ranker's own boundary). */
const LOOSE_MATCH = 0.5
const QUOTE_LIMIT = 320
const REVIEWED_LIMIT = 160

const STATUS: Record<ClaimStatus, { label: string; tone: Tone }> = {
  supported: { label: 'Supported', tone: 'good' },
  contradicted: { label: 'Contradicted', tone: 'bad' },
  mixed: { label: 'Mixed evidence', tone: 'warn' },
  unverified: { label: 'Unverified', tone: 'neutral' },
  vague: { label: 'Too vague to check', tone: 'neutral' },
}

export const isVague = (c: ClaimReport) => c.context === 'missing'

/** The verifier reached a verdict, but on a passage that only loosely matches the claim. */
export function isLooseMatch(c: ClaimReport): boolean {
  const decided = c.verdict === 'Supported' || c.verdict === 'Refuted' || c.verdict?.startsWith('Conflicting')
  return Boolean(decided) && !c.evidence_off_topic && c.evidence_relevance != null && c.evidence_relevance < LOOSE_MATCH
}

/**
 * What the reader is told about one claim. A verdict stands only when the evidence is about this
 * claim: evidence about something else, or only loosely related, leaves the claim unverified.
 */
export function claimStatus(c: ClaimReport): ClaimStatus {
  if (isVague(c)) return 'vague'
  if (c.evidence_off_topic || isLooseMatch(c)) return 'unverified'
  if (c.verdict === 'Supported') return 'supported'
  if (c.verdict === 'Refuted') return 'contradicted'
  if (c.verdict?.startsWith('Conflicting')) return 'mixed'
  return 'unverified'
}

function clip(text: string, limit = QUOTE_LIMIT): string {
  return text.length > limit ? `${text.slice(0, limit).trimEnd()}…` : text
}

function sourceView(e: Evidence, uncountedReason: string | null): SourceView {
  const { question, quote } = splitQuote(e.snippet ?? '')
  return {
    publication: publication(e.source, e.url),
    url: safeUrl(e.url),
    date: formatDate(e.date),
    question,
    quote: endQuote(clip(quote)),
    counted: uncountedReason === null,
    uncountedReason,
  }
}

function whyNotCounted(c: ClaimReport, e: Evidence): string | null {
  if (c.evidence_off_topic) return 'About a different claim: not counted.'
  if (isLooseMatch(c)) return 'Only loosely matches this claim: not counted.'
  if (e.counted === false) {
    return e.claim_match != null
      ? 'A verdict on a different claim: not counted.'
      : 'Does not confirm the claim: not counted.'
  }
  return null
}

function newsNote(c: ClaimReport): string | null {
  const coverage = c.coverage
  if (!coverage?.expected || !coverage.searched) return null
  const found = coverage.search?.trusted_reports ?? 0
  const reports = found === 0 ? 'no report' : found === 1 ? 'one report' : `${found} reports`
  const confirms = coverage.confirmed && found > 0 ? ', and one confirms it' : ''
  return `A claim like this would normally be in the news. We searched trusted outlets and found ${reports}${confirms}.`
}

function notes(c: ClaimReport, status: ClaimStatus): string[] {
  const out: string[] = []
  if (status === 'vague') {
    out.push('This sentence does not say who or what it is about, so it could not be checked. Add names, places or dates.')
    return out
  }
  if (isLooseMatch(c)) {
    const leaning =
      c.verdict === 'Supported' ? 'backing' : c.verdict === 'Refuted' ? 'contradicting' : 'both backing and contradicting'
    out.push(
      `The closest sources lean towards ${leaning} this claim, but they only loosely match it, so they were not counted.`,
    )
  } else if (c.evidence_off_topic) {
    out.push('The closest sources we found are about other claims, so they were not counted.')
  }
  const news = newsNote(c)
  if (news) out.push(news)
  if (status === 'unverified' && out.length === 0) {
    out.push('We found nothing published that confirms or contradicts this.')
  }
  return out
}

function claimView(c: ClaimReport): ClaimView {
  const status = claimStatus(c)
  const sources =
    status === 'vague' ? [] : (c.evidence ?? []).filter((e) => e?.snippet).map((e) => sourceView(e, whyNotCounted(c, e)))
  // The lead quote is the passage the verdict rests on, so it must be one that counted.
  const lead = sources[0]?.counted ? sources[0] : null
  return {
    text: c.text,
    status,
    statusLabel: STATUS[status].label,
    tone: STATUS[status].tone,
    notes: notes(c, status),
    lead,
    factChecks: (c.fact_checks ?? [])
      .filter((f) => f?.publisher && f?.rating)
      .slice(0, 2)
      .map((f) => ({
        publisher: f.publisher,
        rating: f.rating,
        claimReviewed: f.claim_reviewed?.trim() ? clip(f.claim_reviewed.trim(), REVIEWED_LIMIT) : null,
        url: safeUrl(f.url),
        date: formatDate(f.date),
      })),
    more: lead ? sources.slice(1) : sources,
  }
}

export function claimViews(report: Report): ClaimView[] {
  return (report.claims ?? []).filter((c) => c?.text).map(claimView)
}
