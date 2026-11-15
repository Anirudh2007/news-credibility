// Tier 1, part one: the editorial verdict. It comes from the rating band the model gave, plus what
// drove a low rating, so a reader sees "Contradicted by evidence" or "Out of context" rather than a
// number.

import type { Report } from '../api/types'
import { contextFindings, countClaims } from './cards'
import type { Tone, VerdictKey } from './types'

export const VERDICTS: Record<VerdictKey, { label: string; tone: Tone }> = {
  well_corroborated: { label: 'Well corroborated', tone: 'good' },
  mostly_credible: { label: 'Mostly credible', tone: 'ok' },
  needs_context: { label: 'Needs context', tone: 'warn' },
  contradicted: { label: 'Contradicted by evidence', tone: 'bad' },
  no_trusted_reporting: { label: 'No trusted reporting found', tone: 'bad' },
  out_of_context: { label: 'Out of context', tone: 'bad' },
  low_credibility: { label: 'Low credibility', tone: 'bad' },
  unverified: { label: 'Unverified', tone: 'neutral' },
  not_checked: { label: 'Could not be checked', tone: 'neutral' },
}

/** Something about the image or the dates does not fit the story. */
export function hasContextMismatch(report: Report): boolean {
  return contextFindings(report).some((f) => f.severity === 'bad')
}

function lowBecause(report: Report): VerdictKey {
  if (countClaims(report).contradicted > 0) return 'contradicted'
  if (report.coverage?.applied) return 'no_trusted_reporting'
  if (hasContextMismatch(report)) return 'out_of_context'
  return 'low_credibility'
}

export function verdictKey(report: Report): VerdictKey {
  if (report.abstained || report.band === 'Unverified') return 'unverified'
  switch (report.band) {
    case 'High':
      return 'well_corroborated'
    case 'Mostly credible':
      return 'mostly_credible'
    case 'Questionable':
      return 'needs_context'
    case 'Low':
      return lowBecause(report)
    default:
      return 'not_checked'
  }
}

export function verdictFor(report: Report): { key: VerdictKey; label: string; tone: Tone } {
  const key = verdictKey(report)
  return { key, ...VERDICTS[key] }
}
