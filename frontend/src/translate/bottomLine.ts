// Tier 1, part two: the bottom line. One sentence on what the evidence did, then the main caveat.
// Every sentence is written here from the report's structured fields. The backend's own explanation
// is about how the rating was calculated; it is shown, as written, only under technical details.

import type { ClaimReport, Report } from '../api/types'
import { contextFindings, countClaims, recordPhrase, sourceStanding } from './cards'
import { claimStatus, isLooseMatch } from './claims'
import { plural, ratingSays } from './format'
import type { ClaimStatus, VerdictKey } from './types'

const NEXT_STEP =
  'Try adding the source link or the image, or look it up with a fact-checker such as PIB Fact Check, BOOM or Alt News.'

/**
 * A fact-checker to name: one whose own rating says the same thing as the verdict on that claim.
 * "A matching claim", because the fact-checker reviewed its own wording of the claim, not the reader's.
 */
function agreeingFactCheck(claims: ClaimReport[], status: ClaimStatus): string | null {
  const says = status === 'contradicted' ? 'false' : 'true'
  for (const claim of claims) {
    if (claimStatus(claim) !== status) continue
    const match = (claim.fact_checks ?? []).find((f) => f?.publisher && ratingSays(f.rating) === says)
    if (match) return `${match.publisher} rated a matching claim “${match.rating}”.`
  }
  return null
}

/** Why nothing could be checked: the backend's reason, which is already plain English. */
function unverifiedLines(report: Report): string[] {
  const reasons = report.reasons ?? []
  const reason = reasons.find((r) => r.startsWith('Not scored:'))
  if (!reason) return ['We could not find evidence about this, so we are not giving a verdict.', NEXT_STEP]

  const first =
    'We could not check this: ' +
    reason
      .slice('Not scored:'.length)
      .replace(/\s*A number here would only be a guess\.?\s*$/, '')
      .trim()
  if (/only a source was given/.test(reason)) return [first] // it already says what to add
  if (reasons.some((r) => r.startsWith('No trusted outlet reported this'))) {
    return [
      first,
      'No trusted news outlet we searched has reported it either, which is a reason for caution but not proof.',
    ]
  }
  return [first, NEXT_STEP]
}

export function bottomLine(report: Report, verdict: VerdictKey): string[] {
  if (verdict === 'unverified') return unverifiedLines(report)
  if (verdict === 'not_checked') return ['We could not run any check on this, so there is no verdict.']

  const claims = (report.claims ?? []).filter((c) => c?.text)
  const checkedClaims = claims.filter((c) => claimStatus(c) !== 'vague' && c.verdict != null)
  const n = countClaims(report)
  const context = contextFindings(report)
  const mismatches = context.filter((f) => f.severity === 'bad')
  const silence = Boolean(report.coverage?.applied)
  const one = n.checked === 1

  // ---- Sentence one: what the evidence did ----
  let first: string
  let usedMismatch = false
  let usedSilence = false
  if (n.contradicted > 0) {
    first = one
      ? 'Independent sources contradict this claim.'
      : `Independent sources contradict ${n.contradicted} of the ${n.checked} claims we checked${n.backed ? ` and back ${n.backed}` : ''}.`
    const checker = agreeingFactCheck(checkedClaims, 'contradicted')
    if (checker) first += ` ${checker}`
  } else if (silence) {
    first = 'No trusted news outlet has reported this, although a claim like it would normally be widely covered.'
    usedSilence = true
  } else if (n.backed > 0) {
    first = one
      ? 'Independent sources back this claim.'
      : n.backed === n.checked
        ? `Independent sources back all ${n.checked} claims we checked.`
        : `Independent sources back ${n.backed} of the ${n.checked} claims we checked.`
    const checker = agreeingFactCheck(checkedClaims, 'supported')
    if (checker) first += ` ${checker}`
  } else if (mismatches.length > 0) {
    first = mismatches[0].text
    usedMismatch = true
  } else if (n.mixed > 0) {
    first = one
      ? 'The sources we found disagree about this claim.'
      : `The sources we found disagree about ${n.mixed} of the ${n.checked} claims we checked.`
  } else if (n.checked > 0) {
    first = `We found no independent source that settles ${one ? 'this claim' : 'these claims'} either way.`
  } else {
    first = 'We found nothing in this that could be checked against evidence.'
  }

  // ---- Sentence two: the main caveat not already said ----
  const second = (): string | null => {
    const nextMismatch = mismatches[usedMismatch ? 1 : 0]
    if (nextMismatch) return nextMismatch.text
    if (usedSilence) return 'Silence is not proof: something from the last few hours may not have been reported yet.'

    const settled = n.backed + n.contradicted
    const left = n.checked - settled
    if (settled > 0 && left > 0) {
      return `${left} other ${plural(left, 'claim', 'claims')} could not be confirmed either way.`
    }
    if (claims.some(isLooseMatch)) {
      return 'The closest sources we found only loosely match what is claimed, so read them before relying on this.'
    }
    const standing = sourceStanding(report)
    if (standing) {
      const from = standing.domain ?? 'an outlet'
      return standing.record === 'satire'
        ? `The link is from ${from}, a known satire site.`
        : `The link is from ${from}, which ${recordPhrase(standing)}.`
    }
    const reassuring = context.find((f) => f.severity === 'good')
    if (reassuring) return reassuring.text
    if (n.vague > 0) {
      return 'Some sentences were too vague to check: they do not say who or what they are about.'
    }
    return null
  }

  const caveat = second()
  return caveat ? [first, caveat] : [first]
}
