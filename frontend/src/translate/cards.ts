// Tier 2: three editorial checks, each in plain words. A check that did not run says what was
// missing and stays neutral: an absent image or link is never a mark against the story.

import type { Report } from '../api/types'
import { claimStatus } from './claims'
import { formatDate, list, plural } from './format'
import type { ReasonCard, Tone } from './types'

// ---- Source corroboration --------------------------------------------------------------------

export interface ClaimCounts {
  /** claims that could be checked (sentences too vague to check are counted apart) */
  checked: number
  backed: number
  contradicted: number
  mixed: number
  open: number
  vague: number
}

export function countClaims(report: Report): ClaimCounts {
  const counts: ClaimCounts = { checked: 0, backed: 0, contradicted: 0, mixed: 0, open: 0, vague: 0 }
  for (const claim of report.claims ?? []) {
    if (!claim?.text) continue
    const status = claimStatus(claim)
    if (status === 'vague') {
      counts.vague += 1
      continue
    }
    if (claim.verdict == null) continue // found, but the evidence search never ran for it
    counts.checked += 1
    if (status === 'supported') counts.backed += 1
    else if (status === 'contradicted') counts.contradicted += 1
    else if (status === 'mixed') counts.mixed += 1
    else counts.open += 1
  }
  return counts
}

function skipped(report: Report, check: string): string | null {
  const entry = (report.skipped_checks ?? []).find((s) => s.startsWith(`${check}:`))
  return entry ? entry.slice(check.length + 1).trim() : null
}

function notChecked(key: ReasonCard['key'], title: string, headline: string): ReasonCard {
  return { key, title, state: 'not_checked', tone: 'neutral', headline, detail: [] }
}

function corroborationCard(report: Report): ReasonCard {
  const title = 'Source corroboration'
  const n = countClaims(report)

  if (n.checked === 0) {
    const why = skipped(report, 'claim_verification') ?? ''
    if (n.vague > 0 || why.includes('who or what')) {
      return notChecked('corroboration', title, 'Not checked: the text does not say who or what it is about.')
    }
    if (why === 'no text') return notChecked('corroboration', title, 'Not checked: no text was given.')
    if (why.includes('not available')) {
      return notChecked('corroboration', title, 'Not checked: the evidence search was not available.')
    }
    return notChecked(
      'corroboration',
      title,
      'Not checked: we found no statement of fact to check. Questions, opinions and jokes are not checked.',
    )
  }

  let headline: string
  if (n.checked === 1) {
    headline = n.backed
      ? 'The claim is backed by independent sources.'
      : n.contradicted
        ? 'The claim is contradicted by independent sources.'
        : n.mixed
          ? 'Sources disagree about the claim.'
          : 'No independent source settles the claim.'
  } else {
    headline = `${n.backed} of ${n.checked} claims are backed by independent sources.`
  }

  const detail: string[] = []
  if (n.checked > 1) {
    if (n.contradicted) detail.push(`${n.contradicted} contradicted.`)
    if (n.mixed) detail.push(`${n.mixed} with mixed evidence.`)
    if (n.open) detail.push(`${n.open} could not be confirmed either way.`)
  }
  if (n.vague) detail.push(`${n.vague} too vague to check.`)

  const checkers = [
    ...new Set((report.claims ?? []).flatMap((c) => (c.fact_checks ?? []).map((f) => f.publisher).filter(Boolean))),
  ]
  if (checkers.length) {
    detail.push(`Published ${plural(checkers.length, 'fact-check', 'fact-checks')} found: ${list(checkers)}.`)
  }

  const tone: Tone = n.contradicted
    ? 'bad'
    : n.backed === n.checked
      ? 'good'
      : n.backed
        ? 'ok'
        : n.mixed
          ? 'warn'
          : 'neutral'
  return { key: 'corroboration', title, state: 'checked', tone, headline, detail }
}

// ---- Context and visuals ----------------------------------------------------------------------

export interface ContextFinding {
  text: string
  severity: 'bad' | 'warn' | 'good'
  /** One more sentence that explains the finding; shown in the card, not in the bottom line. */
  more?: string
}

// How the caption-and-image check names the kind of mismatch it sees, in a reader's words.
const MISMATCH_KINDS: Record<string, string> = {
  false_connection: 'It looks like a real image attached to a story it does not belong to.',
  misleading_content: 'It looks like a real image used to give a misleading impression.',
  manipulated_content: 'It looks as if the image may have been edited.',
  imposter_content: 'It looks like an image made to pass as coming from a real source.',
  satire: 'It looks like satire rather than a news photograph.',
}

const SEVERITY_ORDER = { bad: 0, warn: 1, good: 2 }

/** Everything the image and date checks found, worst first. Empty when neither check ran. */
export function contextFindings(report: Report): ContextFinding[] {
  const media = report.media ?? {}
  const temporal = report.channels?.temporal
  const flags = new Set(temporal?.flags ?? [])
  const found: ContextFinding[] = []

  if (media.text_image_subtype) {
    found.push({
      text: 'The image does not appear to match what the text says about it.',
      severity: 'bad',
      more: MISMATCH_KINDS[media.text_image_subtype],
    })
  } else if (media.text_image_match != null) {
    found.push({ text: 'The image fits what the text says about it.', severity: 'good' })
  }

  const firstSeen = formatDate(media.earliest_match_date)
  if (media.reuse_found && flags.has('media_predates_claim')) {
    found.push({
      text: firstSeen
        ? `An earlier copy of this image was already online on ${firstSeen}, before the date the story gives.`
        : 'An earlier copy of this image was already online before the date the story gives.',
      severity: 'bad',
    })
  } else if (media.reuse_found) {
    found.push({
      text: firstSeen
        ? `An earlier copy of this image was already online on ${firstSeen}.`
        : 'An earlier copy of this image is already online.',
      severity: 'warn',
    })
  } else if (flags.has('media_predates_claim')) {
    found.push({ text: 'The image was online before the date the story gives.', severity: 'bad' })
  } else if (media.reuse_found === false) {
    found.push({ text: 'We found no earlier copy of this image in our reference collection.', severity: 'good' })
  }

  if (media.location_mismatch) {
    found.push({
      text: media.location_checked
        ? `The image does not resemble reference photos of ${media.location_checked}.`
        : 'The image does not resemble reference photos of the place the story names.',
      severity: 'bad',
    })
  }
  if (flags.has('exif_predates_claim')) {
    found.push({ text: 'The photo’s own camera data dates it before the date the story gives.', severity: 'bad' })
  }
  if (flags.has('published_before_event')) {
    found.push({
      text: 'The article was published before the date of the event it reports, which only makes sense for a scheduled event.',
      severity: 'warn',
    })
  }
  if (flags.has('evidence_much_older')) {
    found.push({
      text: 'The reporting we found is much older than the story’s date, so this may be old news resurfacing.',
      severity: 'warn',
    })
  }
  if (temporal && flags.size === 0) {
    found.push({ text: 'The dates in the story line up.', severity: 'good' })
  }

  return found.sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity])
}

function contextCard(report: Report): ReasonCard {
  const title = 'Context and visuals'
  const findings = contextFindings(report)
  if (findings.length === 0) return notChecked('context', title, 'Not checked: no image or date was given.')

  const [first, ...rest] = findings
  const detail = [...(first.more ? [first.more] : []), ...rest.flatMap((f) => (f.more ? [f.text, f.more] : [f.text]))]
  return {
    key: 'context',
    title,
    state: 'checked',
    tone: first.severity === 'bad' ? 'bad' : first.severity === 'warn' ? 'warn' : 'good',
    headline: first.text,
    detail,
  }
}

// ---- Source standing --------------------------------------------------------------------------

export interface SourceStanding {
  domain: string | null
  record: 'reliable' | 'mixed' | 'poor' | 'satire'
}

// Where the outlet's rating (0 to 1, from public media-rating lists) falls. The reader sees the
// words, never the number.
const RELIABLE_FROM = 0.75
const MIXED_FROM = 0.45

function hostOf(url: string | null | undefined): string | null {
  try {
    return url ? new URL(url).hostname.replace(/^www\./i, '') : null
  } catch {
    return null
  }
}

/** The outlet's record, when the link was to a rated outlet; otherwise null. */
export function sourceStanding(report: Report): SourceStanding | null {
  const rating = report.channels?.source?.score
  if (rating == null) return null
  const reasons = report.reasons ?? []
  const named = reasons.map((r) => /^Source reliability rating [\d.]+ \(([^)]+)\)/.exec(r)?.[1]).find(Boolean)
  const domain = named ?? hostOf(report.fetched_article?.url)
  if (reasons.some((r) => /known satire site/i.test(r))) return { domain, record: 'satire' }
  return { domain, record: rating >= RELIABLE_FROM ? 'reliable' : rating >= MIXED_FROM ? 'mixed' : 'poor' }
}

const RECORD_WORDS: Record<SourceStanding['record'], { phrase: string; tone: Tone; detail: string }> = {
  reliable: {
    phrase: 'has a reliable record',
    tone: 'good',
    detail: 'Public media-rating lists rate it as generally accurate.',
  },
  mixed: {
    phrase: 'has a mixed record',
    tone: 'warn',
    detail: 'Public media-rating lists show a mix of accurate and inaccurate reporting.',
  },
  poor: {
    phrase: 'has a poor record',
    tone: 'bad',
    detail: 'Public media-rating lists rate it as often inaccurate.',
  },
  satire: {
    phrase: 'is a known satire site',
    tone: 'warn',
    detail: 'Satire is written to entertain, not to report.',
  },
}

export function recordPhrase(standing: SourceStanding): string {
  return RECORD_WORDS[standing.record].phrase
}

function sourceCard(report: Report): ReasonCard {
  const title = 'Source standing'
  const standing = sourceStanding(report)
  if (standing) {
    const words = RECORD_WORDS[standing.record]
    return {
      key: 'source',
      title,
      state: 'checked',
      tone: words.tone,
      headline: `${standing.domain ?? 'This outlet'} ${words.phrase}.`,
      detail: [words.detail, 'This describes the outlet in general, not this story.'],
    }
  }

  const why = skipped(report, 'source') ?? ''
  const platform = /^(\S+) is a platform or link shortener/.exec(why)
  if (platform) return notChecked('source', title, `Not rated: ${platform[1]} is a platform, not the author.`)
  const unlisted = /^(\S+) is not in the ratings list/.exec(why)
  if (unlisted) {
    return notChecked('source', title, `Not rated: ${unlisted[1]} is not among the outlets we have ratings for.`)
  }
  return notChecked('source', title, 'Not checked: no link was given.')
}

export function reasonCards(report: Report): ReasonCard[] {
  return [corroborationCard(report), contextCard(report), sourceCard(report)]
}
