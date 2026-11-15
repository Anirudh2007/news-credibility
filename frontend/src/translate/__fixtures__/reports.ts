// Reports shaped like real API answers, one per situation the result screen has to explain.
// The backend strings in them (explanations, justifications, reasons) deliberately carry score
// arithmetic, so the tests can prove none of it reaches the reader.

import type { ClaimReport, Report } from '../../api/types'

const TEXT_ONLY_SKIPS = [
  'text_image_match: no image',
  'media_reuse: no image',
  'source: no source URL or domain given',
  'temporal: no claimed date found',
]

const JARGON_EXPLANATION = {
  summary: 'The evidence contradicts the claim (fact-checked by BOOM: False).',
  explanation: 'Scored 12/100 (Low). Lowered by claim verification (-38). Claims: 1 Refuted.',
  source: 'template',
}

function claim(overrides: Partial<ClaimReport> & { text: string }): ClaimReport {
  return {
    checkworthiness: 0.9,
    verdict: 'Not Enough Evidence',
    credibility: null,
    confidence: null,
    evidence_relevance: null,
    evidence_off_topic: false,
    evidence: [],
    fact_checks: [],
    justification: null,
    context: null,
    coverage: null,
    ...overrides,
  }
}

const rbiRefuted = claim({
  text: 'Reserve Bank of India announces new Rs 1000 currency notes.',
  verdict: 'Refuted',
  credibility: 0.04,
  confidence: 0.91,
  evidence_relevance: 0.93,
  evidence: [
    {
      source: 'pib.gov.in',
      url: 'https://pib.gov.in/factcheck/rs-1000-notes',
      snippet: 'Q: Has the RBI announced new Rs 1000 notes? A: No. The RBI has made no such announcement.',
      weight: 0.8,
      date: '2024-03-12',
      counted: true,
    },
    {
      source: 'www.boomlive.in',
      url: 'https://www.boomlive.in/fact-check/rs-1000-note',
      snippet: 'The viral message about a new Rs 1000 note is false.',
      weight: 0.2,
      date: '2024-03-14',
      counted: true,
    },
  ],
  fact_checks: [
    {
      publisher: 'BOOM',
      rating: 'False',
      claim_reviewed: 'RBI to issue new Rs 1000 notes',
      url: 'https://www.boomlive.in/fact-check/rs-1000-note',
      date: '2024-03-14',
      relevance: 0.9,
    },
  ],
  justification: 'The evidence found contradicts this claim (91% confidence).',
})

export const refuted: Report = {
  analysis_id: 'aaa111aaa111',
  score: 12,
  band: 'Low',
  abstained: false,
  claims: [rbiRefuted],
  media: {},
  channels: { claims: { score: 0.04 } },
  baseline_score: 50,
  contributions: [
    { check: 'claim_verification', points: -38, applicable: true },
    { check: 'calibration', points: 0.4, applicable: true },
  ],
  explanation: JARGON_EXPLANATION,
  skipped_checks: TEXT_ONLY_SKIPS,
  reasons: [],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
  disclaimer: 'This score is an aid, not a final judgement. Read the evidence.',
  created_at: '2026-11-12T10:00:00+00:00',
  cached: false,
}

const chandrayaan = claim({
  text: 'ISRO successfully launches Chandrayaan-3 lunar mission.',
  verdict: 'Supported',
  credibility: 0.95,
  confidence: 0.88,
  evidence_relevance: 0.97,
  evidence: [
    {
      source: 'thehindu.com',
      url: 'https://www.thehindu.com/sci-tech/chandrayaan-3-launch',
      snippet: 'ISRO launched Chandrayaan-3 from Sriharikota on Friday afternoon.',
      weight: 1,
      date: '2023-07-14',
      counted: true,
    },
  ],
  coverage: {
    expected: true,
    event: 'launch',
    why: 'a space launch by a national agency',
    searched: true,
    news_state: 'ran',
    search: { ok: true, from: '2023-07-01', to: '2023-07-20', results: 9, trusted_reports: 4, outlets: 312 },
    confirmed: true,
  },
  justification: 'The evidence found supports this claim (88% confidence).',
})

export const supported: Report = {
  analysis_id: 'bbb222bbb222',
  score: 88,
  band: 'High',
  abstained: false,
  claims: [chandrayaan],
  media: {},
  channels: { claims: { score: 0.95 }, source: { score: 0.9, confidence: 0.8 } },
  baseline_score: 55,
  contributions: [
    { check: 'claim_verification', points: 27.5, applicable: true },
    { check: 'source', points: 5.5, applicable: true },
  ],
  explanation: {
    summary: 'The evidence supports the claim.',
    explanation: 'Scored 88/100 (High). Raised by claim verification (+28), the source rating (+6).',
    source: 'template',
  },
  skipped_checks: ['text_image_match: no image', 'media_reuse: no image', 'temporal: no claimed date found'],
  reasons: ['Source reliability rating 0.90 (thehindu.com).'],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
  created_at: '2026-11-12T10:05:00+00:00',
  cached: true,
}

export const mostlyCredible: Report = { ...supported, analysis_id: 'bbb333bbb333', score: 72, band: 'Mostly credible' }

export const mixedThree: Report = {
  analysis_id: 'ccc333ccc333',
  score: 45,
  band: 'Questionable',
  abstained: false,
  claims: [
    chandrayaan,
    rbiRefuted,
    claim({ text: 'The mission cost less than a Hollywood film.', justification: 'Not enough evidence was found.' }),
  ],
  media: {},
  channels: { claims: { score: 0.5 } },
  baseline_score: 50,
  contributions: [{ check: 'claim_verification', points: -5, applicable: true }],
  explanation: JARGON_EXPLANATION,
  skipped_checks: TEXT_ONLY_SKIPS,
  reasons: [],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
}

export const conflicting: Report = {
  ...mixedThree,
  analysis_id: 'ccc444ccc444',
  claims: [
    claim({
      text: 'Fuel prices fell in every state last month.',
      verdict: 'Conflicting Evidence/Cherrypicking',
      confidence: 0.7,
      evidence_relevance: 0.8,
      evidence: [{ source: 'reuters.com', url: 'https://www.reuters.com/x', snippet: 'Prices fell in some states.', counted: true }],
    }),
    claim({ text: 'They said it was true.', verdict: null, context: 'missing' }),
  ],
}

const SILENCE_REASON =
  'No trusted outlet reported this. A claim like this (a nationwide policy change) would be widely reported if it ' +
  'were true, but a search of 312 trusted news sites (from 2026-10-01 to 2026-11-12) found no report about it. ' +
  'Silence counts as evidence here because reports would be expected; it is not proof (an event from the last few ' +
  'hours may not be indexed yet).'

export const silence: Report = {
  analysis_id: 'ddd444ddd444',
  score: 14,
  band: 'Low',
  abstained: false,
  claims: [
    claim({
      text: 'WhatsApp to start charging users a subscription fee from next month.',
      evidence_off_topic: true,
      evidence_relevance: 0.004,
      evidence: [{ source: 'snopes.com', url: 'https://www.snopes.com/x', snippet: 'An unrelated rumour about Facebook.', counted: false }],
      coverage: {
        expected: true,
        event: 'policy',
        why: 'a nationwide policy change',
        searched: true,
        news_state: 'ran',
        search: { ok: true, from: '2026-10-01', to: '2026-11-12', results: 3, trusted_reports: 0, outlets: 312 },
        confirmed: false,
      },
    }),
  ],
  media: {},
  channels: {},
  baseline_score: 50,
  contributions: [{ check: 'missing_coverage', points: -36, applicable: true }],
  coverage: {
    applied: true,
    claim: 'WhatsApp to start charging users a subscription fee from next month.',
    why: 'a nationwide policy change',
    search: { ok: true, from: '2026-10-01', to: '2026-11-12', trusted_reports: 0, outlets: 312 },
  },
  explanation: {
    summary: 'No evidence settled the claim. Mostly lowered by no trusted news report of a claim that would be widely reported if true.',
    explanation: 'Scored 14/100 (Low). Lowered by no trusted news report (-36).',
    source: 'template',
  },
  skipped_checks: TEXT_ONLY_SKIPS,
  reasons: [SILENCE_REASON],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
}

export const unverified: Report = {
  analysis_id: 'eee555eee555',
  score: null,
  band: 'Unverified',
  abstained: true,
  claims: [
    claim({
      text: 'A man in Pune taught his parrot to file tax returns.',
      evidence_off_topic: true,
      evidence_relevance: 0.002,
    }),
  ],
  media: {},
  channels: {},
  baseline_score: null,
  contributions: [],
  explanation: {
    summary: 'Not enough evidence to score this item.',
    explanation: 'Not scored: no evidence about its claims was found. Add a source URL or an image.',
    source: 'template',
  },
  skipped_checks: TEXT_ONLY_SKIPS,
  reasons: [
    'Not scored: no evidence about its claims was found, and no image or source URL was given. A number here would only be a guess.',
  ],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
}

export const imageMismatch: Report = {
  analysis_id: 'fff666fff666',
  score: 18,
  band: 'Low',
  abstained: false,
  claims: [claim({ text: 'Flooding at India Gate this morning.' })],
  media: {
    text_image_match: 0.21,
    text_image_subtype: 'false_connection',
    reuse_found: true,
    earliest_match_date: '2019-08-03',
    location_match: 0.2,
    location_checked: 'India Gate',
    location_mismatch: true,
  },
  channels: {
    text_image: { score: 0.21, confidence: 0.7, subtype: 'false_connection' },
    reuse: { score: 0.1, confidence: 0.9 },
    temporal: { score: 0.4, confidence: 0.75, flags: ['media_predates_claim'] },
  },
  baseline_score: 48,
  contributions: [
    { check: 'media_reuse', points: -17, applicable: true },
    { check: 'text_image_match', points: -9, applicable: true },
    { check: 'temporal', points: -4, applicable: true },
  ],
  explanation: {
    summary: 'No evidence settled the claim. Mostly lowered by the media reuse check.',
    explanation: 'Scored 18/100 (Low). Lowered by the media reuse check (-17), the caption-image match (-9).',
    source: 'template',
  },
  skipped_checks: ['source: no source URL or domain given'],
  reasons: [
    "Caption and image look inconsistent (likely 'false connection').",
    'Earliest copy found 1920 days before the claimed date (2019-08-03).',
    'Image does not resemble reference photos of India Gate.',
    'The image/video appeared online 1920 days before the claimed date.',
  ],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
}

export const imageFine: Report = {
  ...imageMismatch,
  analysis_id: 'fff777fff777',
  score: 70,
  band: 'Mostly credible',
  media: { text_image_match: 0.86, text_image_subtype: null, reuse_found: false, location_mismatch: false },
  channels: { text_image: { score: 0.86 }, reuse: { score: 0.8 }, temporal: { score: 1, flags: [] } },
  reasons: [],
}

export const linkOnlyArticle: Report = {
  ...mostlyCredible,
  analysis_id: 'ggg777ggg777',
  fetched_article: {
    url: 'https://www.thehindu.com/sci-tech/chandrayaan-3-launch',
    title: 'Chandrayaan-3 lifts off from Sriharikota',
    description: 'India’s third lunar mission began on Friday.',
    published: '2023-07-14',
    error: null,
  },
}

export const linkNotRead: Report = {
  analysis_id: 'ggg888ggg888',
  score: null,
  band: 'Unverified',
  abstained: true,
  claims: [],
  channels: { source: { score: 0.9 } },
  skipped_checks: ['claim_verification: no text', 'text_image_match: no image', 'media_reuse: no image'],
  reasons: [
    'Not scored: only a source was given. Its rating says how reliable the outlet is in general, not whether this item is true; add the text or image to check it.',
    'Source reliability rating 0.90 (thehindu.com).',
    'The article itself was not checked: the page could not be loaded. Only the source was used.',
  ],
  fetched_article: { url: 'https://www.thehindu.com/x', title: null, description: null, published: null, error: 'the page could not be loaded' },
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
}

export const satireSource: Report = {
  analysis_id: 'hhh888hhh888',
  score: 20,
  band: 'Low',
  abstained: false,
  claims: [claim({ text: 'Parliament votes to rename Monday.' })],
  media: {},
  channels: { source: { score: 0.1, confidence: 0.8 } },
  baseline_score: 50,
  contributions: [{ check: 'source', points: -30, applicable: true }],
  explanation: JARGON_EXPLANATION,
  skipped_checks: ['text_image_match: no image', 'media_reuse: no image', 'temporal: no claimed date found'],
  reasons: ['Source is a known satire site.', 'Source reliability rating 0.10 (theonion.com).'],
  meta_model: 'lightgbm+isotonic',
  model_version: '0.10.1+ab12cd',
}

export const platformSource: Report = {
  ...refuted,
  analysis_id: 'iii999iii999',
  skipped_checks: [
    'text_image_match: no image',
    'media_reuse: no image',
    'source: x.com is a platform or link shortener, not the author: the source is not rated.',
    'temporal: no claimed date found',
  ],
}

export const unratedSource: Report = {
  ...refuted,
  analysis_id: 'jjj000jjj000',
  skipped_checks: ['source: example.org is not in the ratings list: the source is not rated.'],
}

/** Weak evidence: the verifier leaned towards "Refuted", but the passage is about something else. */
export const weakEvidence: Report = {
  ...refuted,
  analysis_id: 'kkk111kkk111',
  score: 27,
  claims: [{ ...rbiRefuted, evidence_relevance: 0.22, fact_checks: [] }],
}

/** A report stored by an early model version: most fields did not exist yet. */
export const oldStored: Report = {
  analysis_id: 'lll222lll222',
  band: 'Questionable',
  score: 41,
  claims: [],
  meta_model: 'equal-weight average (meta-learner not trained)',
  model_version: '0.4.0',
}

export const unknownBand: Report = { analysis_id: 'mmm333mmm333', band: 'Unknown', score: null }

export const unsafeLink: Report = {
  ...refuted,
  analysis_id: 'nnn444nnn444',
  claims: [
    {
      ...rbiRefuted,
      evidence: [{ source: 'evil.example', url: 'javascript:alert(1)', snippet: 'Click me.', counted: true }],
      fact_checks: [{ publisher: 'Fake Checker', rating: 'False', url: 'data:text/html,<script>1</script>' }],
    },
  ],
}

export const ALL: Record<string, Report> = {
  refuted,
  supported,
  mostlyCredible,
  mixedThree,
  conflicting,
  silence,
  unverified,
  imageMismatch,
  imageFine,
  linkOnlyArticle,
  linkNotRead,
  satireSource,
  platformSource,
  unratedSource,
  weakEvidence,
  oldStored,
  unknownBand,
  unsafeLink,
}
