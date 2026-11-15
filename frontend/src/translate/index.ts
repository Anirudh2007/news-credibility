// The human explanation layer: an API report in, a reader-facing result out.
//
//   tier 1  verdict + bottom line       verdict.ts, bottomLine.ts
//   tier 2  three reasoning cards       cards.ts
//   tier 3  claim-by-claim evidence     claims.ts
//   last    technical details           technical.ts (the model's own numbers, collapsed by default)
//
// Pure functions with no React in them, so every sentence a reader can see is covered by a unit test.

import type { Report } from '../api/types'
import { bottomLine } from './bottomLine'
import { reasonCards } from './cards'
import { claimViews } from './claims'
import { formatDate } from './format'
import { technicalView } from './technical'
import type { ResultView } from './types'
import { verdictFor } from './verdict'

export type * from './types'

const DISCLAIMER = 'This is an aid for reading the evidence, not a final judgement.'

function whatWasRead(report: Report): ResultView['checked'] {
  const article = report.fetched_article
  if (!article) return { headline: null, published: null, note: null }
  if (article.error) {
    return {
      headline: null,
      published: null,
      note: `The article itself was not checked: ${article.error}. Paste the headline or the text to check what it says.`,
    }
  }
  return { headline: article.title?.trim() || null, published: formatDate(article.published), note: null }
}

export function translateReport(report: Report): ResultView {
  const verdict = verdictFor(report)
  return {
    id: report.analysis_id,
    verdict: {
      ...verdict,
      bottomLine: bottomLine(report, verdict.key),
      bandPosition: report.abstained || report.score == null ? null : Math.min(100, Math.max(0, report.score)),
    },
    cards: reasonCards(report),
    claims: claimViews(report),
    checked: whatWasRead(report),
    meta: { checkedAt: formatDate(report.created_at), fromCache: Boolean(report.cached) },
    technical: technicalView(report),
    disclaimer: DISCLAIMER,
  }
}
