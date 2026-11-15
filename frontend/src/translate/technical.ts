// The model's own account of the result, for the collapsed "technical details" section only.
// Nothing here is reworded: these are the numbers and strings the backend produced.

import type { Report } from '../api/types'
import type { TechnicalView } from './types'

const CHECK_LABELS: Record<string, string> = {
  claim_verification: 'Claim verification',
  text_image_match: 'Caption and image match',
  media_reuse: 'Image reuse',
  source: 'Source rating',
  temporal: 'Date consistency',
  input_type: 'Type of input',
  missing_coverage: 'Not reported by trusted news',
  not_applicable: 'Checks that did not run',
  calibration: 'Calibration',
}

export function technicalView(report: Report): TechnicalView {
  const explanation = report.explanation
  return {
    score: report.score ?? null,
    band: report.band,
    baseline: report.baseline_score ?? null,
    contributions: (report.contributions ?? [])
      .filter((c) => Math.abs(c.points) >= 0.05)
      .map((c) => ({ label: CHECK_LABELS[c.check] ?? c.check, points: c.points })),
    claims: (report.claims ?? []).map((c) => ({
      text: c.text,
      verdict: c.verdict ?? null,
      confidence: c.confidence ?? null,
      relevance: c.evidence_relevance ?? null,
    })),
    explanation: explanation?.summary
      ? {
          summary: explanation.summary,
          text: explanation.explanation ?? '',
          writtenBy: explanation.source === 'llm' ? 'model' : 'template',
        }
      : null,
    skipped: report.skipped_checks ?? [],
    findings: report.reasons ?? [],
    modelVersion: report.model_version ?? null,
    analysisId: report.analysis_id,
  }
}
