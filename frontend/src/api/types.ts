// The API contract, mirrored from api/schemas.py. Only `band` is required on a report: analyses are
// stored for good, so a report written by an older model version may lack any of the newer fields.

export interface Evidence {
  source?: string | null
  url?: string | null
  snippet: string
  weight?: number | null
  date?: string | null
  /** false: the passage was found but did not count towards the verdict */
  counted?: boolean | null
  /** set when the passage is a published verdict on a different claim */
  claim_match?: number | null
}

export interface FactCheckBadge {
  publisher: string
  rating: string
  claim_reviewed?: string | null
  url?: string | null
  date?: string | null
  relevance?: number | null
}

export interface NewsSearch {
  ok?: boolean | null
  from?: string | null
  to?: string | null
  results?: number | null
  trusted_reports?: number | null
  outlets?: number | null
}

export interface ClaimCoverage {
  expected?: boolean | null
  event?: string | null
  why?: string | null
  searched?: boolean | null
  news_state?: string | null
  search?: NewsSearch | null
  confirmed?: boolean | null
}

export type ClaimVerdict =
  | 'Supported'
  | 'Refuted'
  | 'Not Enough Evidence'
  | 'Conflicting Evidence/Cherrypicking'

export interface ClaimReport {
  text: string
  checkworthiness?: number | null
  verdict?: ClaimVerdict | string | null
  credibility?: number | null
  confidence?: number | null
  evidence_relevance?: number | null
  evidence_off_topic?: boolean | null
  evidence?: Evidence[] | null
  fact_checks?: FactCheckBadge[] | null
  justification?: string | null
  context?: string | null
  coverage?: ClaimCoverage | null
}

export interface Contribution {
  check: string
  points: number
  applicable: boolean
}

export interface Explanation {
  summary: string
  explanation: string
  source?: 'llm' | 'template' | string | null
  model?: string | null
}

export interface Media {
  text_image_match?: number | null
  text_image_subtype?: string | null
  reuse_found?: boolean | null
  earliest_match_date?: string | null
  location_match?: number | null
  location_checked?: string | null
  location_mismatch?: boolean | null
}

export interface Channel {
  score?: number | null
  confidence?: number | null
  subtype?: string | null
  flags?: string[] | null
}

/** Set only when the silence of trusted outlets was used as evidence. */
export interface Absence {
  applied?: boolean | null
  claim?: string | null
  why?: string | null
  search?: NewsSearch | null
}

export interface FetchedArticle {
  url?: string | null
  title?: string | null
  description?: string | null
  published?: string | null
  error?: string | null
}

export type Band = 'Low' | 'Questionable' | 'Mostly credible' | 'High' | 'Unverified' | 'Unknown'

export interface Report {
  analysis_id: string
  band: Band | string
  score?: number | null
  abstained?: boolean | null
  claims?: ClaimReport[] | null
  media?: Media | null
  channels?: Record<string, Channel> | null
  baseline_score?: number | null
  contributions?: Contribution[] | null
  explanation?: Explanation | null
  skipped_checks?: string[] | null
  reasons?: string[] | null
  coverage?: Absence | null
  meta_model?: string | null
  model_version?: string | null
  disclaimer?: string | null
  created_at?: string | null
  cached?: boolean | null
  fetched_article?: FetchedArticle | null
}

export type CorrectLabel = 'credible' | 'not_credible' | 'mixed' | 'unsure'

export interface FeedbackBody {
  analysis_id: string
  agrees: boolean
  correct_label?: CorrectLabel
  comment?: string
}

export interface Health {
  status: string
  model_version?: string
  explainer?: string | null
}

export interface AnalyzeInput {
  text?: string
  url?: string
  image?: File
  claimedDate?: string
  claimedLocation?: string
  publishDate?: string
  refresh?: boolean
}
