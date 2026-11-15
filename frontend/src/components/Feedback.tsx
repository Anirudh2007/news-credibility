import { useId, useState } from 'react'
import { sendFeedback } from '../api/client'
import type { CorrectLabel } from '../api/types'
import { Expander } from './Expander'
import { Check } from './Icons'

const OPTIONS: { value: CorrectLabel; label: string }[] = [
  { value: 'credible', label: 'Credible' },
  { value: 'not_credible', label: 'Not credible' },
  { value: 'mixed', label: 'Mixed or partly true' },
  { value: 'unsure', label: 'Not sure' },
]

// Which checks this visitor has already answered, so the question is not asked twice in a visit.
const SENT_KEY = 'newscred.feedback.v1'

function alreadySent(id: string): boolean {
  try {
    return (JSON.parse(window.sessionStorage.getItem(SENT_KEY) ?? '[]') as string[]).includes(id)
  } catch {
    return false
  }
}

function markSent(id: string): void {
  try {
    const sent = JSON.parse(window.sessionStorage.getItem(SENT_KEY) ?? '[]') as string[]
    window.sessionStorage.setItem(SENT_KEY, JSON.stringify([...new Set([...sent, id])]))
  } catch {
    // storage refused: the worst case is that the question shows again
  }
}

const choice =
  'press hover-fade h-11 min-w-16 cursor-pointer rounded-lg border border-line bg-surface px-4 text-sm font-semibold text-ink hover:border-accent'

/**
 * "Was this verdict right?" A Yes is saved at once. A No opens the follow-up, because only then is
 * there something more to ask. The answer shows as saved immediately and is taken back, with the
 * reason, if the save fails.
 */
export function Feedback({ analysisId }: { analysisId: string }) {
  const [state, setState] = useState<'asking' | 'disagreeing' | 'sent'>(() =>
    alreadySent(analysisId) ? 'sent' : 'asking',
  )
  const [label, setLabel] = useState<CorrectLabel | null>(null)
  const [comment, setComment] = useState('')
  const [problem, setProblem] = useState(false)
  const followUpId = useId()

  function send(agrees: boolean) {
    const before = state
    setProblem(false)
    setState('sent')
    const text = comment.trim()
    sendFeedback({
      analysis_id: analysisId,
      agrees,
      ...(agrees || !label ? {} : { correct_label: label }),
      ...(agrees || !text ? {} : { comment: text }),
    }).then(
      () => markSent(analysisId),
      () => {
        setState(before)
        setProblem(true)
      },
    )
  }

  if (state === 'sent') {
    return (
      <p role="status" className="mt-12 flex items-center gap-2 text-[15px] leading-6 text-ink">
        <Check size={18} className="shrink-0 text-accent" />
        Thanks. Your answer was saved.
      </p>
    )
  }

  return (
    <section aria-labelledby="feedback-heading" className="mt-12">
      <div className="flex flex-wrap items-center gap-3">
        <h2 id="feedback-heading" className="mr-1 text-[15px] font-semibold text-ink">
          Was this verdict right?
        </h2>
        <button type="button" className={choice} onClick={() => send(true)}>
          Yes
        </button>
        <button
          type="button"
          className={choice}
          aria-expanded={state === 'disagreeing'}
          aria-controls={followUpId}
          onClick={() => setState('disagreeing')}
        >
          No
        </button>
      </div>

      {problem && (
        <p role="alert" className="mt-3 text-sm leading-6 text-rust">
          We could not save your answer. Please try again.
        </p>
      )}

      <Expander id={followUpId} open={state === 'disagreeing'}>
        <div className="pt-4">
          <fieldset>
            <legend className="text-sm font-medium text-ink">What do you think is right?</legend>
            <div className="mt-2 flex flex-wrap gap-2">
              {OPTIONS.map((option) => (
                <label
                  key={option.value}
                  className="hover-fade flex h-11 cursor-pointer items-center gap-2 rounded-lg border border-line bg-surface px-3 text-sm text-ink has-[:checked]:border-accent has-[:checked]:font-semibold"
                >
                  <input
                    type="radio"
                    name={`${followUpId}-label`}
                    value={option.value}
                    checked={label === option.value}
                    onChange={() => setLabel(option.value)}
                    className="accent-[var(--accent)]"
                  />
                  {option.label}
                </label>
              ))}
            </div>
          </fieldset>
          <label className="mt-4 block">
            <span className="text-sm font-medium text-ink">Anything to add? (optional)</span>
            <textarea
              value={comment}
              maxLength={1000}
              rows={3}
              onChange={(e) => setComment(e.target.value)}
              className="mt-1 block w-full rounded-lg border border-line bg-surface px-3 py-2 text-[15px] leading-6 text-body"
            />
          </label>
          <button
            type="button"
            onClick={() => send(false)}
            className="press mt-3 h-11 cursor-pointer rounded-lg bg-accent px-5 text-sm font-semibold text-on-accent"
          >
            Send
          </button>
        </div>
      </Expander>
    </section>
  )
}
