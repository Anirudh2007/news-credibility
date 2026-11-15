// The verification desk: one field for a link, a claim or an article, an optional image, optional
// details, and the Verify button inside the same box.

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ClipboardEvent,
  type DragEvent,
  type FormEvent,
  type KeyboardEvent,
} from 'react'
import type { ApiError } from '../api/client'
import type { AnalyzeInput } from '../api/types'
import { detectInput } from '../lib/detectInput'
import { TIMING, prefersReducedMotion } from '../lib/timing'
import type { AnalyzePhase } from '../lib/useAnalyze'
import { Expander } from './Expander'
import { AlertTriangle, ChevronDown, Close, ImageIcon, Spinner } from './Icons'

// The same limits the service enforces, checked here first so a refusal costs no upload.
const MAX_TEXT = 20_000
const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/bmp']

const EXAMPLES = [
  {
    label: 'A viral health claim',
    text: 'Drinking cow urine and applying cow dung can cure and prevent COVID-19.',
  },
  { label: 'A real event', text: "Chandrayaan-3 landed near the Moon's south pole on 23 August 2023." },
  { label: 'A rumour about money', text: 'Reserve Bank of India announces new Rs 1000 currency notes.' },
]

const NOTHING_TO_CHECK = 'Paste a link, a claim or some text first, or add an image.'
const IMAGE_TOO_LARGE = 'That image is larger than 10 MB. Try a smaller copy.'
const NOT_AN_IMAGE = 'That file is not an image we can read. Use a JPEG, PNG, WebP, GIF or BMP.'

interface Draft {
  value: string
  image: File | null
  sourceLink: string
  claimedDate: string
  claimedPlace: string
  publishDate: string
  detailsOpen: boolean
}

const emptyDraft = (): Draft => ({
  value: '',
  image: null,
  sourceLink: '',
  claimedDate: '',
  claimedPlace: '',
  publishDate: '',
  detailsOpen: false,
})

// What the reader has typed outlives the page: pressing Back from a result returns to the desk
// with everything still in place.
let draft = emptyDraft()

/** Start the desk clean (used by "Check something else"). */
export function clearDraft(): void {
  draft = emptyDraft()
}

/** A source link typed by hand may lack its scheme; tidy it when we can, pass it on as typed when not. */
function tidyLink(raw: string): string | undefined {
  const detected = detectInput(raw)
  if (detected.kind === 'link') return detected.url
  return detected.kind === 'text' ? detected.text : undefined
}

interface DeskProps {
  phase: AnalyzePhase
  /** true while the skeleton and status line are on screen */
  progressShown: boolean
  error: ApiError | null
  onSubmit: (input: AnalyzeInput) => void
}

const fieldLabel = 'text-[13px] font-medium text-muted'
const fieldInput =
  'mt-1 h-11 w-full rounded-lg border border-line bg-paper px-3 text-[15px] text-body placeholder:text-muted'
const quietButton =
  'hover-fade inline-flex h-11 cursor-pointer items-center gap-2 rounded-lg px-3 text-sm font-medium text-muted hover:bg-paper hover:text-ink'

export function Desk({ phase, progressShown, error, onSubmit }: DeskProps) {
  const [form, setForm] = useState<Draft>(draft)
  const [notice, setNotice] = useState<string | null>(null)
  const [imageLeaving, setImageLeaving] = useState(false)
  const field = useRef<HTMLTextAreaElement>(null)

  const busy = phase === 'pressed' || phase === 'pending' || phase === 'leaving'
  const detected = useMemo(() => detectInput(form.value), [form.value])
  const update = (patch: Partial<Draft>) => setForm((current) => ({ ...current, ...patch }))

  useEffect(() => {
    draft = form
  }, [form])

  // A refused or failed check hands the field back, with the text intact and the cursor in it.
  useEffect(() => {
    if (phase === 'error') field.current?.focus()
  }, [phase])

  const thumbnail = useMemo(
    () => (form.image && typeof URL.createObjectURL === 'function' ? URL.createObjectURL(form.image) : null),
    [form.image],
  )
  useEffect(() => () => void (thumbnail && URL.revokeObjectURL(thumbnail)), [thumbnail])

  function attach(file: File | undefined) {
    if (!file) return
    if (!IMAGE_TYPES.includes(file.type)) return setNotice(NOT_AN_IMAGE)
    if (file.size > MAX_IMAGE_BYTES) return setNotice(IMAGE_TOO_LARGE)
    setNotice(null)
    setImageLeaving(false)
    update({ image: file })
  }

  function removeImage() {
    if (prefersReducedMotion()) return update({ image: null })
    setImageLeaving(true) // fade out first, so the reader sees where it went
    window.setTimeout(() => {
      update({ image: null })
      setImageLeaving(false)
    }, TIMING.leave)
  }

  function submit(event?: FormEvent) {
    event?.preventDefault()
    if (busy) return
    if (detected.kind === 'empty' && !form.image) {
      setNotice(NOTHING_TO_CHECK)
      field.current?.focus()
      return
    }
    setNotice(null)

    const input: AnalyzeInput = {}
    if (detected.kind === 'link') {
      input.url = detected.url
    } else {
      if (detected.kind === 'text') input.text = detected.text
      const link = tidyLink(form.sourceLink)
      if (link) input.url = link
    }
    if (form.image) input.image = form.image
    if (form.claimedDate) input.claimedDate = form.claimedDate
    if (form.claimedPlace.trim()) input.claimedLocation = form.claimedPlace.trim()
    if (form.publishDate) input.publishDate = form.publishDate
    onSubmit(input)
  }

  function runExample(text: string) {
    if (busy) return
    setNotice(null)
    setForm({ ...emptyDraft(), value: text })
    onSubmit({ text })
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) submit()
  }

  function onPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const file = Array.from(event.clipboardData?.files ?? []).find((f) => f.type.startsWith('image/'))
    if (!file) return
    event.preventDefault()
    attach(file)
  }

  function onDrop(event: DragEvent<HTMLFormElement>) {
    const file = event.dataTransfer?.files?.[0]
    if (!file) return
    event.preventDefault()
    attach(file)
  }

  const checking =
    detected.kind === 'link'
      ? 'Checking this link'
      : detected.kind === 'text'
        ? 'Checking this text'
        : form.image
          ? 'Checking this image'
          : ''
  const message = notice ?? (phase === 'error' ? error?.message : null) ?? null

  return (
    <div className="mt-8">
      <form
        onSubmit={submit}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        noValidate
        className="rounded-2xl border border-line bg-surface shadow-[0_1px_2px_rgb(22_32_46/0.05),0_12px_32px_rgb(22_32_46/0.07)] has-[textarea:focus]:border-accent has-[textarea:focus]:shadow-[0_0_0_1px_var(--accent),0_12px_32px_rgb(22_32_46/0.07)]"
      >
        <textarea
          ref={field}
          aria-label="Paste a link, a claim, or the text of an article"
          aria-describedby="desk-hint"
          placeholder="Paste a link, a claim, or the text of an article"
          value={form.value}
          maxLength={MAX_TEXT}
          readOnly={busy}
          rows={4}
          onChange={(e) => update({ value: e.target.value })}
          onKeyDown={onKeyDown}
          onPaste={onPaste}
          className="block max-h-[40dvh] min-h-[124px] w-full resize-none rounded-t-2xl bg-transparent px-5 pb-2 pt-5 text-[17px] leading-7 text-body [field-sizing:content] placeholder:text-muted focus:outline-none"
        />

        {form.image && (
          <div className={`mx-4 mb-2 ${imageLeaving ? 'leaving' : 'pop'}`}>
            <div className="inline-flex max-w-full items-center gap-3 rounded-xl border border-line bg-paper py-1 pl-1 pr-1">
              {thumbnail ? (
                <img src={thumbnail} alt="" className="size-10 rounded-lg object-cover" />
              ) : (
                <span className="grid size-10 place-items-center rounded-lg bg-line text-muted">
                  <ImageIcon size={18} />
                </span>
              )}
              <span className="min-w-0 truncate text-sm text-ink">{form.image.name}</span>
              <button
                type="button"
                onClick={removeImage}
                disabled={busy}
                aria-label={`Remove ${form.image.name}`}
                className="hover-fade grid size-11 shrink-0 cursor-pointer place-items-center rounded-lg text-muted hover:bg-line hover:text-ink"
              >
                <Close size={18} />
              </button>
            </div>
          </div>
        )}

        <Expander id="desk-details" open={form.detailsOpen}>
          <div className="grid gap-4 px-5 pb-3 pt-2 sm:grid-cols-2">
            {detected.kind !== 'link' && (
              <label className="sm:col-span-2">
                <span className={fieldLabel}>Link to the source</span>
                <input
                  type="url"
                  inputMode="url"
                  autoComplete="off"
                  placeholder="https://www.example.com/article"
                  value={form.sourceLink}
                  readOnly={busy}
                  onChange={(e) => update({ sourceLink: e.target.value })}
                  className={fieldInput}
                />
              </label>
            )}
            <label>
              <span className={fieldLabel}>Date the story gives</span>
              <input
                type="date"
                value={form.claimedDate}
                readOnly={busy}
                onChange={(e) => update({ claimedDate: e.target.value })}
                className={fieldInput}
              />
            </label>
            <label>
              <span className={fieldLabel}>Date it was published</span>
              <input
                type="date"
                value={form.publishDate}
                readOnly={busy}
                onChange={(e) => update({ publishDate: e.target.value })}
                className={fieldInput}
              />
            </label>
            <label className="sm:col-span-2">
              <span className={fieldLabel}>Place it names</span>
              <input
                type="text"
                autoComplete="off"
                placeholder="For example: India Gate, Delhi"
                value={form.claimedPlace}
                readOnly={busy}
                onChange={(e) => update({ claimedPlace: e.target.value })}
                className={fieldInput}
              />
            </label>
            <p className="text-[13px] leading-5 text-muted sm:col-span-2">
              All optional. A check that needs something you did not give is skipped, never counted against the
              story.
            </p>
          </div>
        </Expander>

        <div className="flex flex-wrap items-center gap-1 px-3 pb-3 pt-1">
          <label className={`${quietButton} has-[input:focus-visible]:outline-2 has-[input:focus-visible]:outline-offset-2 has-[input:focus-visible]:outline-accent`}>
            <ImageIcon size={18} />
            Image
            <input
              type="file"
              aria-label="Add an image"
              accept={IMAGE_TYPES.join(',')}
              disabled={busy}
              className="sr-only"
              onChange={(e) => {
                attach(e.target.files?.[0])
                e.target.value = '' // so the same file can be chosen again after removing it
              }}
            />
          </label>
          <button
            type="button"
            aria-expanded={form.detailsOpen}
            aria-controls="desk-details"
            onClick={() => update({ detailsOpen: !form.detailsOpen })}
            className={quietButton}
          >
            Add details
            <ChevronDown
              size={16}
              className={`transition-transform duration-[var(--dur-expand)] ease-[var(--ease-in-out)] ${form.detailsOpen ? 'rotate-180' : ''}`}
            />
          </button>

          <button
            type="submit"
            disabled={busy}
            className="press press-darken relative ml-auto inline-flex h-11 min-w-[132px] basis-full cursor-pointer items-center justify-center rounded-lg bg-accent px-6 text-[15px] font-semibold text-on-accent disabled:cursor-default sm:basis-auto"
          >
            <span aria-hidden={progressShown} className={progressShown ? 'invisible' : ''}>
              Verify
            </span>
            {progressShown && (
              <span className="absolute inset-0 flex items-center justify-center gap-2">
                <Spinner size={18} />
                Checking…
              </span>
            )}
          </button>
        </div>
      </form>

      <p id="desk-hint" className="mt-2 min-h-5 px-1 text-[13px] leading-5 text-muted">
        {checking}
        {form.value.length >= MAX_TEXT && ' · Only the first 20,000 characters will be checked.'}
      </p>

      {message && (
        <p role="alert" className="mt-2 flex items-start gap-2 px-1 text-[15px] leading-6 text-rust">
          <AlertTriangle size={18} className="mt-[3px] shrink-0" />
          <span>{message}</span>
        </p>
      )}

      {!progressShown && (
        <div role="group" aria-label="Try an example" className="mt-6 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-sm text-muted">Try an example</span>
          {EXAMPLES.map((example) => (
            <button
              key={example.label}
              type="button"
              disabled={busy}
              onClick={() => runExample(example.text)}
              className="press hover-fade h-11 cursor-pointer rounded-full border border-line bg-surface px-4 text-sm font-medium text-ink hover:border-accent disabled:cursor-default"
            >
              {example.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
