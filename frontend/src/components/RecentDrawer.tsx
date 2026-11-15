import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, deleteAnalysis } from '../api/client'
import * as recent from '../lib/recent'
import type { RecentItem } from '../lib/recent'
import { Link } from '../lib/router'
import { formatDate } from '../translate/format'
import { Close } from './Icons'

/** How long "Undo" stays on offer before a deleted check is really deleted. */
const UNDO_WINDOW = 5_000

interface Pending {
  item: RecentItem
  /** where the row was, so Undo puts it back in the same place */
  index: number
}

interface RecentDrawerProps {
  open: boolean
  onClose: () => void
}

/**
 * "Recent checks": a drawer that slides in from the right edge and leaves the same way
 * (.drawer and .scrim in theme.css). It stays mounted so it can animate out, and is inert while
 * shut. The list is this browser's own; nobody else's checks are ever shown.
 *
 * Deleting a check cannot be reversed on the server, so the row goes at once but the request is
 * held for a few seconds behind an Undo.
 */
export function RecentDrawer({ open, onClose }: RecentDrawerProps) {
  const [items, setItems] = useState<RecentItem[]>([])
  const [pending, setPending] = useState<Pending | null>(null)
  const [problem, setProblem] = useState(false)
  const closeButton = useRef<HTMLButtonElement>(null)
  const timer = useRef<number | null>(null)
  const pendingRef = useRef<Pending | null>(null)
  pendingRef.current = pending

  /** Send the delete that was waiting behind Undo. */
  const commit = useCallback(() => {
    const waiting = pendingRef.current
    if (!waiting) return
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = null
    pendingRef.current = null
    setPending(null)

    const { item, index } = waiting
    deleteAnalysis(item.id).then(
      () => recent.remove(item.id),
      (cause: unknown) => {
        // Already gone from the server: the list entry was stale, so drop it all the same.
        if (cause instanceof ApiError && cause.kind === 'not_found') return recent.remove(item.id)
        setItems((current) => [...current.slice(0, index), item, ...current.slice(index)])
        setProblem(true)
      },
    )
  }, [])

  // Opening reads the list afresh and moves focus into the drawer; closing sends any waiting delete.
  useEffect(() => {
    if (open) {
      setItems(recent.list())
      setProblem(false)
      closeButton.current?.focus()
    } else {
      commit()
    }
  }, [open, commit])

  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const { overflow } = document.documentElement.style
    document.documentElement.style.overflow = 'hidden' // the page behind does not scroll
    return () => {
      document.removeEventListener('keydown', onKey)
      document.documentElement.style.overflow = overflow
    }
  }, [open, onClose])

  // Leaving the page with a delete still waiting: send it now.
  useEffect(() => {
    window.addEventListener('pagehide', commit)
    return () => window.removeEventListener('pagehide', commit)
  }, [commit])

  function remove(item: RecentItem) {
    commit() // one Undo at a time: an earlier delete is sent before the next one starts waiting
    setProblem(false)
    const index = items.findIndex((i) => i.id === item.id)
    setItems((current) => current.filter((i) => i.id !== item.id))
    const next = { item, index }
    pendingRef.current = next
    setPending(next)
    timer.current = window.setTimeout(commit, UNDO_WINDOW)
  }

  function undo() {
    const waiting = pendingRef.current
    if (!waiting) return
    if (timer.current !== null) window.clearTimeout(timer.current)
    timer.current = null
    pendingRef.current = null
    setPending(null)
    setItems((current) => [...current.slice(0, waiting.index), waiting.item, ...current.slice(waiting.index)])
  }

  const storageWorks = recent.available()

  return (
    <>
      <div data-testid="drawer-scrim" data-open={open} onClick={onClose} className="scrim fixed inset-0 z-40 bg-black/40" />
      <aside
        id="recent-checks"
        role="dialog"
        aria-modal="true"
        aria-labelledby="recent-heading"
        data-open={open}
        inert={!open}
        className="drawer fixed inset-y-0 right-0 z-50 flex w-full max-w-[400px] flex-col border-l border-line bg-surface data-[open=true]:shadow-[-12px_0_32px_rgb(22_32_46/0.12)] max-[479px]:max-w-none max-[479px]:border-l-0"
      >
        <div className="flex items-center justify-between gap-2 py-3 pl-5 pr-3">
          <h2 id="recent-heading" className="font-display text-[22px] font-semibold text-ink">
            Recent checks
          </h2>
          <button
            ref={closeButton}
            type="button"
            aria-label="Close recent checks"
            onClick={onClose}
            className="hover-fade grid size-11 cursor-pointer place-items-center rounded-lg text-muted hover:bg-paper hover:text-ink"
          >
            <Close />
          </button>
        </div>
        <p className="px-5 text-[13px] leading-5 text-muted">
          Kept in this browser only. Deleting a check here also deletes the stored copy.
        </p>

        {pending && (
          <p role="status" className="mx-5 mt-4 flex items-center justify-between gap-3 rounded-lg bg-paper py-1 pl-3 pr-1 text-sm text-ink">
            Check deleted.
            <button
              type="button"
              onClick={undo}
              className="hover-fade h-11 cursor-pointer rounded-lg px-3 font-semibold text-accent hover:bg-surface"
            >
              Undo
            </button>
          </p>
        )}
        {problem && (
          <p role="alert" className="mx-5 mt-4 text-sm leading-6 text-rust">
            We could not delete that check. Please try again.
          </p>
        )}

        <div className="mt-4 min-h-0 flex-1 overflow-y-auto overscroll-contain px-2 pb-6">
          {items.length === 0 ? (
            <p className="px-3 text-[15px] leading-6 text-muted">
              {storageWorks
                ? 'Checks you run on this device will appear here.'
                : 'This browser is not keeping a list, which happens in private windows.'}
            </p>
          ) : (
            <ul>
              {items.map((item) => (
                <li key={item.id} className="flex items-start gap-1">
                  <Link
                    href={`/check/${item.id}`}
                    onClick={onClose}
                    className={`hover-fade tone-${item.tone} block min-w-0 flex-1 rounded-lg px-3 py-3 hover:bg-paper`}
                  >
                    <span className="line-clamp-2 text-[15px] leading-6 text-ink">{item.preview || 'A check'}</span>
                    <span className="mt-1 flex flex-wrap items-center gap-x-2 text-[13px] leading-5 text-muted">
                      <span className="inline-flex items-center gap-1.5">
                        <span aria-hidden="true" className="size-2 shrink-0 rounded-full bg-tone" />
                        <span>{item.label}</span>
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>{formatDate(item.at) ?? ''}</span>
                    </span>
                  </Link>
                  <button
                    type="button"
                    aria-label={`Delete the check “${item.preview.slice(0, 60)}”`}
                    onClick={() => remove(item)}
                    className="hover-fade mt-1 grid size-11 shrink-0 cursor-pointer place-items-center rounded-lg text-muted hover:bg-paper hover:text-rust"
                  >
                    <Close size={18} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </>
  )
}
