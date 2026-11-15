import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { useTheme, type ThemeChoice } from '../lib/theme'
import { Check, Monitor, Moon, Sun } from './Icons'

const OPTIONS: { value: ThemeChoice; label: string; Icon: (p: { size?: number; className?: string }) => ReactNode }[] = [
  { value: 'system', label: 'System', Icon: Monitor },
  { value: 'light', label: 'Light', Icon: Sun },
  { value: 'dark', label: 'Dark', Icon: Moon },
]

/**
 * The theme choice, at the far right of the header where settings usually sit. One button showing
 * the current choice; it opens a three-item menu (System, Light, Dark).
 *
 * The menu grows out of its button (.popover in theme.css: scale from the top-right corner) and
 * stays mounted so it can fade out; while shut it is inert. Moving between items with the arrow
 * keys has no animation: the keys are faster than any transition.
 */
export function ThemeMenu() {
  const [choice, setChoice] = useTheme()
  const [open, setOpen] = useState(false)
  const container = useRef<HTMLDivElement>(null)
  const button = useRef<HTMLButtonElement>(null)
  const items = useRef<(HTMLButtonElement | null)[]>([])
  const menuId = useId()
  const current = OPTIONS.find((o) => o.value === choice) ?? OPTIONS[0]

  // Opening puts focus on the option in use.
  useEffect(() => {
    if (open) items.current[OPTIONS.indexOf(current)]?.focus()
    // only when the menu opens, not when the choice changes while it is open
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // A click anywhere else closes it.
  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    return () => document.removeEventListener('pointerdown', onPointerDown)
  }, [open])

  function close() {
    setOpen(false)
    button.current?.focus()
  }

  function choose(value: ThemeChoice) {
    setChoice(value)
    close()
  }

  function onMenuKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const at = items.current.findIndex((el) => el === document.activeElement)
    const go = (index: number) => {
      event.preventDefault()
      items.current[(index + OPTIONS.length) % OPTIONS.length]?.focus()
    }
    if (event.key === 'ArrowDown') go(at + 1)
    else if (event.key === 'ArrowUp') go(at - 1)
    else if (event.key === 'Home') go(0)
    else if (event.key === 'End') go(OPTIONS.length - 1)
    else if (event.key === 'Escape') {
      event.stopPropagation()
      close()
    } else if (event.key === 'Tab') setOpen(false)
  }

  return (
    <div ref={container} className="relative">
      <button
        ref={button}
        type="button"
        aria-label={`Theme: ${current.label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={menuId}
        onClick={() => setOpen(!open)}
        className="hover-fade grid size-11 cursor-pointer place-items-center rounded-lg text-muted hover:bg-surface hover:text-ink"
      >
        <current.Icon size={18} />
      </button>

      <div
        id={menuId}
        role="menu"
        aria-label="Theme"
        data-open={open}
        inert={!open}
        onKeyDown={onMenuKeyDown}
        className="popover absolute right-0 top-full z-30 mt-1 w-44 rounded-xl border border-line bg-surface p-1 shadow-[0_8px_24px_rgb(0_0_0/0.14)]"
      >
        {OPTIONS.map((option, i) => {
          const checked = option.value === choice
          return (
            <button
              key={option.value}
              ref={(el) => {
                items.current[i] = el
              }}
              type="button"
              role="menuitemradio"
              aria-checked={checked}
              onClick={() => choose(option.value)}
              className={`flex h-11 w-full cursor-pointer items-center gap-3 rounded-lg px-3 text-left text-sm hover:bg-paper ${checked ? 'font-semibold text-ink' : 'text-body'}`}
            >
              <option.Icon size={18} className="shrink-0 text-muted" />
              <span className="flex-1">{option.label}</span>
              {checked && <Check size={16} className="shrink-0 text-accent" />}
            </button>
          )
        })}
      </div>
    </div>
  )
}
