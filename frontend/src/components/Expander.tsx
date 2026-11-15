import { useState, type ReactNode } from 'react'

interface ExpanderProps {
  open: boolean
  id: string
  children: ReactNode
  /** Do not build the contents until the first time it opens (used for technical details). */
  lazy?: boolean
  className?: string
}

/**
 * A region that opens and closes in place. The height animates through CSS grid rows
 * (.expander in theme.css), so nothing is measured in JavaScript and a half-finished open or
 * close simply turns around. While closed the contents are inert: no focus, no clicks, and
 * screen readers skip them.
 */
export function Expander({ open, id, children, lazy = false, className = '' }: ExpanderProps) {
  const [everOpened, setEverOpened] = useState(open)
  if (open && !everOpened) setEverOpened(true)

  return (
    <div id={id} className={`expander ${className}`} data-open={open}>
      <div className="expander-inner" inert={!open}>
        {lazy && !everOpened ? null : children}
      </div>
    </div>
  )
}
