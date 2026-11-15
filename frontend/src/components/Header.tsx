import type { RefObject } from 'react'
import { clearDraft } from './Desk'
import { History } from './Icons'
import { ThemeMenu } from './ThemeMenu'
import { Link } from '../lib/router'

interface HeaderProps {
  recentOpen: boolean
  onOpenRecent: () => void
  /** the drawer hands focus back to this button when it closes */
  recentButton: RefObject<HTMLButtonElement | null>
}

const quiet =
  'hover-fade inline-flex h-11 min-w-11 cursor-pointer items-center justify-center gap-2 rounded-lg px-2 text-sm font-medium text-muted hover:bg-surface hover:text-ink sm:px-3'

/** The name on the left links home. On the right: two quiet links, then the theme choice. Nothing else. */
export function Header({ recentOpen, onOpenRecent, recentButton }: HeaderProps) {
  return (
    <header className="mx-auto flex w-full max-w-[720px] items-center justify-between gap-2 px-4 pt-4 sm:px-6">
      <Link
        href="/"
        onClick={clearDraft}
        className="-ml-2 inline-flex h-11 items-center rounded-lg px-2 font-display text-[20px] font-semibold text-ink"
      >
        News Credibility
      </Link>
      <nav aria-label="Site" className="-mr-2 flex items-center">
        <Link href="/how-it-works" className={quiet}>
          How it works
        </Link>
        <button
          ref={recentButton}
          type="button"
          aria-label="Recent checks"
          aria-expanded={recentOpen}
          aria-controls="recent-checks"
          onClick={onOpenRecent}
          className={quiet}
        >
          <History size={18} />
          <span className="max-[479px]:sr-only">Recent checks</span>
        </button>
        <ThemeMenu />
      </nav>
    </header>
  )
}
