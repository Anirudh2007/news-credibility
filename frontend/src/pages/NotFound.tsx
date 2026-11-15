import { useEffect, useRef } from 'react'
import { Link } from '../lib/router'

const COPY = {
  check: {
    title: 'We could not find that check.',
    body: 'It may have been removed, or the link may be incomplete.',
  },
  page: {
    title: 'There is nothing at this address.',
    body: 'The link may be mistyped or out of date.',
  },
}

export function NotFound({ what }: { what: 'check' | 'page' }) {
  const heading = useRef<HTMLHeadingElement>(null)
  useEffect(() => {
    document.title = `${COPY[what].title} · News Credibility`
    heading.current?.focus({ preventScroll: true })
  }, [what])

  return (
    <main id="main" className="mx-auto w-full max-w-[720px] px-4 pb-24 pt-16 sm:px-6">
      <h1
        ref={heading}
        tabIndex={-1}
        className="font-display text-[36px] font-semibold leading-[1.1] text-ink focus:outline-none"
      >
        {COPY[what].title}
      </h1>
      <p className="mt-4 text-lg leading-7 text-muted">{COPY[what].body}</p>
      <Link
        href="/"
        className="press mt-8 inline-flex h-11 items-center rounded-lg bg-accent px-5 text-[15px] font-semibold text-on-accent"
      >
        Go to the desk
      </Link>
    </main>
  )
}
