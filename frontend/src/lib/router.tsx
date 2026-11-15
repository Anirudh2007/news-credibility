// Three pages do not need a routing library: the address bar is the state, and this reads it.

import { useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from 'react'

export type Route =
  | { name: 'desk' }
  | { name: 'check'; id: string }
  | { name: 'how' }
  | { name: 'not_found' }

export function parseRoute(pathname: string): Route {
  const path = pathname.replace(/\/+$/, '') || '/'
  if (path === '/') return { name: 'desk' }
  if (path === '/how-it-works') return { name: 'how' }
  const check = /^\/check\/([A-Za-z0-9_-]+)$/.exec(path)
  if (check) return { name: 'check', id: check[1] }
  return { name: 'not_found' }
}

const NAVIGATED = 'newscred:navigated'

export function navigate(path: string): void {
  if (path !== window.location.pathname) window.history.pushState(null, '', path)
  window.dispatchEvent(new Event(NAVIGATED))
  window.scrollTo(0, 0)
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener('popstate', onChange)
  window.addEventListener(NAVIGATED, onChange)
  return () => {
    window.removeEventListener('popstate', onChange)
    window.removeEventListener(NAVIGATED, onChange)
  }
}

export function usePathname(): string {
  return useSyncExternalStore(subscribe, () => window.location.pathname)
}

export function useRoute(): Route {
  return parseRoute(usePathname())
}

/** An ordinary link that changes page without a reload. Modified clicks (new tab) behave as usual. */
export function Link({ href, onClick, ...rest }: AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) {
  const go = (event: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(event)
    const modified = event.metaKey || event.ctrlKey || event.shiftKey || event.altKey
    if (event.defaultPrevented || event.button !== 0 || modified) return
    event.preventDefault()
    navigate(href)
  }
  return <a href={href} onClick={go} {...rest} />
}
