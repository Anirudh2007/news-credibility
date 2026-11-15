import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

afterEach(() => {
  cleanup()
  try {
    window.localStorage.clear()
    window.sessionStorage.clear()
  } catch {
    // a test may have replaced storage with one that throws
  }
  window.history.replaceState(null, '', '/')
})

// jsdom has no matchMedia; the app asks it about reduced motion.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList
}

// jsdom does not scroll; the router asks it to on every page change.
window.scrollTo = () => {}

afterEach(() => {
  // a test may have chosen a theme
  document.documentElement.removeAttribute('data-theme')
  document.documentElement.classList.remove('theme-switching')
})
