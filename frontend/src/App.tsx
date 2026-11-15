import { useCallback, useEffect, useRef, useState } from 'react'
import { Header } from './components/Header'
import { RecentDrawer } from './components/RecentDrawer'
import { useRoute } from './lib/router'
import { CheckPage } from './pages/CheckPage'
import { DeskPage } from './pages/DeskPage'
import { HowItWorksPage } from './pages/HowItWorksPage'
import { NotFound } from './pages/NotFound'

export function App() {
  const route = useRoute()
  const [recentOpen, setRecentOpen] = useState(false)
  const recentButton = useRef<HTMLButtonElement>(null)
  const wasOpen = useRef(false)
  const closeRecent = useCallback(() => setRecentOpen(false), [])

  // When the drawer closes, focus goes back to the button that opened it.
  useEffect(() => {
    if (wasOpen.current && !recentOpen) recentButton.current?.focus()
    wasOpen.current = recentOpen
  }, [recentOpen])

  return (
    <>
      {/* While the drawer is open the page behind it is inert, so Tab stays inside the drawer. */}
      <div inert={recentOpen}>
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:font-semibold focus:text-ink"
        >
          Skip to the content
        </a>
        <Header recentOpen={recentOpen} onOpenRecent={() => setRecentOpen(true)} recentButton={recentButton} />
        {route.name === 'desk' && <DeskPage />}
        {route.name === 'check' && <CheckPage key={route.id} id={route.id} />}
        {route.name === 'how' && <HowItWorksPage />}
        {route.name === 'not_found' && <NotFound what="page" />}
      </div>
      <RecentDrawer open={recentOpen} onClose={closeRecent} />
    </>
  )
}
