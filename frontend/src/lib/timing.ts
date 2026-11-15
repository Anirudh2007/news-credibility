// ==========================================================================================
// STORYBOARD: Verify -> verdict            (milliseconds; T = press, R = the answer arrives)
//
//   T+0     the button shows pressed and is disabled; the field locks
//   T+300   still waiting: the button reads "Checking…", the guide gives way to a skeleton of
//           the result and one status line. Once shown it stays at least 500ms, so a fast
//           answer never makes it flicker. An answer before T+300 skips this stage entirely.
//   status  the line changes with elapsed time. The service sends no progress events, so each
//           line only says what is true at that moment:
//             0s   Reading your submission.
//             2s   Searching fact-checks, Wikipedia and trusted news. This is the slow part.
//             20s  Still searching. Some checks take up to a minute.
//             60s  The first check after a restart loads the models and can take a few minutes.
//   R+0     the desk and skeleton fade out (150)
//   R+150   the result page takes over; from here the delays below are measured
//     +0    verdict label
//     +90   bottom line
//     +180  band strip
//     +270  reasoning cards, 40 apart
//     +400  claim cards, the first six 40 apart, the rest with the sixth
//     +640  feedback and the rest of the page
//   ~R+1100 settled; focus moves to the verdict so screen readers announce it
//
// Each piece rises 8px and fades in over --dur-reveal on --ease-out (src/styles/theme.css).
// With reduced motion the fade-out and every delay are zero and the end state is the same.
// ==========================================================================================

export const TIMING = {
  /** wait this long before showing a busy state, so stored answers never flash one */
  showPendingAfter: 300,
  /** once the busy state shows, keep it at least this long */
  pendingMinVisible: 500,
  /** the desk and skeleton making way for the verdict (matches --dur-leave) */
  leave: 150,
  reveal: {
    verdict: 0,
    bottomLine: 90,
    band: 180,
    cards: 270,
    cardStep: 40,
    claims: 400,
    claimStep: 40,
    /** only the first six claims stagger; the rest arrive with the sixth */
    claimStaggerMax: 6,
    rest: 640,
  },
} as const

export const STATUS_LINES = [
  { after: 0, text: 'Reading your submission.' },
  { after: 2_000, text: 'Searching fact-checks, Wikipedia and trusted news. This is the slow part.' },
  { after: 20_000, text: 'Still searching. Some checks take up to a minute.' },
  { after: 60_000, text: 'The first check after a restart loads the models and can take a few minutes.' },
] as const

export function prefersReducedMotion(): boolean {
  return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/** The delay for the nth claim card in the reveal. */
export function claimDelay(index: number): number {
  const { claims, claimStep, claimStaggerMax } = TIMING.reveal
  return claims + Math.min(index, claimStaggerMax - 1) * claimStep
}
