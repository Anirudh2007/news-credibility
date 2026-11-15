const STEPS = [
  {
    title: 'Submit a story or claim',
    body: 'Paste a news link, a headline or an excerpt from an article, or add an image.',
  },
  {
    title: 'We cross-check the evidence',
    body: 'We pick out the factual claims and check them against published fact-checks and independent reporting.',
  },
  {
    title: 'Read the breakdown',
    body: 'See what is corroborated and what lacks context, then open the sources and judge for yourself.',
  },
]

/** The empty state: how a check works, in three steps, shown until one is running. */
export function Guide() {
  return (
    <section aria-labelledby="guide-heading" className="mt-16">
      <h2 id="guide-heading" className="sr-only">
        How it works
      </h2>
      <ol className="grid gap-8 sm:grid-cols-3 sm:gap-6">
        {STEPS.map((step, i) => (
          <li key={step.title}>
            <span aria-hidden="true" className="font-display text-[32px] leading-8 text-accent">
              {i + 1}
            </span>
            <h3 className="mt-2 text-base font-semibold text-ink">{step.title}</h3>
            <p className="mt-1 text-[15px] leading-6 text-muted">{step.body}</p>
          </li>
        ))}
      </ol>
    </section>
  )
}
