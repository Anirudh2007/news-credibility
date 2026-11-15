import { useEffect, useRef, useState, type ReactNode } from 'react'
import { getHealth } from '../api/client'
import { ToneIcon } from '../components/Icons'
import { Link } from '../lib/router'
import type { VerdictKey } from '../translate/types'
import { VERDICTS } from '../translate/verdict'

const MEANINGS: [VerdictKey, string][] = [
  ['well_corroborated', 'Independent sources back what it says.'],
  ['mostly_credible', 'The evidence leans its way, with some gaps.'],
  ['needs_context', 'The evidence is mixed or thin. Read the claims one by one.'],
  ['contradicted', 'Independent sources say otherwise.'],
  ['no_trusted_reporting', 'It would be news if it were true, and no trusted outlet has reported it.'],
  ['out_of_context', 'The image or the dates do not fit the story.'],
  ['low_credibility', 'What we found counts against it, without one decisive source.'],
  ['unverified', 'We found no evidence either way, so we give no verdict.'],
]

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mt-12">
      <h2 className="font-display text-2xl font-semibold text-ink">{title}</h2>
      <div className="mt-4 space-y-4 text-[17px] leading-7 text-body">{children}</div>
    </section>
  )
}

function Point({ name, children }: { name: string; children: ReactNode }) {
  return (
    <p>
      <strong className="font-semibold text-ink">{name}</strong> {children}
    </p>
  )
}

type ModelUse = 'unknown' | 'off' | 'on' | 'unreachable'

export function HowItWorksPage() {
  const heading = useRef<HTMLHeadingElement>(null)
  // Whether this server sends text to an outside language model. Nothing is said until it is known,
  // so the privacy section never shows one thing and then changes.
  const [modelUse, setModelUse] = useState<ModelUse>('unknown')

  useEffect(() => {
    document.title = 'How it works · News Credibility'
    heading.current?.focus({ preventScroll: true })
    let live = true
    getHealth().then(
      (health) => live && setModelUse((health.explainer ?? '').startsWith('templates only') ? 'off' : 'on'),
      () => live && setModelUse('unreachable'),
    )
    return () => {
      live = false
    }
  }, [])

  return (
    <main id="main" className="mx-auto w-full max-w-[720px] px-4 pb-24 pt-10 sm:px-6 sm:pt-16">
      <h1
        ref={heading}
        tabIndex={-1}
        className="font-display text-[40px] font-semibold leading-[1.1] tracking-[-0.01em] text-ink focus:outline-none sm:text-[56px]"
      >
        How it works
      </h1>
      <p className="mt-4 text-lg leading-7 text-muted">
        A story is checked the way a fact-checker would do it: one claim at a time, against published evidence,
        with the image, the outlet and the dates looked at too.
      </p>

      <Section title="What we check">
        <Point name="The claims.">
          We pick out the statements of fact and check each one on its own against published fact-checks, an
          archive of claims that have already been checked, Wikipedia as it stood on the date of the claim, and
          reporting from about 300 trusted news outlets. Wikipedia and news reports can only confirm a claim;
          only a fact-check can contradict one.
        </Point>
        <Point name="Silence.">
          Some things are always reported: a national policy change, a major launch, a death in public life. If a
          claim like that has no report from any trusted outlet, we count the silence as a reason for doubt. It is
          not proof.
        </Point>
        <Point name="The image.">
          Whether it fits the text, whether an earlier copy is already in our reference collection, and, when the
          story names a landmark, whether the image resembles reference photos of that place.
        </Point>
        <Point name="The outlet.">
          If you give a link, we look the outlet up in public media-rating lists covering 1,082 outlets. Social
          platforms and sites that are not on those lists get no rating, because the platform is not the author.
        </Point>
        <Point name="The dates.">
          Whether the date the story gives fits the dates of the image and of the reporting we found.
        </Point>
        <p>
          A check that needs something you did not give, such as an image or a link, is skipped. It is never
          counted against the story.
        </p>
      </Section>

      <Section title="What we do not judge">
        <Point name="Wording and tone.">
          How a story is worded is never judged. A calm lie and an excitable truth are treated alike: only the
          evidence counts.
        </Point>
        <Point name="Opinions, questions and jokes.">Only statements of fact can be checked.</Point>
        <Point name="Video, and languages other than English.">Neither is supported yet.</Point>
      </Section>

      <Section title="What the verdicts mean">
        <dl className="space-y-3">
          {MEANINGS.map(([key, meaning]) => (
            <div key={key} className={`tone-${VERDICTS[key].tone} flex items-start gap-3`}>
              <ToneIcon tone={VERDICTS[key].tone} size={20} className="mt-1 shrink-0 text-tone" />
              <div>
                <dt className="font-semibold text-ink">{VERDICTS[key].label}</dt>
                <dd className="text-muted">{meaning}</dd>
              </div>
            </div>
          ))}
        </dl>
      </Section>

      <Section title="Limits">
        <p>
          Fact-checkers mostly write about false claims, so a true story is confirmed less often than a false one
          is caught. “Unverified” does not mean false.
        </p>
        <p>
          Image reuse is found only against a small reference collection, and an event from the last few hours may
          not have been reported yet.
        </p>
        <p>This is an aid for reading the evidence, not a final judgement.</p>
      </Section>

      <Section title="Your privacy">
        <p>
          Each check is stored on our server: the text or link you gave, and the result. An image is never stored,
          only a fingerprint of it.
        </p>
        <p>
          A check can be opened by anyone who has its link. The list under Recent checks is kept only in your
          browser, and deleting a check there also deletes the stored copy.
        </p>
        <p>
          To find evidence, the claims are sent to the Google Fact Check Tools service, to Wikipedia and to a news
          search service (Tavily). Images never leave our server.
        </p>
        {modelUse === 'on' && (
          <p>
            On this server the findings and your text are also sent to an outside language model provider, which
            writes the summary shown under technical details.
          </p>
        )}
        {modelUse === 'unreachable' && (
          <p>
            Some servers also send the findings and your text to an outside language model provider, which writes
            the summary shown under technical details. We could not confirm just now whether this one does.
          </p>
        )}
      </Section>

      <Link
        href="/"
        className="press mt-12 inline-flex h-11 items-center rounded-lg bg-accent px-5 text-[15px] font-semibold text-on-accent"
      >
        Check a story
      </Link>
    </main>
  )
}
