import { describe, expect, it } from 'vitest'
import * as fx from './__fixtures__/reports'
import { endQuote, formatDate, publication, safeUrl, splitQuote } from './format'
import { translateReport } from './index'
import type { ResultView } from './types'

const card = (view: ResultView, key: string) => view.cards.find((c) => c.key === key)!

describe('verdict label', () => {
  it.each([
    ['supported', 'Well corroborated', 'good'],
    ['mostlyCredible', 'Mostly credible', 'ok'],
    ['mixedThree', 'Needs context', 'warn'],
    ['refuted', 'Contradicted by evidence', 'bad'],
    ['silence', 'No trusted reporting found', 'bad'],
    ['imageMismatch', 'Out of context', 'bad'],
    ['satireSource', 'Low credibility', 'bad'],
    ['unverified', 'Unverified', 'neutral'],
    ['unknownBand', 'Could not be checked', 'neutral'],
  ])('%s reads "%s"', (name, label, tone) => {
    const { verdict } = translateReport(fx.ALL[name])
    expect(verdict.label).toBe(label)
    expect(verdict.tone).toBe(tone)
  })

  it('does not call a claim contradicted when the only evidence is about something else', () => {
    const { verdict } = translateReport(fx.weakEvidence)
    expect(verdict.label).toBe('Low credibility')
  })

  it('places the band marker from the rating and leaves it off when there is none', () => {
    expect(translateReport(fx.refuted).verdict.bandPosition).toBe(12)
    expect(translateReport(fx.unverified).verdict.bandPosition).toBeNull()
  })
})

describe('bottom line', () => {
  it('says the claim is contradicted and names the fact-checker whose rating agrees', () => {
    const { bottomLine } = translateReport(fx.refuted).verdict
    expect(bottomLine[0]).toBe('Independent sources contradict this claim. BOOM rated a matching claim “False”.')
  })

  it('says the claim is backed and adds the outlet’s record as context', () => {
    const { bottomLine } = translateReport(fx.supported).verdict
    expect(bottomLine).toEqual([
      'Independent sources back this claim.',
      'The link is from thehindu.com, which has a reliable record.',
    ])
  })

  it('counts contradicted, backed and open claims when there are several', () => {
    const { bottomLine } = translateReport(fx.mixedThree).verdict
    expect(bottomLine).toEqual([
      'Independent sources contradict 1 of the 3 claims we checked and back 1. BOOM rated a matching claim “False”.',
      '1 other claim could not be confirmed either way.',
    ])
  })

  it('explains silence as evidence and says it is not proof', () => {
    const { bottomLine } = translateReport(fx.silence).verdict
    expect(bottomLine).toEqual([
      'No trusted news outlet has reported this, although a claim like it would normally be widely covered.',
      'Silence is not proof: something from the last few hours may not have been reported yet.',
    ])
  })

  it('leads with the image when the image is what does not fit', () => {
    const { bottomLine } = translateReport(fx.imageMismatch).verdict
    expect(bottomLine).toEqual([
      'The image does not appear to match what the text says about it.',
      'An earlier copy of this image was already online on 3 Aug 2019, before the date the story gives.',
    ])
  })

  it('says why nothing could be checked, in the backend’s reason without its talk of numbers', () => {
    const { bottomLine } = translateReport(fx.unverified).verdict
    expect(bottomLine).toEqual([
      'We could not check this: no evidence about its claims was found, and no image or source URL was given.',
      'Try adding the source link or the image, or look it up with a fact-checker such as PIB Fact Check, BOOM or Alt News.',
    ])
  })

  it('flags evidence that only loosely matches instead of treating it as settled', () => {
    const { bottomLine } = translateReport(fx.weakEvidence).verdict
    expect(bottomLine).toEqual([
      'We found no independent source that settles this claim either way.',
      'The closest sources we found only loosely match what is claimed, so read them before relying on this.',
    ])
  })

  it('still produces a sentence for a report from an early model version', () => {
    const { bottomLine } = translateReport(fx.oldStored).verdict
    expect(bottomLine).toEqual(['We found nothing in this that could be checked against evidence.'])
  })
})

describe('reasoning cards', () => {
  it('always returns corroboration, context and source, in that order', () => {
    for (const report of Object.values(fx.ALL)) {
      expect(translateReport(report).cards.map((c) => c.key)).toEqual(['corroboration', 'context', 'source'])
    }
  })

  it('counts backed, contradicted and open claims, and names the fact-checkers found', () => {
    const c = card(translateReport(fx.mixedThree), 'corroboration')
    expect(c.state).toBe('checked')
    expect(c.headline).toBe('1 of 3 claims are backed by independent sources.')
    expect(c.detail).toEqual([
      '1 contradicted.',
      '1 could not be confirmed either way.',
      'Published fact-check found: BOOM.',
    ])
    expect(c.tone).toBe('bad')
  })

  it('words a single claim without counting', () => {
    expect(card(translateReport(fx.supported), 'corroboration').headline).toBe(
      'The claim is backed by independent sources.',
    )
    expect(card(translateReport(fx.refuted), 'corroboration').headline).toBe(
      'The claim is contradicted by independent sources.',
    )
    expect(card(translateReport(fx.unverified), 'corroboration').headline).toBe(
      'No independent source settles the claim.',
    )
  })

  it('counts vague sentences separately from claims that were checked', () => {
    const c = card(translateReport(fx.conflicting), 'corroboration')
    expect(c.headline).toBe('Sources disagree about the claim.')
    expect(c.detail).toEqual(['1 too vague to check.'])
  })

  it('says a check did not run and why, without making it sound like a mark against the story', () => {
    const view = translateReport(fx.refuted)
    expect(card(view, 'context')).toMatchObject({
      state: 'not_checked',
      tone: 'neutral',
      headline: 'Not checked: no image or date was given.',
    })
    expect(card(view, 'source')).toMatchObject({
      state: 'not_checked',
      tone: 'neutral',
      headline: 'Not checked: no link was given.',
    })
    expect(card(translateReport(fx.linkNotRead), 'corroboration').headline).toBe('Not checked: no text was given.')
  })

  it('lists what does not fit about an image, worst first', () => {
    const c = card(translateReport(fx.imageMismatch), 'context')
    expect(c.state).toBe('checked')
    expect(c.tone).toBe('bad')
    expect(c.headline).toBe('The image does not appear to match what the text says about it.')
    expect(c.detail).toEqual([
      'It looks like a real image attached to a story it does not belong to.',
      'An earlier copy of this image was already online on 3 Aug 2019, before the date the story gives.',
      'The image does not resemble reference photos of India Gate.',
    ])
  })

  it('says so plainly when the image and dates fit', () => {
    const c = card(translateReport(fx.imageFine), 'context')
    expect(c.tone).toBe('good')
    expect(c.headline).toBe('The image fits what the text says about it.')
    expect(c.detail).toEqual([
      'We found no earlier copy of this image in our reference collection.',
      'The dates in the story line up.',
    ])
  })

  it('describes the outlet’s record in words', () => {
    expect(card(translateReport(fx.supported), 'source')).toMatchObject({
      state: 'checked',
      tone: 'good',
      headline: 'thehindu.com has a reliable record.',
    })
    expect(card(translateReport(fx.satireSource), 'source')).toMatchObject({
      tone: 'warn',
      headline: 'theonion.com is a known satire site.',
    })
  })

  it('explains why a platform or an unlisted site has no record', () => {
    expect(card(translateReport(fx.platformSource), 'source').headline).toBe(
      'Not rated: x.com is a platform, not the author.',
    )
    expect(card(translateReport(fx.unratedSource), 'source').headline).toBe(
      'Not rated: example.org is not among the outlets we have ratings for.',
    )
  })

  it('renders three not-checked cards for a report from an early model version', () => {
    const { cards } = translateReport(fx.oldStored)
    expect(cards.every((c) => c.state === 'not_checked' && c.tone === 'neutral')).toBe(true)
  })
})

describe('claim cards', () => {
  it('shows the strongest source with its publication, date and link, and the fact-check found', () => {
    const [claim] = translateReport(fx.refuted).claims
    expect(claim).toMatchObject({ status: 'contradicted', statusLabel: 'Contradicted', tone: 'bad' })
    expect(claim.lead).toEqual({
      publication: 'pib.gov.in',
      url: 'https://pib.gov.in/factcheck/rs-1000-notes',
      date: '12 Mar 2024',
      question: 'Has the RBI announced new Rs 1000 notes?',
      quote: 'No. The RBI has made no such announcement.',
      counted: true,
      uncountedReason: null,
    })
    expect(claim.more).toHaveLength(1)
    expect(claim.more[0].publication).toBe('boomlive.in')
    expect(claim.factChecks).toEqual([
      {
        publisher: 'BOOM',
        rating: 'False',
        claimReviewed: 'RBI to issue new Rs 1000 notes',
        url: 'https://www.boomlive.in/fact-check/rs-1000-note',
        date: '14 Mar 2024',
      },
    ])
  })

  it.each([
    ['Supported', 'supported', 'Supported'],
    ['Refuted', 'contradicted', 'Contradicted'],
    ['Conflicting Evidence/Cherrypicking', 'mixed', 'Mixed evidence'],
    ['Not Enough Evidence', 'unverified', 'Unverified'],
  ])('maps the verdict %s to the pill "%s"', (verdict, status, label) => {
    const report = { ...fx.refuted, claims: [{ ...fx.refuted.claims![0], verdict }] }
    expect(translateReport(report).claims[0]).toMatchObject({ status, statusLabel: label })
  })

  it('marks a sentence that names nothing as too vague to check', () => {
    const vague = translateReport(fx.conflicting).claims[1]
    expect(vague).toMatchObject({ status: 'vague', statusLabel: 'Too vague to check', lead: null })
    expect(vague.notes[0]).toMatch(/does not say who or what it is about/)
  })

  it('keeps sources about a different claim out of the lead and labels them as not counted', () => {
    const [claim] = translateReport(fx.silence).claims
    expect(claim.status).toBe('unverified')
    expect(claim.lead).toBeNull()
    expect(claim.more[0]).toMatchObject({ counted: false, uncountedReason: 'About a different claim: not counted.' })
    expect(claim.notes).toContain(
      'A claim like this would normally be in the news. We searched trusted outlets and found no report.',
    )
  })

  it('reports what the news search found for a claim that would normally be news', () => {
    const [claim] = translateReport(fx.supported).claims
    expect(claim.notes).toContain(
      'A claim like this would normally be in the news. We searched trusted outlets and found 4 reports, and one confirms it.',
    )
  })

  it('shows a claim with only loosely matching evidence as unverified, with the reason', () => {
    const [claim] = translateReport(fx.weakEvidence).claims
    expect(claim).toMatchObject({ status: 'unverified', statusLabel: 'Unverified' })
    expect(claim.notes[0]).toBe(
      'The closest sources lean towards contradicting this claim, but they only loosely match it, so they were not counted.',
    )
  })

  it('never turns a non-web address into a link', () => {
    const [claim] = translateReport(fx.unsafeLink).claims
    expect(claim.lead?.url).toBeNull()
    expect(claim.factChecks[0].url).toBeNull()
  })
})

describe('what was read from a link', () => {
  it('echoes the headline and date read from the page', () => {
    expect(translateReport(fx.linkOnlyArticle).checked).toEqual({
      headline: 'Chandrayaan-3 lifts off from Sriharikota',
      published: '14 Jul 2023',
      note: null,
    })
  })

  it('says when the article could not be read', () => {
    expect(translateReport(fx.linkNotRead).checked.note).toBe(
      'The article itself was not checked: the page could not be loaded. Paste the headline or the text to check what it says.',
    )
  })
})

describe('technical details', () => {
  it('keeps the numbers for readers who ask for them', () => {
    const { technical } = translateReport(fx.refuted)
    expect(technical.score).toBe(12)
    expect(technical.band).toBe('Low')
    expect(technical.baseline).toBe(50)
    expect(technical.contributions).toEqual([
      { label: 'Claim verification', points: -38 },
      { label: 'Calibration', points: 0.4 },
    ])
    expect(technical.claims[0]).toMatchObject({ verdict: 'Refuted', confidence: 0.91, relevance: 0.93 })
    expect(technical.explanation).toMatchObject({ writtenBy: 'template' })
    expect(technical.skipped).toHaveLength(4)
    expect(technical.modelVersion).toBe('0.10.1+ab12cd')
  })
})

describe('reader-facing copy', () => {
  // Words and number shapes that belong to the model, not to the reader. Quoted evidence, claim text
  // and names are the sources' own words and are not scanned.
  const BANNED =
    /\b(score[ds]?|points?|weights?|weighted|logits?|shap|calibrat\w*|heuristic\w*|entailment|probabilit\w*|confidence)\b|(^|[\s(])[+−-]\d|\d\s*%|\/\s*100/i

  function writtenByUs(view: ResultView): string[] {
    return [
      view.verdict.label,
      ...view.verdict.bottomLine,
      ...view.cards.flatMap((c) => [c.title, c.headline, ...c.detail]),
      ...view.claims.flatMap((c) => [
        c.statusLabel,
        ...c.notes,
        ...[c.lead, ...c.more].flatMap((s) => (s?.uncountedReason ? [s.uncountedReason] : [])),
      ]),
      view.checked.note ?? '',
      view.disclaimer,
    ]
  }

  it.each(Object.keys(fx.ALL))('%s: no model vocabulary or score arithmetic above the fold', (name) => {
    for (const sentence of writtenByUs(translateReport(fx.ALL[name]))) {
      expect(sentence, sentence).not.toMatch(BANNED)
    }
  })

  it.each(Object.keys(fx.ALL))('%s: every sentence is a real sentence', (name) => {
    const view = translateReport(fx.ALL[name])
    for (const sentence of [...view.verdict.bottomLine, ...view.cards.flatMap((c) => c.detail)]) {
      expect(sentence).toMatch(/^[A-Z0-9“].*[.”]$/)
      expect(sentence).not.toMatch(/undefined|null|NaN|\s{2,}/)
    }
  })
})

describe('formatting helpers', () => {
  it('prints dates the way a newspaper would', () => {
    expect(formatDate('2024-03-12')).toBe('12 Mar 2024')
    expect(formatDate('2026-11-12T10:00:00+00:00')).toBe('12 Nov 2026')
    expect(formatDate('not a date')).toBeNull()
    expect(formatDate(null)).toBeNull()
  })

  it('allows only web addresses as links', () => {
    expect(safeUrl('https://example.com/a')).toBe('https://example.com/a')
    expect(safeUrl('http://example.com')).toBe('http://example.com')
    expect(safeUrl('javascript:alert(1)')).toBeNull()
    expect(safeUrl('  JAVASCRIPT:alert(1)')).toBeNull()
    expect(safeUrl('data:text/html,x')).toBeNull()
    expect(safeUrl('/relative')).toBeNull()
    expect(safeUrl(undefined)).toBeNull()
  })

  it('splits a question-and-answer passage and leaves other passages whole', () => {
    expect(splitQuote('Q: Is it true? A: No, it is not.')).toEqual({ question: 'Is it true?', quote: 'No, it is not.' })
    expect(splitQuote('Prices fell in some states.')).toEqual({ question: null, quote: 'Prices fell in some states.' })
  })

  it('drops the "<site> reported (<date>):" prefix news passages are stored with', () => {
    expect(splitQuote('theconversation.com reported (2023-08-24): India landed on the Moon.')).toEqual({
      question: null,
      quote: 'India landed on the Moon.',
    })
  })

  it('marks a passage that was cut off mid-sentence', () => {
    expect(endQuote('India made history with its Chandrayaan-3 lan')).toBe('India made history with its Chandrayaan-3 lan…')
    expect(endQuote('India made history.')).toBe('India made history.')
    expect(endQuote('He said “it landed”')).toBe('He said “it landed”')
  })

  it('names a publication from its domain or, failing that, from the link', () => {
    expect(publication('www.boomlive.in', null)).toBe('boomlive.in')
    expect(publication(null, 'https://www.thehindu.com/a/b')).toBe('thehindu.com')
    expect(publication(null, null)).toBe('Unnamed source')
  })
})
