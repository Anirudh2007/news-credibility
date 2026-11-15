import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import type { Report } from '../api/types'
import * as session from '../lib/session'
import * as fx from '../translate/__fixtures__/reports'
import { CheckPage } from './CheckPage'

vi.mock('../api/client', async (original) => ({
  ...(await original<typeof import('../api/client')>()),
  analyze: vi.fn(),
  getAnalysis: vi.fn(),
  sendFeedback: vi.fn(),
}))
const client = await import('../api/client')
const analyzeMock = vi.mocked(client.analyze)
const getAnalysisMock = vi.mocked(client.getAnalysis)
const sendFeedbackMock = vi.mocked(client.sendFeedback)

/** Open a result the way the desk does: the report is already in the session. */
function openFresh(report: Report, input: { text?: string; url?: string } = { text: 'What the reader typed.' }) {
  session.put(report.analysis_id, { report, input, fresh: true })
  return render(<CheckPage id={report.analysis_id} />)
}

beforeEach(() => {
  sendFeedbackMock.mockResolvedValue(undefined)
})
afterEach(() => {
  vi.resetAllMocks()
  session.clear()
})

describe('a result', () => {
  it('leads with the verdict in words and the bottom line', () => {
    openFresh(fx.refuted)
    expect(screen.getByRole('heading', { level: 1, name: 'Contradicted by evidence' })).toBeInTheDocument()
    expect(
      screen.getByText(/Independent sources contradict this claim\. BOOM rated a matching claim “False”\./),
    ).toBeInTheDocument()
  })

  it('moves focus to the verdict so it is announced', () => {
    openFresh(fx.refuted)
    expect(screen.getByRole('heading', { level: 1 })).toHaveFocus()
  })

  it('echoes what the reader asked about and offers a way back to the desk', () => {
    openFresh(fx.refuted, { text: 'Reserve Bank of India announces new Rs 1000 currency notes.' })
    expect(screen.getByText('You asked about')).toBeInTheDocument()
    expect(screen.getAllByText('Reserve Bank of India announces new Rs 1000 currency notes.').length).toBeGreaterThan(0)
    expect(screen.getByRole('link', { name: /check something else/i })).toHaveAttribute('href', '/')
  })

  it('shows the three reasoning cards, with unrun checks stated plainly', () => {
    openFresh(fx.refuted)
    const reasons = screen.getByRole('region', { name: 'Why we say this' })
    expect(within(reasons).getByText('Source corroboration')).toBeInTheDocument()
    expect(within(reasons).getByText('The claim is contradicted by independent sources.')).toBeInTheDocument()
    expect(within(reasons).getByText('Not checked: no image or date was given.')).toBeInTheDocument()
    expect(within(reasons).getByText('Not checked: no link was given.')).toBeInTheDocument()
  })

  it('shows each claim with its status, the quote it rests on and a link to read the source', () => {
    openFresh(fx.refuted)
    const claims = screen.getByRole('region', { name: 'Claim by claim' })
    expect(within(claims).getByText('Contradicted')).toBeInTheDocument()
    expect(within(claims).getByText('No. The RBI has made no such announcement.')).toBeInTheDocument()

    const source = within(claims).getByRole('link', { name: /pib\.gov\.in/ })
    expect(source).toHaveAttribute('href', 'https://pib.gov.in/factcheck/rs-1000-notes')
    expect(source).toHaveAttribute('target', '_blank')
    expect(source).toHaveAttribute('rel', expect.stringContaining('noopener'))
    expect(within(claims).getByRole('link', { name: /BOOM/ })).toHaveAttribute(
      'href',
      'https://www.boomlive.in/fact-check/rs-1000-note',
    )
    // the fact-checker's verdict is shown with the claim it actually reviewed
    expect(
      within(claims).getByText(/checked the claim “RBI to issue new Rs 1000 notes” and rated it “False” on 14 Mar 2024\./),
    ).toBeInTheDocument()
  })

  it('keeps further sources folded under the claim until asked for', async () => {
    const user = userEvent.setup()
    openFresh(fx.refuted)
    const toggle = screen.getByRole('button', { name: 'Show 1 more source' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.getByText('The viral message about a new Rs 1000 note is false.').closest('[inert]')).not.toBeNull()

    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('The viral message about a new Rs 1000 note is false.').closest('[inert]')).toBeNull()
  })

  it('never renders a non-web address as a link', () => {
    openFresh(fx.unsafeLink)
    expect(screen.getByText('evil.example')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /evil\.example/ })).not.toBeInTheDocument()
    expect(document.querySelector('a[href^="javascript:"], a[href^="data:"]')).toBeNull()
  })

  it('explains an image that does not fit the story', () => {
    openFresh(fx.imageMismatch)
    expect(screen.getByRole('heading', { level: 1, name: 'Out of context' })).toBeInTheDocument()
    expect(screen.getByText('The image does not resemble reference photos of India Gate.')).toBeInTheDocument()
  })

  it('says what was read from a link', () => {
    openFresh(fx.linkOnlyArticle, { url: 'https://www.thehindu.com/sci-tech/chandrayaan-3-launch' })
    expect(screen.getByText(/Chandrayaan-3 lifts off from Sriharikota/)).toBeInTheDocument()
    expect(screen.getByText(/published 14 Jul 2023/i)).toBeInTheDocument()
  })

  it.each(Object.keys(fx.ALL))('%s: shows no rating number or percentage until technical details are opened', (name) => {
    openFresh(fx.ALL[name])
    const text = document.body.textContent ?? ''
    expect(text).not.toMatch(/\d\s*%/)
    expect(text).not.toMatch(/\/\s*100|out of 100/)
    expect(text).not.toMatch(/\b(SHAP|calibration|confidence|Lowered by|Raised by)\b/)
  })

  it('renders a report from an early model version without breaking', () => {
    openFresh(fx.oldStored)
    expect(screen.getByRole('heading', { level: 1, name: 'Needs context' })).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Claim by claim' })).not.toBeInTheDocument()
  })
})

describe('the reveal', () => {
  it('plays for a result the reader just asked for', () => {
    openFresh(fx.refuted)
    expect(screen.getByRole('main')).toHaveAttribute('data-reveal', 'on')
  })

  it('does not replay when the same result is opened again', () => {
    openFresh(fx.refuted).unmount()
    render(<CheckPage id={fx.refuted.analysis_id} />)
    expect(screen.getByRole('main')).toHaveAttribute('data-reveal', 'off')
  })
})

describe('technical details', () => {
  it('stay out of the page until opened, then show the model’s own numbers', async () => {
    const user = userEvent.setup()
    openFresh(fx.refuted)
    const toggle = screen.getByRole('button', { name: 'View technical verification details' })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.queryByText('Claim verification')).not.toBeInTheDocument()

    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText('12 out of 100')).toBeInTheDocument()
    expect(screen.getByText('Claim verification')).toBeInTheDocument()
    expect(screen.getByText('−38.0')).toBeInTheDocument()
    expect(screen.getByText('91%')).toBeInTheDocument()
    expect(screen.getByText(/Lowered by claim verification/)).toBeInTheDocument()
  })
})

describe('feedback', () => {
  it('saves a Yes at once and thanks the reader', async () => {
    const user = userEvent.setup()
    openFresh(fx.refuted)
    await user.click(screen.getByRole('button', { name: 'Yes' }))

    expect(sendFeedbackMock).toHaveBeenCalledWith({ analysis_id: 'aaa111aaa111', agrees: true })
    expect(screen.getByText('Thanks. Your answer was saved.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Yes' })).not.toBeInTheDocument()
  })

  it('puts the question back, with the reason, when the answer could not be saved', async () => {
    sendFeedbackMock.mockRejectedValueOnce(new ApiError('unreachable', 'network'))
    const user = userEvent.setup()
    openFresh(fx.refuted)
    await user.click(screen.getByRole('button', { name: 'Yes' }))

    expect(await screen.findByText('We could not save your answer. Please try again.')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Yes' })).toBeInTheDocument()
  })

  it('asks what the reader thinks is right only after a No', async () => {
    const user = userEvent.setup()
    openFresh(fx.refuted)
    expect(screen.getByLabelText('Anything to add? (optional)').closest('[inert]')).not.toBeNull()

    await user.click(screen.getByRole('button', { name: 'No' }))
    expect(sendFeedbackMock).not.toHaveBeenCalled()
    await user.click(screen.getByRole('radio', { name: 'Mixed or partly true' }))
    await user.type(screen.getByLabelText('Anything to add? (optional)'), 'Half of it is right.')
    await user.click(screen.getByRole('button', { name: 'Send' }))

    expect(sendFeedbackMock).toHaveBeenCalledWith({
      analysis_id: 'aaa111aaa111',
      agrees: false,
      correct_label: 'mixed',
      comment: 'Half of it is right.',
    })
    expect(screen.getByText('Thanks. Your answer was saved.')).toBeInTheDocument()
  })

  it('does not ask again for a check already answered in this visit', async () => {
    const user = userEvent.setup()
    openFresh(fx.refuted)
    await user.click(screen.getByRole('button', { name: 'Yes' }))
    screen.getByRole('main') // still mounted
    const again = render(<CheckPage id={fx.refuted.analysis_id} />)
    expect(within(again.container).queryByRole('button', { name: 'Yes' })).not.toBeInTheDocument()
  })
})

describe('a result opened from a link', () => {
  it('fetches the stored check and shows it at rest', async () => {
    getAnalysisMock.mockResolvedValue({ ...fx.supported, cached: true })
    render(<CheckPage id="bbb222bbb222" />)

    expect(await screen.findByRole('heading', { level: 1, name: 'Well corroborated' })).toBeInTheDocument()
    expect(getAnalysisMock).toHaveBeenCalledWith('bbb222bbb222')
    expect(screen.getByRole('main')).toHaveAttribute('data-reveal', 'off')
    // nothing was typed in this visit, so the first claim stands in for the input
    expect(screen.getAllByText('ISRO successfully launches Chandrayaan-3 lunar mission.').length).toBeGreaterThan(1)
    expect(screen.queryByRole('button', { name: 'Check again' })).not.toBeInTheDocument()
  })

  it('says plainly when the check does not exist and points back to the desk', async () => {
    getAnalysisMock.mockRejectedValue(new ApiError('That check could not be found.', 'not_found', 404))
    render(<CheckPage id="zzz" />)

    expect(await screen.findByRole('heading', { level: 1, name: 'We could not find that check.' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Go to the desk' })).toHaveAttribute('href', '/')
  })

  it('offers a retry when the service cannot be reached', async () => {
    getAnalysisMock.mockRejectedValueOnce(new ApiError('We could not reach the checking service.', 'network'))
    getAnalysisMock.mockResolvedValueOnce(fx.supported)
    const user = userEvent.setup()
    render(<CheckPage id="bbb222bbb222" />)

    expect(await screen.findByRole('alert')).toHaveTextContent('We could not reach the checking service.')
    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByRole('heading', { level: 1, name: 'Well corroborated' })).toBeInTheDocument()
  })
})

describe('a stored answer', () => {
  it('says when it was checked and can run the check again', async () => {
    analyzeMock.mockResolvedValue({ ...fx.supported, analysis_id: 'new999new999', cached: false })
    const user = userEvent.setup()
    openFresh({ ...fx.supported, cached: true }, { text: 'ISRO successfully launches Chandrayaan-3 lunar mission.' })

    expect(screen.getByText(/Checked on 12 Nov 2026/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Check again' }))

    expect(analyzeMock).toHaveBeenCalledWith(
      { text: 'ISRO successfully launches Chandrayaan-3 lunar mission.', refresh: true },
      expect.anything(),
    )
    await waitFor(() => expect(window.location.pathname).toBe('/check/new999new999'))
  })
})
