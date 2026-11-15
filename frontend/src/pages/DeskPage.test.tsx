import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiError } from '../api/client'
import type { Report } from '../api/types'
import { clearDraft } from '../components/Desk'
import * as session from '../lib/session'
import { STATUS_LINES } from '../lib/timing'
import { DeskPage } from './DeskPage'

vi.mock('../api/client', async (original) => ({
  ...(await original<typeof import('../api/client')>()),
  analyze: vi.fn(),
}))
const { analyze } = await import('../api/client')
const analyzeMock = vi.mocked(analyze)

const REPORT: Report = { analysis_id: 'abc123', band: 'High', score: 88, claims: [] }
const field = () => screen.getByRole('textbox', { name: /paste a link, a claim, or the text of an article/i })
const verify = () => screen.getByRole('button', { name: /^verify$/i })

beforeEach(() => {
  clearDraft()
  analyzeMock.mockResolvedValue(REPORT)
})
afterEach(() => {
  analyzeMock.mockReset()
  session.clear()
})

describe('the desk', () => {
  it('says what the tool is for and how to use it before anything is typed', () => {
    render(<DeskPage />)
    expect(screen.getByRole('heading', { level: 1, name: 'Check a story before you share it.' })).toBeInTheDocument()
    expect(screen.getByText('Submit a story or claim')).toBeInTheDocument()
    expect(screen.getByText('We cross-check the evidence')).toBeInTheDocument()
    expect(screen.getByText('Read the breakdown')).toBeInTheDocument()
  })

  it('asks for something to check instead of sending an empty request', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.click(verify())

    expect(analyzeMock).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent('Paste a link, a claim or some text first, or add an image.')
    expect(field()).toHaveFocus()
  })

  it('recognises a pasted link, says so, and sends it as a link', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.type(field(), '  thehindu.com/news/x ')
    expect(screen.getByText('Checking this link')).toBeInTheDocument()

    await user.click(verify())
    expect(analyzeMock).toHaveBeenCalledWith({ url: 'https://thehindu.com/news/x' }, expect.anything())
  })

  it('sends a claim as text and opens the result when the answer arrives', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.type(field(), 'ISRO launches Chandrayaan-3.')
    expect(screen.getByText('Checking this text')).toBeInTheDocument()

    await user.click(verify())
    expect(analyzeMock).toHaveBeenCalledWith({ text: 'ISRO launches Chandrayaan-3.' }, expect.anything())
    await waitFor(() => expect(window.location.pathname).toBe('/check/abc123'))
  })

  it('submits with Ctrl+Enter from the field', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.type(field(), 'A claim.{Control>}{Enter}{/Control}')
    expect(analyzeMock).toHaveBeenCalledTimes(1)
  })

  it('fills the field and runs the check when an example is chosen', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.click(screen.getByRole('button', { name: 'A real event' }))

    expect(field()).toHaveValue("Chandrayaan-3 landed near the Moon's south pole on 23 August 2023.")
    expect(analyzeMock).toHaveBeenCalledWith(
      { text: "Chandrayaan-3 landed near the Moon's south pole on 23 August 2023." },
      expect.anything(),
    )
  })

  it('offers exactly three examples', () => {
    render(<DeskPage />)
    const examples = screen.getByRole('group', { name: /try an example/i })
    expect(examples.querySelectorAll('button')).toHaveLength(3)
  })

  it('keeps the optional details out of the way until asked, then sends them', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    const toggle = screen.getByRole('button', { name: /add details/i })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')
    expect(screen.getByLabelText('Place it names').closest('[inert]')).not.toBeNull()

    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByLabelText('Place it names').closest('[inert]')).toBeNull()

    await user.type(field(), 'Flooding at India Gate this morning.')
    await user.type(screen.getByLabelText('Place it names'), 'India Gate')
    fireEvent.change(screen.getByLabelText('Date the story gives'), { target: { value: '2024-07-03' } })
    await user.type(screen.getByLabelText('Link to the source'), 'ndtv.com/x')
    await user.click(verify())

    expect(analyzeMock).toHaveBeenCalledWith(
      {
        text: 'Flooding at India Gate this morning.',
        url: 'https://ndtv.com/x',
        claimedLocation: 'India Gate',
        claimedDate: '2024-07-03',
      },
      expect.anything(),
    )
  })

  it('does not ask for a source link when the thing being checked is already a link', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.click(screen.getByRole('button', { name: /add details/i }))
    await user.type(field(), 'https://example.com/a')
    expect(screen.queryByLabelText('Link to the source')).not.toBeInTheDocument()
  })

  it('attaches an image and sends it with the text', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    const photo = new File(['x'], 'flood.png', { type: 'image/png' })
    await user.upload(screen.getByLabelText('Add an image'), photo)
    expect(screen.getByText('flood.png')).toBeInTheDocument()

    await user.type(field(), 'Flooding at India Gate.')
    await user.click(verify())
    expect(analyzeMock.mock.calls[0][0]).toMatchObject({ text: 'Flooding at India Gate.', image: photo })
  })

  it('lets an image be checked on its own, and removed again', async () => {
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.upload(screen.getByLabelText('Add an image'), new File(['x'], 'flood.png', { type: 'image/png' }))
    await user.click(screen.getByRole('button', { name: 'Remove flood.png' }))
    await waitFor(() => expect(screen.queryByText('flood.png')).not.toBeInTheDocument())

    await user.click(verify())
    expect(analyzeMock).not.toHaveBeenCalled()
  })

  it('refuses a file that is too large or not an image, and says which', async () => {
    render(<DeskPage />)
    const input = screen.getByLabelText('Add an image')

    const huge = new File(['x'], 'huge.png', { type: 'image/png' })
    Object.defineProperty(huge, 'size', { value: 11 * 1024 * 1024 })
    fireEvent.change(input, { target: { files: [huge] } })
    expect(screen.getByRole('alert')).toHaveTextContent('That image is larger than 10 MB. Try a smaller copy.')
    expect(screen.queryByText('huge.png')).not.toBeInTheDocument()

    fireEvent.change(input, { target: { files: [new File(['x'], 'notes.pdf', { type: 'application/pdf' })] } })
    expect(screen.getByRole('alert')).toHaveTextContent(
      'That file is not an image we can read. Use a JPEG, PNG, WebP, GIF or BMP.',
    )
  })

  it('shows why a check was refused, keeps what was typed and lets the reader try again', async () => {
    analyzeMock.mockRejectedValueOnce(
      new ApiError('This does not look like English text. Nothing was analysed.', 'refused', 422),
    )
    const user = userEvent.setup()
    render(<DeskPage />)
    await user.type(field(), 'asdf qwer zxcv')
    await user.click(verify())

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'This does not look like English text. Nothing was analysed.',
    )
    expect(field()).toHaveValue('asdf qwer zxcv')
    expect(field()).not.toHaveAttribute('readonly')
    expect(field()).toHaveFocus()
    expect(verify()).toBeEnabled()
  })

  it('keeps what was typed when the reader comes back to the desk', async () => {
    const user = userEvent.setup()
    const first = render(<DeskPage />)
    await user.type(field(), 'A half-written claim')
    first.unmount()

    render(<DeskPage />)
    expect(field()).toHaveValue('A half-written claim')
  })
})

describe('the desk while a check runs', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('locks the field at once, shows progress only after a moment, and can be cancelled', async () => {
    analyzeMock.mockImplementation(
      (_input, signal) =>
        new Promise((_resolve, reject) => {
          signal?.addEventListener('abort', () => reject(new DOMException('cancelled', 'AbortError')))
        }),
    )
    render(<DeskPage />)
    fireEvent.change(field(), { target: { value: 'A slow claim.' } })
    fireEvent.click(verify())

    // the press registers on the same frame; no spinner yet
    expect(field()).toHaveAttribute('readonly')
    expect(screen.getByRole('button', { name: /^verify$/i })).toBeDisabled()
    expect(screen.queryByText(STATUS_LINES[0].text)).not.toBeInTheDocument()

    await act(() => vi.advanceTimersByTimeAsync(300))
    expect(screen.getByRole('button', { name: /checking/i })).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent(STATUS_LINES[0].text)
    expect(screen.queryByText('Submit a story or claim')).not.toBeInTheDocument()

    await act(() => vi.advanceTimersByTimeAsync(2_000))
    expect(screen.getByRole('status')).toHaveTextContent(STATUS_LINES[1].text)

    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    await act(() => vi.advanceTimersByTimeAsync(0))
    expect(field()).not.toHaveAttribute('readonly')
    expect(field()).toHaveValue('A slow claim.')
    expect(verify()).toBeEnabled()
    expect(screen.getByText('Submit a story or claim')).toBeInTheDocument()
  })
})
