import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { ApiError } from './api/client'
import { clearDraft } from './components/Desk'
import * as recent from './lib/recent'
import * as session from './lib/session'
import * as fx from './translate/__fixtures__/reports'

vi.mock('./api/client', async (original) => ({
  ...(await original<typeof import('./api/client')>()),
  analyze: vi.fn(),
  getAnalysis: vi.fn(),
  deleteAnalysis: vi.fn(),
  getHealth: vi.fn(),
}))
const client = await import('./api/client')
const getAnalysisMock = vi.mocked(client.getAnalysis)
const deleteAnalysisMock = vi.mocked(client.deleteAnalysis)
const getHealthMock = vi.mocked(client.getHealth)

const ITEM: recent.RecentItem = {
  id: 'aaa111aaa111',
  preview: 'Reserve Bank of India announces new Rs 1000 currency notes.',
  label: 'Contradicted by evidence',
  tone: 'bad',
  at: '2026-11-12T10:00:00+00:00',
}

function at(path: string) {
  window.history.replaceState(null, '', path)
  return render(<App />)
}

const recentButton = () => screen.getByRole('button', { name: 'Recent checks' })
const drawer = () => screen.getByRole('dialog', { name: 'Recent checks', hidden: true })

beforeEach(() => {
  clearDraft()
  deleteAnalysisMock.mockResolvedValue(undefined)
  getHealthMock.mockResolvedValue({ status: 'ok', explainer: 'templates only (EXPLAIN_ENABLED=0)' })
})
afterEach(() => {
  vi.resetAllMocks()
  session.clear()
})

describe('the app shell', () => {
  it('opens on the desk, with the name linking home and two quiet links on the right', () => {
    at('/')
    expect(screen.getByRole('heading', { level: 1, name: 'Check a story before you share it.' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'News Credibility' })).toHaveAttribute('href', '/')
    expect(screen.getByRole('link', { name: 'How it works' })).toHaveAttribute('href', '/how-it-works')
    expect(recentButton()).toBeInTheDocument()
  })

  it('shows no history, status or model information on the main screen', () => {
    recent.add(ITEM)
    at('/')
    const main = screen.getByRole('main')
    expect(within(main).queryByText(ITEM.preview)).not.toBeInTheDocument()
    expect(main.textContent).not.toMatch(/API online|model version|analyses stored/i)
    expect(within(main).queryByRole('table')).not.toBeInTheDocument()
  })

  it('moves between pages without reloading', async () => {
    const user = userEvent.setup()
    at('/')
    await user.click(screen.getByRole('link', { name: 'How it works' }))
    expect(window.location.pathname).toBe('/how-it-works')
    expect(screen.getByRole('heading', { level: 1, name: 'How it works' })).toBeInTheDocument()

    await user.click(screen.getByRole('link', { name: 'News Credibility' }))
    expect(screen.getByRole('heading', { level: 1, name: 'Check a story before you share it.' })).toBeInTheDocument()
  })

  it('follows the browser’s Back button', async () => {
    const user = userEvent.setup()
    at('/')
    await user.click(screen.getByRole('link', { name: 'How it works' }))
    act(() => {
      window.history.replaceState(null, '', '/')
      window.dispatchEvent(new PopStateEvent('popstate'))
    })
    expect(screen.getByRole('heading', { level: 1, name: 'Check a story before you share it.' })).toBeInTheDocument()
  })

  it('opens a stored check at its own address', async () => {
    getAnalysisMock.mockResolvedValue(fx.refuted)
    at('/check/aaa111aaa111')
    expect(await screen.findByRole('heading', { level: 1, name: 'Contradicted by evidence' })).toBeInTheDocument()
  })

  it('says so when an address leads nowhere', () => {
    at('/nope')
    expect(screen.getByRole('heading', { level: 1, name: 'There is nothing at this address.' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Go to the desk' })).toHaveAttribute('href', '/')
  })
})

describe('recent checks', () => {
  it('stays shut until asked for, then lists this browser’s checks as links', async () => {
    recent.add(ITEM)
    const user = userEvent.setup()
    at('/')
    expect(drawer()).toHaveAttribute('inert')
    expect(recentButton()).toHaveAttribute('aria-expanded', 'false')

    await user.click(recentButton())
    expect(drawer()).not.toHaveAttribute('inert')
    expect(recentButton()).toHaveAttribute('aria-expanded', 'true')
    const row = within(drawer()).getByRole('link', { name: /Reserve Bank of India announces/ })
    expect(row).toHaveAttribute('href', '/check/aaa111aaa111')
    expect(within(drawer()).getByText('Contradicted by evidence')).toBeInTheDocument()
    expect(within(drawer()).getByText('12 Nov 2026')).toBeInTheDocument()
    expect(drawer().contains(document.activeElement)).toBe(true)
  })

  it('closes on Escape and gives focus back to the button that opened it', async () => {
    const user = userEvent.setup()
    at('/')
    await user.click(recentButton())
    await user.keyboard('{Escape}')

    expect(drawer()).toHaveAttribute('inert')
    await waitFor(() => expect(recentButton()).toHaveFocus())
  })

  it('closes when the page behind it is clicked, and when a check is opened from it', async () => {
    recent.add(ITEM)
    getAnalysisMock.mockResolvedValue(fx.refuted)
    const user = userEvent.setup()
    at('/')
    await user.click(recentButton())
    fireEvent.click(screen.getByTestId('drawer-scrim'))
    expect(drawer()).toHaveAttribute('inert')

    await user.click(recentButton())
    await user.click(within(drawer()).getByRole('link', { name: /Reserve Bank of India announces/ }))
    expect(drawer()).toHaveAttribute('inert')
    expect(window.location.pathname).toBe('/check/aaa111aaa111')
  })

  it('explains an empty list', async () => {
    const user = userEvent.setup()
    at('/')
    await user.click(recentButton())
    expect(within(drawer()).getByText('Checks you run on this device will appear here.')).toBeInTheDocument()
  })

  it('still works, and says why the list is empty, when the browser refuses storage', async () => {
    const blocked = () => {
      throw new DOMException('denied', 'SecurityError')
    }
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(blocked)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(blocked)
    const user = userEvent.setup()
    at('/')
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument()

    await user.click(recentButton())
    expect(
      within(drawer()).getByText('This browser is not keeping a list, which happens in private windows.'),
    ).toBeInTheDocument()
  })
})

describe('removing a recent check', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    recent.add(ITEM)
  })
  afterEach(() => vi.useRealTimers())

  function openDrawer() {
    at('/')
    fireEvent.click(recentButton())
    fireEvent.click(within(drawer()).getByRole('button', { name: /^Delete the check/ }))
  }

  it('takes the row away at once but waits, offering Undo, before deleting anything', async () => {
    openDrawer()
    expect(within(drawer()).queryByRole('link', { name: /Reserve Bank/ })).not.toBeInTheDocument()
    expect(within(drawer()).getByRole('status')).toHaveTextContent('Check deleted.')
    expect(deleteAnalysisMock).not.toHaveBeenCalled()

    fireEvent.click(within(drawer()).getByRole('button', { name: 'Undo' }))
    expect(within(drawer()).getByRole('link', { name: /Reserve Bank/ })).toBeInTheDocument()
    await act(() => vi.advanceTimersByTimeAsync(10_000))
    expect(deleteAnalysisMock).not.toHaveBeenCalled()
    expect(recent.list()).toHaveLength(1)
  })

  it('deletes the stored check once the Undo window has passed', async () => {
    openDrawer()
    await act(() => vi.advanceTimersByTimeAsync(4_999))
    expect(deleteAnalysisMock).not.toHaveBeenCalled()
    await act(() => vi.advanceTimersByTimeAsync(1))

    expect(deleteAnalysisMock).toHaveBeenCalledWith('aaa111aaa111')
    expect(recent.list()).toHaveLength(0)
    expect(within(drawer()).queryByRole('button', { name: 'Undo' })).not.toBeInTheDocument()
  })

  it('does not wait for the Undo window when the drawer is closed first', async () => {
    openDrawer()
    fireEvent.keyDown(document, { key: 'Escape' })
    await act(() => vi.advanceTimersByTimeAsync(0))
    expect(deleteAnalysisMock).toHaveBeenCalledWith('aaa111aaa111')
  })

  it('drops a check the server no longer has', async () => {
    deleteAnalysisMock.mockRejectedValue(new ApiError('gone', 'not_found', 404))
    openDrawer()
    await act(() => vi.advanceTimersByTimeAsync(5_000))
    expect(recent.list()).toHaveLength(0)
    expect(within(drawer()).queryByRole('alert')).not.toBeInTheDocument()
  })

  it('puts the row back and says so when the delete fails', async () => {
    deleteAnalysisMock.mockRejectedValue(new ApiError('unreachable', 'network'))
    openDrawer()
    await act(() => vi.advanceTimersByTimeAsync(5_000))

    expect(within(drawer()).getByRole('link', { name: /Reserve Bank/ })).toBeInTheDocument()
    expect(within(drawer()).getByRole('alert')).toHaveTextContent('We could not delete that check. Please try again.')
    expect(recent.list()).toHaveLength(1)
  })
})

describe('How it works', () => {
  it('says what is checked, what is not, and what happens to what you submit', async () => {
    at('/how-it-works')
    expect(screen.getByRole('heading', { level: 1, name: 'How it works' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'What we check' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'What we do not judge' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'What the verdicts mean' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Limits' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Your privacy' })).toBeInTheDocument()
    expect(screen.getByText(/how a story is worded is never judged/i)).toBeInTheDocument()
    expect(screen.getByText(/kept only in your browser/i)).toBeInTheDocument()
  })

  it('does not mention an outside model provider on a server that uses none', async () => {
    at('/how-it-works')
    await waitFor(() => expect(getHealthMock).toHaveBeenCalled())
    expect(screen.queryByText(/language model/i)).not.toBeInTheDocument()
  })

  it('discloses the outside model provider on a server that uses one', async () => {
    getHealthMock.mockResolvedValue({ status: 'ok', explainer: 'llm via http://localhost:3001/v1 (model: auto)' })
    at('/how-it-works')
    expect(
      await screen.findByText(/also sent to an outside language model provider/i),
    ).toBeInTheDocument()
  })
})
