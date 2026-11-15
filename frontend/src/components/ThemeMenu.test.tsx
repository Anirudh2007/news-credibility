import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { THEME_KEY } from '../lib/theme'
import { ThemeMenu } from './ThemeMenu'

const trigger = () => screen.getByRole('button', { name: /^Theme/ })
const menu = () => screen.getByRole('menu', { name: 'Theme', hidden: true })
const option = (name: string) => within(menu()).getByRole('menuitemradio', { name, hidden: true })

describe('theme menu', () => {
  it('is one quiet button until opened, and says which theme is in use', () => {
    render(<ThemeMenu />)
    expect(trigger()).toHaveAccessibleName('Theme: System')
    expect(trigger()).toHaveAttribute('aria-expanded', 'false')
    expect(menu()).toHaveAttribute('inert')
  })

  it('offers System, Light and Dark, with System chosen by default', async () => {
    const user = userEvent.setup()
    render(<ThemeMenu />)
    await user.click(trigger())

    expect(trigger()).toHaveAttribute('aria-expanded', 'true')
    expect(menu()).not.toHaveAttribute('inert')
    expect(within(menu()).getAllByRole('menuitemradio').map((o) => o.textContent)).toEqual(['System', 'Light', 'Dark'])
    expect(option('System')).toHaveAttribute('aria-checked', 'true')
    expect(option('Light')).toHaveAttribute('aria-checked', 'false')
    expect(option('System')).toHaveFocus() // focus lands on the current choice
  })

  it('applies a choice at once, remembers it, closes, and gives focus back to the button', async () => {
    const user = userEvent.setup()
    render(<ThemeMenu />)
    await user.click(trigger())
    await user.click(option('Dark'))

    expect(document.documentElement.dataset.theme).toBe('dark')
    expect(window.localStorage.getItem(THEME_KEY)).toBe('dark')
    expect(menu()).toHaveAttribute('inert')
    expect(trigger()).toHaveFocus()
    expect(trigger()).toHaveAccessibleName('Theme: Dark')
    expect(option('Dark')).toHaveAttribute('aria-checked', 'true')
  })

  it('shows a remembered choice when the page is opened again', () => {
    window.localStorage.setItem(THEME_KEY, 'light')
    render(<ThemeMenu />)
    expect(trigger()).toHaveAccessibleName('Theme: Light')
    expect(option('Light')).toHaveAttribute('aria-checked', 'true')
  })

  it('works from the keyboard: arrows move between options, Enter chooses, Escape closes', async () => {
    const user = userEvent.setup()
    render(<ThemeMenu />)
    trigger().focus()
    await user.keyboard('{Enter}')
    expect(option('System')).toHaveFocus()

    await user.keyboard('{ArrowDown}')
    expect(option('Light')).toHaveFocus()
    await user.keyboard('{ArrowUp}{ArrowUp}')
    expect(option('Dark')).toHaveFocus() // wraps round
    await user.keyboard('{Escape}')
    expect(menu()).toHaveAttribute('inert')
    expect(trigger()).toHaveFocus()
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false) // Escape chose nothing

    await user.keyboard('{Enter}{ArrowDown}{Enter}')
    expect(document.documentElement.dataset.theme).toBe('light')
  })

  it('closes when the reader clicks elsewhere on the page', async () => {
    const user = userEvent.setup()
    render(
      <div>
        <ThemeMenu />
        <p>Elsewhere</p>
      </div>,
    )
    await user.click(trigger())
    await user.click(screen.getByText('Elsewhere'))
    expect(menu()).toHaveAttribute('inert')
  })
})
