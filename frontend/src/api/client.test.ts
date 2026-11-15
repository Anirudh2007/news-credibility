import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, analyze, deleteAnalysis, getAnalysis, sendFeedback } from './client'

function respond(status: number, body?: unknown): Response {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

function mockFetch(impl: (url: string, init?: RequestInit) => Promise<Response>) {
  const spy = vi.fn(impl)
  vi.stubGlobal('fetch', spy)
  return spy
}

afterEach(() => vi.unstubAllGlobals())

describe('analyze', () => {
  it('posts a multipart form that always asks for the explanation and omits empty fields', async () => {
    const fetchSpy = mockFetch(async () => respond(200, { analysis_id: 'abc', band: 'Low' }))
    const report = await analyze({ text: 'RBI announces new notes.', claimedLocation: '' })

    expect(report.analysis_id).toBe('abc')
    const [url, init] = fetchSpy.mock.calls[0]
    expect(url).toBe('/api/v1/analyze')
    expect(init?.method).toBe('POST')
    const form = init?.body as FormData
    expect(form.get('text')).toBe('RBI announces new notes.')
    expect(form.get('explain')).toBe('true')
    expect(form.get('refresh')).toBe('false')
    expect(form.has('url')).toBe(false)
    expect(form.has('claimed_location')).toBe(false)
    expect(form.has('image')).toBe(false)
  })

  it('sends the link, the image, the optional details and the refresh flag under the API field names', async () => {
    const fetchSpy = mockFetch(async () => respond(200, { analysis_id: 'abc', band: 'High' }))
    const image = new File(['x'], 'photo.png', { type: 'image/png' })
    await analyze({
      url: 'https://example.com/a',
      image,
      claimedDate: '2024-07-03',
      claimedLocation: 'India Gate',
      publishDate: '2024-07-04',
      refresh: true,
    })

    const form = fetchSpy.mock.calls[0][1]?.body as FormData
    expect(form.get('url')).toBe('https://example.com/a')
    expect((form.get('image') as File).name).toBe('photo.png')
    expect(form.get('claimed_date')).toBe('2024-07-03')
    expect(form.get('claimed_location')).toBe('India Gate')
    expect(form.get('publish_date')).toBe('2024-07-04')
    expect(form.get('refresh')).toBe('true')
  })

  it("reports a refusal in the API's own words", async () => {
    mockFetch(async () => respond(422, { detail: 'This does not look like English text. Nothing was analysed.' }))
    const error = await analyze({ text: 'asdf' }).catch((e) => e)

    expect(error).toBeInstanceOf(ApiError)
    expect(error.kind).toBe('refused')
    expect(error.status).toBe(422)
    expect(error.message).toBe('This does not look like English text. Nothing was analysed.')
  })

  it('joins a list of validation errors into one readable message', async () => {
    mockFetch(async () =>
      respond(422, {
        detail: [
          { loc: ['body', 'claimed_date'], msg: 'invalid date' },
          { loc: ['body', 'url'], msg: 'too long' },
        ],
      }),
    )
    const error = await analyze({ text: 'x' }).catch((e) => e)
    expect(error.message).toBe('claimed_date: invalid date; url: too long')
  })

  it('treats an unsupported or oversize image as a refusal', async () => {
    mockFetch(async () => respond(413, { detail: 'image larger than 10 MB' }))
    const error = await analyze({ text: 'x' }).catch((e) => e)
    expect(error.kind).toBe('refused')
    expect(error.message).toBe('image larger than 10 MB')
  })

  it('says the service could not be reached when the request never completes', async () => {
    mockFetch(async () => {
      throw new TypeError('Failed to fetch')
    })
    const error = await analyze({ text: 'x' }).catch((e) => e)

    expect(error.kind).toBe('network')
    expect(error.status).toBeNull()
    expect(error.message).toBe(
      'We could not reach the checking service. Nothing was lost: try again in a moment.',
    )
  })

  it('reports a server failure without leaking a stack trace or HTML', async () => {
    mockFetch(async () => new Response('<html>502 Bad Gateway</html>', { status: 502 }))
    const error = await analyze({ text: 'x' }).catch((e) => e)

    expect(error.kind).toBe('server')
    expect(error.message).toBe('The checking service had a problem. Nothing was lost: try again in a moment.')
  })

  it('lets a cancelled check surface as an abort, not as a network failure', async () => {
    mockFetch(async (_url, init) => {
      init?.signal?.throwIfAborted()
      return respond(200, { analysis_id: 'abc', band: 'Low' })
    })
    const controller = new AbortController()
    controller.abort()
    const error = await analyze({ text: 'x' }, controller.signal).catch((e) => e)
    expect(error.name).toBe('AbortError')
  })
})

describe('stored analyses', () => {
  it('fetches one analysis by id', async () => {
    const fetchSpy = mockFetch(async () => respond(200, { analysis_id: 'abc123', band: 'High' }))
    const report = await getAnalysis('abc123')
    expect(fetchSpy.mock.calls[0][0]).toBe('/api/v1/analyses/abc123')
    expect(report.band).toBe('High')
  })

  it('marks a missing analysis as not found', async () => {
    mockFetch(async () => respond(404, { detail: 'no analysis zzz' }))
    const error = await getAnalysis('zzz').catch((e) => e)
    expect(error.kind).toBe('not_found')
  })

  it('deletes an analysis and accepts the empty 204 answer', async () => {
    const fetchSpy = mockFetch(async () => respond(204))
    await expect(deleteAnalysis('abc123')).resolves.toBeUndefined()
    expect(fetchSpy.mock.calls[0][1]?.method).toBe('DELETE')
  })
})

describe('sendFeedback', () => {
  it('posts the reader’s answer as JSON', async () => {
    const fetchSpy = mockFetch(async () => respond(201, { id: 1 }))
    await sendFeedback({ analysis_id: 'abc123', agrees: false, correct_label: 'mixed', comment: 'Half true.' })

    const [url, init] = fetchSpy.mock.calls[0]
    expect(url).toBe('/api/v1/feedback')
    expect(JSON.parse(init?.body as string)).toEqual({
      analysis_id: 'abc123',
      agrees: false,
      correct_label: 'mixed',
      comment: 'Half true.',
    })
  })
})
