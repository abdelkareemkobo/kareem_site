import http from 'node:http'
import handler from 'serve-handler'
import fs from 'node:fs/promises'
const port = Number(process.env.PORT ?? 8088)
http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost')
    if (process.env.KAREEM_PREVIEW_LOG === '1') console.log(url.pathname)
    const redirects = await fs.readFile('public/_redirects', 'utf8').catch(() => '')
    const redirect = redirects.split('\n').map(line => line.split(/\s+/)).find(([from]) => from === url.pathname)
    if (redirect) { res.writeHead(parseInt(redirect[2], 10), { Location: redirect[1] + url.search }); res.end(); return }
    let pathname = decodeURIComponent(url.pathname)
    if (pathname.endsWith('/')) pathname += 'index.html'
    else {
      try { await fs.access('public' + pathname) }
      catch {
        try { await fs.access('public' + pathname + '.html'); pathname += '.html' }
        catch { pathname += '/index.html' }
      }
    }
    req.url = encodeURI(pathname) + url.search
    await handler(req, res, { public: 'public', cleanUrls: false, directoryListing: false })
  } catch { res.writeHead(500); res.end('Preview server error') }
}).listen(port, '127.0.0.1', () => console.log(`Quartz preview: http://localhost:${port}`))
