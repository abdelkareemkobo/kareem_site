import fs from 'node:fs/promises'
import path from 'node:path'
import { gzipSync } from 'node:zlib'
import { fromHtml } from 'hast-util-from-html'
import { visit } from 'unist-util-visit'
import { root } from './site.mjs'

const routes = JSON.parse(await fs.readFile(path.join(root, 'migration/legacy-urls.json'), 'utf8'))
const output = process.argv[2] ?? 'migration/seo-after.json'
const pageMetadata = {}
async function audit(dir, key) {
  const result = { pages: 0, canonical: 0, descriptions: 0, socialImages: 0, structuredData: 0, siteNameSchema: 0, articleSchema: 0, missingImageDimensions: 0, images: 0, htmlBytes: 0, homeResources: [] }
  for (const route of routes) {
    const relative = route[key] === '/' ? 'index.html' : decodeURI(route[key]).slice(1)
    let html
    try { html = await fs.readFile(path.join(dir, relative), 'utf8') } catch { continue }
    result.pages++; result.htmlBytes += Buffer.byteLength(html)
    const flags = new Set()
    const metadata = {}
    visit(fromHtml(html), 'element', node => {
      const p = node.properties
      if (node.tagName === 'title' && !metadata.title) metadata.title = node.children.map(c => c.value ?? '').join('')
      if (node.tagName === 'meta' && ['description', 'og:image', 'og:title'].includes(p.name ?? p.property)) metadata[p.name ?? p.property] = p.content
      if (node.tagName === 'link' && String(p.rel).includes('canonical')) flags.add('canonical')
      if (node.tagName === 'meta' && p.name === 'description' && p.content) flags.add('descriptions')
      if (node.tagName === 'meta' && p.property === 'og:image' && p.content) flags.add('socialImages')
      if (node.tagName === 'script' && p.type === 'application/ld+json') {
        try {
          const schema = JSON.parse(node.children.map(c => c.value ?? '').join(''))
          flags.add('structuredData')
          const types = JSON.stringify(schema)
          if (types.includes('WebSite')) flags.add('siteNameSchema')
          if (types.includes('BlogPosting') || types.includes('Article')) flags.add('articleSchema')
        } catch {}
      }
      if (node.tagName === 'img') { result.images++; if (!p.width || !p.height) result.missingImageDimensions++ }
      if (relative === 'index.html' && ((node.tagName === 'script' && p.src) || (node.tagName === 'link' && String(p.rel).includes('stylesheet')))) result.homeResources.push(p.src ?? p.href)
    })
    for (const flag of flags) result[flag]++
    pageMetadata[route.current] ??= {}
    pageMetadata[route.current][key] = metadata
  }
  result.sharedAssets = {}
  for (const file of ['index.css', 'postscript.js', 'static/contentIndex.json', 'static/navigationIndex.json', 'search.json', 'index.html']) {
    try { const bytes = await fs.readFile(path.join(dir, file)); result.sharedAssets[file] = { bytes: bytes.length, gzip: gzipSync(bytes).length } } catch {}
  }
  return result
}
const report = { measuredAt: new Date().toISOString(), method: 'Static output inspection; gzip estimates, not browser timing or field Core Web Vitals.', oldQuarto: await audit(path.join(root, '../Karem Site/_site'), 'old'), quartz: await audit(path.join(root, 'public'), 'current') }
const normalize = text => String(text).replace(/[‘’]/g, "'").replace(/[“”]/g, '"').replace(/\s+/g, ' ').trim()
report.descriptionsPreservedIgnoringTypography = Object.values(pageMetadata).filter(row => row.old && row.current && normalize(row.old.description) === normalize(row.current.description)).length
await fs.writeFile(path.join(root, 'migration/seo-metadata-comparison.json'), JSON.stringify(pageMetadata, null, 2) + '\n')
await fs.writeFile(path.resolve(root, output), JSON.stringify(report, null, 2) + '\n')
console.log(JSON.stringify(report, null, 2))
