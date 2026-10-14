import fs from 'node:fs/promises'
import path from 'node:path'
import assert from 'node:assert/strict'
import { fromHtml } from 'hast-util-from-html'
import { visit } from 'unist-util-visit'
import { slugifyFilePath } from '@quartz-community/utils'
import { root, files } from './site.mjs'
import matter from 'gray-matter'

process.chdir(root)
const manifest = JSON.parse(await fs.readFile('.quartz/publish-manifest.json', 'utf8'))
const publicFiles = new Set((await files(path.join(root, 'public'))).map(f => path.relative(path.join(root, 'public'), f)))
const failures = []
const brokenLinks = []
for (const note of manifest.notes) {
  const file = slugifyFilePath(note) + '.html'
  const html = await fs.readFile(path.join('public', file), 'utf8')
  const tree = fromHtml(html)
  const data = matter(await fs.readFile(path.join('.publish-content', note), 'utf8')).data
  const expectedURL = file === 'index.html' ? 'https://kareemai.com/' : `https://kareemai.com/${file}`
  let canonical = 0, description = 0, schema = 0
  visit(tree, 'element', node => {
    const p = node.properties
    if (node.tagName === 'link' && String(p.rel).includes('canonical')) { canonical++; if (p.href !== expectedURL) failures.push(`${file}: canonical URL differs from published URL`) }
    if (node.tagName === 'meta' && p.name === 'description') description++
    if (node.tagName === 'script' && p.type === 'application/ld+json') {
      schema++
      try {
        const parsed = JSON.parse(node.children.map(c => c.value ?? '').join(''))
        const graph = parsed['@graph'] ?? [parsed]
        if (file === 'index.html' && !graph.some(n => n['@type'] === 'WebSite')) failures.push('Missing WebSite schema')
        if (data.date) {
          const article = graph.find(n => n['@type'] === 'BlogPosting')
          if (!article || article.headline !== data.title || Number.isNaN(Date.parse(article.datePublished)) || article.mainEntityOfPage?.['@id'] !== expectedURL + '#webpage') failures.push(`${file}: invalid article identity/date/schema`)
        }
      } catch { failures.push(`${file}: invalid JSON-LD`) }
    }
    if (node.tagName === 'meta' && p.name === 'robots' && String(p.content).includes('noindex')) failures.push(`${file}: published note must be indexable`)
    if (node.tagName === 'img' && typeof p.src === 'string' && !/^(?:https?:|data:|\/\/)/.test(p.src) && (!p.width || !p.height)) failures.push(`${file}: local image needs dimensions`)
    if (typeof p.srcSet === 'string') for (const candidate of p.srcSet.split(', ')) {
      const ref = candidate.split(' ')[0]
      const url = new URL(ref, 'https://preview.local/' + file)
      if (url.origin === 'https://preview.local' && !publicFiles.has(decodeURIComponent(url.pathname.slice(1)))) failures.push(`${file}: missing responsive image ${ref}`)
    }
    const ref = ['img', 'iframe', 'video', 'audio'].includes(node.tagName) ? p.src : node.tagName === 'a' ? p.href : null
    if (typeof ref !== 'string' || /^(?:[a-z][a-z\d+.-]*:|#|\/\/)/i.test(ref)) return
    const url = new URL(ref, 'https://preview.local/' + file)
    let target = decodeURIComponent(url.pathname.slice(1))
    const candidates = [target, target + '.html', target.replace(/\/$/, '') + '/index.html', target + 'index.html']
    if (!candidates.some(f => publicFiles.has(f))) {
      const issue = `${file}: ${ref}`
      if (node.tagName !== 'a') failures.push(`Missing embedded asset ${issue}`)
      else brokenLinks.push(issue)
    }
  })
  if (canonical !== 1 || description !== 1) failures.push(`${file}: expected one canonical and description`)
  if (schema !== 1) failures.push(`${file}: expected one structured data graph`)
  if (html.includes('::: {.')) failures.push(`${file}: leftover Quarto fences`)
}
for (const name of ['sitemap.xml', 'blog/feed.xml', 'robots.txt', 'CNAME']) assert(publicFiles.has(name), `Missing ${name}`)
const blogFeed = await fs.readFile('public/blog/feed.xml', 'utf8')
const tilFeed = await fs.readFile('public/til/index.xml', 'utf8')
assert(!blogFeed.includes('<link>https://kareemai.com/til/'), 'Blog feed must exclude TIL notes')
assert(!tilFeed.includes('<link>https://kareemai.com/blog/'), 'TIL feed must exclude blog articles')
assert(blogFeed.includes('content:encoded'), 'RSS must contain full article content')
for (const route of JSON.parse(await fs.readFile('migration/legacy-urls.json', 'utf8'))) {
  const current = 'https://kareemai.com' + route.current
  const original = 'https://kareemai.com' + route.old
  const item = [...(blogFeed + tilFeed).matchAll(/<item>[\s\S]*?<\/item>/g)].find(m => m[0].includes(`<link>${current}</link>`))
  if (item) assert(item[0].includes(`<guid isPermaLink="true">${original}</guid>`), `RSS identity changed for ${route.old}`)
}
const redirects = await fs.readFile('public/_redirects', 'utf8')
for (const route of JSON.parse(await fs.readFile('migration/urls.json', 'utf8'))) {
  if (route.redirect) assert(redirects.includes(`${encodeURI(route.old)} ${encodeURI(route.current)} 301`), `Missing redirect for ${route.old}`)
  if (route.redirect && route.old.endsWith('.html') && route.old !== '/index.html') {
    const page = await fs.readFile(path.join('public', decodeURI(route.old).slice(1)), 'utf8')
    assert(page.includes('http-equiv="refresh"') && page.includes('noindex,follow') && page.includes('https://kareemai.com' + encodeURI(route.current)), `Missing GitHub Pages redirect for ${route.old}`)
  }
}
const home = await fs.readFile('public/index.html', 'utf8')
const blogEntry = await fs.readFile('public/blog/index.html', 'utf8')
assert.match(blogEntry, /http-equiv="refresh"/, 'Blog directory must open the actual feed')
assert.match(blogEntry, /url=\/blog\/feed\.html/, 'Blog directory must target the actual feed')
const folder = await fs.readFile('public/blog/posts/index.html', 'utf8')
assert.equal((folder.match(/class="page-listing"/g) ?? []).length, 1, 'Generated folder listings must render only once')
assert(home.includes('تصفح الملاحظات') && home.includes('روابط واردة'), 'Site Arabic labels must use the corrected locale')
assert(!home.includes('katex@'), 'Pages without math must not load KaTeX')
assert(!home.includes('@layer obsidian-theme {'), 'Things CSS must be cached separately, not embedded in every page')
assert.match(home, /href="\/static\/things-[a-f0-9]+\.css"/, 'Things stylesheet must be loaded')
const navigation = JSON.parse(await fs.readFile('public/static/navigationIndex.json', 'utf8'))
const search = JSON.parse(await fs.readFile('public/static/contentIndex.json', 'utf8'))
assert.equal(Object.keys(navigation).length, manifest.notes.length)
assert(Object.values(navigation).every(n => !('content' in n) && !('richContent' in n)), 'Navigation must not download article bodies')
assert(Object.values(search).every(n => !('richContent' in n)), 'Search must not duplicate full HTML')
assert(Object.values(search).some(n => n.content?.length > 100), 'Full-text search must retain article text')
const postscript = await fs.readFile('public/postscript.js', 'utf8')
assert(!['No results.', 'Search results', 'Try another search term?', 'Global Graph'].some(s => postscript.includes(s)), 'Search/graph interface must be translated')
assert.match(home, /lang="ar" dir="rtl"/)
const english = await fs.readFile('public/' + slugifyFilePath('blog/posts/life_style/why_I_am_blogging.md') + '.html', 'utf8')
assert.match(english, /lang="en" dir="ltr"/)
await fs.writeFile('migration/link-audit.json', JSON.stringify({ brokenLinks, failures }, null, 2) + '\n')
if (failures.length || brokenLinks.length) throw new Error(JSON.stringify({ failures, brokenLinks }, null, 2))
console.log(`Verified ${manifest.notes.length} notes: images, local links, canonical metadata, JSON-LD, Arabic/English direction, sitemap, RSS, and publishing isolation.`)
