import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawn } from 'node:child_process'
import matter from 'gray-matter'
import { visit } from 'unist-util-visit'
import { slugifyFilePath } from '@quartz-community/utils'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { publishVault } from './publish-vault.mjs'
import { optimizeOutput } from './optimize-output.mjs'

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
export async function files(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const result = []
  for (const entry of entries) {
    if (entry.name.startsWith('.') || ['private', 'templates', 'node_modules'].includes(entry.name)) continue
    const file = path.join(dir, entry.name)
    if (entry.isDirectory()) result.push(...await files(file))
    else if (entry.isFile()) result.push(file)
  }
  return result
}

export const vaultRoot = path.resolve(process.env.KAREEM_VAULT ?? path.join(root, '../../../..'))
export async function stage() {
  let isVault = false
  try { await fs.access(path.join(vaultRoot, '.obsidian')); isVault = true } catch {}
  if (process.env.KAREEM_PUBLISHED_ONLY === '1' || (!isVault && !process.env.KAREEM_VAULT)) {
    // Hosting only needs the approved export, never access to the private vault.
    const exported = path.join(root, 'published')
    let manifest
    try { manifest = JSON.parse(await fs.readFile(path.join(exported, 'publish-manifest.json'), 'utf8')) }
    catch { throw new Error('No Obsidian vault or publishing export found. Run npm run export inside your vault before pushing this website repository, or set KAREEM_VAULT to the vault folder.') }
    await fs.rm(path.join(root, '.publish-content'), { recursive: true, force: true })
    await fs.cp(path.join(exported, 'content'), path.join(root, '.publish-content'), { recursive: true })
    await fs.mkdir(path.join(root, '.quartz'), { recursive: true })
    await fs.writeFile(path.join(root, '.quartz/publish-manifest.json'), JSON.stringify(manifest, null, 2))
    return manifest
  }
  return publishVault({ vaultRoot, projectRoot: root, target: path.join(root, '.publish-content'), manifestPath: path.join(root, '.quartz/publish-manifest.json') })
}

async function run(command, args) {
  await new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, stdio: 'inherit', env: process.env })
    child.on('error', reject)
    child.on('exit', code => code === 0 ? resolve() : reject(new Error(`${command} exited with ${code}`)))
  })
}

async function saveExport(manifest) {
  const exported = path.join(root, 'published')
  await fs.rm(exported, { recursive: true, force: true })
  await fs.mkdir(exported, { recursive: true })
  await fs.cp(path.join(root, '.publish-content'), path.join(exported, 'content'), { recursive: true })
  const portable = { ...manifest, records: manifest.records.map(({ note, permalink, section }) => ({ note, permalink, section })) }
  await fs.writeFile(path.join(exported, 'publish-manifest.json'), JSON.stringify(portable, null, 2))
  console.log('Updated published/: approved notes and referenced media for remote builds.')
}

export async function finish(manifest, options = {}) {
  const output = options.output ?? path.join(root, 'public')
  const input = options.input ?? path.join(root, '.publish-content')
  const migration = options.migration ?? path.join(root, 'migration')
  await fs.mkdir(migration, { recursive: true })
  await optimizeOutput(output)
  const aliases = []
  let legacy = []
  try { legacy = JSON.parse(await fs.readFile(path.join(migration, 'legacy-urls.json'), 'utf8')) } catch {}
  const urls = legacy.filter(route => manifest.notes.includes(route.current.slice(1).replace(/\.html$/, '.md'))).map(route => ({ ...route, redirect: route.old !== route.current }))
  let historical = []
  try { historical = JSON.parse(await fs.readFile(path.join(migration, 'historical-redirects.json'), 'utf8')) } catch {}
  for (const route of historical) if (route.current === '/' || manifest.notes.includes(route.current.slice(1).replace(/\.html$/, '.md'))) urls.push({ ...route, redirect: true })
  // Plugin-generated links may omit .html. Consolidate those variants at the host.
  for (const note of manifest.notes) {
    const slug = slugifyFilePath(note)
    const current = slug === 'index' ? '/' : `/${slug}.html`
    urls.push({ old: `/${slug}`, current, redirect: true, force: true })
    if (slug.endsWith('/index')) for (const old of [`/${slug.slice(0, -6)}`, `/${slug.slice(0, -5)}`]) urls.push({ old, current, redirect: true, force: true })
  }
  for (const route of urls) if (route.redirect) aliases.push(`${encodeURI(route.old)} ${encodeURI(route.current)} 301!`)
  await fs.writeFile(path.join(output, '_redirects'), aliases.join('\n') + '\n')
  // GitHub Pages has no _redirects support. Carry legacy URLs as instant HTML
  // redirects, with canonical targets and noindex. Never overwrite the homepage.
  for (const route of urls.filter(r => r.redirect && r.old.endsWith('.html') && r.old !== '/index.html')) {
    const file = path.join(output, decodeURI(route.old).slice(1))
    const target = encodeURI(route.current)
    const canonical = 'https://kareemai.com' + target
    const html = `<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="robots" content="noindex,follow"><link rel="canonical" href="${canonical}"><meta http-equiv="refresh" content="0;url=${target}"><title>الانتقال إلى الصفحة</title></head><body><a href="${target}">الانتقال إلى الصفحة</a><script>const target=new URL(${JSON.stringify(target)},location.origin);target.search=location.search;target.hash=location.hash;location.replace(target.href)</script></body></html>`
    await fs.mkdir(path.dirname(file), { recursive: true })
    await fs.writeFile(file, html)
  }
  await fs.writeFile(path.join(output, '.nojekyll'), '')
  await fs.writeFile(path.join(migration, 'urls.json'), JSON.stringify(urls, null, 2) + '\n')
  await fs.writeFile(path.join(output, 'robots.txt'), 'User-agent: *\nAllow: /\nSitemap: https://kareemai.com/sitemap.xml\n')
  const entries = []
  const sitemap = []
  const escape = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
  for (const file of manifest.notes) {
    const { data } = matter(await fs.readFile(path.join(input, file), 'utf8'))
    const slug = slugifyFilePath(file)
    const url = slug === 'index' ? 'https://kareemai.com/' : `https://kareemai.com/${slug}.html`
    const updated = data.modified ?? data.updated ?? data.date
    const date = updated && new Date(updated)
    sitemap.push(`<url><loc>${escape(url)}</loc>${date && !Number.isNaN(date.getTime()) ? `<lastmod>${date.toISOString()}</lastmod>` : ''}</url>`)
    if (!data.date) continue
    const original = legacy.find(route => route.current === `/${slug}.html`)
    // Quarto used literal spaces in some GUIDs. Preserve the exact identifier.
    const guid = original ? 'https://kareemai.com' + original.old : url
    const tree = fromHtml(await fs.readFile(path.join(output, slug + '.html'), 'utf8'))
    let article
    visit(tree, 'element', node => {
      if (node.tagName === 'article' && !article) article = node
      for (const attr of ['src', 'href']) {
        if (typeof node.properties[attr] === 'string') node.properties[attr] = new URL(node.properties[attr], url).href
      }
      if (typeof node.properties.srcSet === 'string') node.properties.srcSet = node.properties.srcSet.split(', ').map(candidate => {
        const [src, width] = candidate.split(' ')
        return `${new URL(src, url).href} ${width}`
      }).join(', ')
    })
    const body = article ? toHtml(article) : data.description ?? ''
    const item = `<item><title>${escape(data.title)}</title><link>${escape(url)}</link><guid isPermaLink="true">${escape(guid)}</guid><pubDate>${new Date(data.date).toUTCString()}</pubDate><description>${escape(data.description ?? '')}</description><content:encoded><![CDATA[${body.replace(/\]\]>/g, ']]]]><![CDATA[>')}]]></content:encoded></item>`
    entries.push({ file, section: data.section, series: data.series, date: String(data.date), item })
  }
  entries.sort((a, b) => b.date.localeCompare(a.date))
  await fs.writeFile(path.join(output, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${sitemap.join('')}</urlset>`)
  for (const [endpoint, prefix, title] of [['blog/feed.xml', 'blog/', 'مدونة عبدالكريم الخطيب'], ['til/index.xml', 'til/', 'ملاحظات عبدالكريم الخطيب'], ['blog/posts/nlp/embedding_world/index.xml', 'blog/posts/nlp/embedding_world/', 'Embedding Models & Vector Search'], ['index.xml', '', 'كتابات عبدالكريم الخطيب']]) {
    const xml = `<?xml version="1.0" encoding="UTF-8"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:atom="http://www.w3.org/2005/Atom"><channel><title>${escape(title)}</title><link>https://kareemai.com/</link><description>${escape(title)}</description><atom:link href="https://kareemai.com/${endpoint}" rel="self" type="application/rss+xml"/>${entries.filter(e => prefix === '' || (prefix === 'blog/' ? e.section === 'blog' : prefix === 'til/' ? e.section === 'til' : e.series === 'embedding')).map(e => e.item).join('')}</channel></rss>`
    await fs.mkdir(path.dirname(path.join(output, endpoint)), { recursive: true })
    await fs.writeFile(path.join(output, endpoint), xml)
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.chdir(root)
  const mode = process.argv[2] ?? 'build'
  if (!['build', 'preview', 'export'].includes(mode)) throw new Error(`Unknown mode ${mode}`)
  await run(process.execPath, ['scripts/setup-plugins.mjs'])
  const manifest = await stage()
  if (mode === 'export') {
    await saveExport(manifest)
    process.exit(0)
  }
  await fs.rm('public', { recursive: true, force: true })
  await run(process.execPath, ['quartz/bootstrap-cli.mjs', 'build', '-d', '.publish-content'])
  await finish(manifest)
  if (manifest.records.some(record => record.source)) await saveExport(manifest)
  if (mode === 'preview') {
    const { watch } = await import('chokidar')
    const server = spawn(process.execPath, ['scripts/serve.mjs'], { cwd: root, stdio: 'inherit' })
    let busy = false, pending = false, timer
    async function rebuild() {
      if (busy) { pending = true; return }
      busy = true
      try { await run(process.execPath, ['scripts/site.mjs', 'build']); console.log('Preview rebuilt. Refresh your browser.') }
      catch (error) { console.error(error.message) }
      finally { busy = false; if (pending) { pending = false; rebuild() } }
    }
    const watcher = watch(vaultRoot, { ignoreInitial: true, ignored: file => { const rel = path.relative(vaultRoot, file); return rel.split(path.sep).some(part => part.startsWith('.') || ['node_modules', '_site', '_book', '__pycache__'].includes(part)) || (file.startsWith(root + path.sep) && !file.startsWith(path.join(root, 'content'))); } }).on('all', () => { clearTimeout(timer); timer = setTimeout(rebuild, 300) })
    for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => { server.kill(); watcher.close(); process.exit() })
  }
}
