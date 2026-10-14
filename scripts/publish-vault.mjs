import fs from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import matter from 'gray-matter'
import { unified } from 'unified'
import remarkParse from 'remark-parse'
import { visit } from 'unist-util-visit'
import { slugifyFilePath } from '@quartz-community/utils'

const ignored = new Set(['node_modules', '.git', '.obsidian', '.trash', '.venv', '_site', '_book', '__pycache__'])
const media = /\.(png|jpe?g|webp|svg|gif|ico|pdf|mp3|mp4|wav|webm|ogg|avif|htm)$/i
const external = ref => /^(?:[a-z][a-z\d+.-]*:|#|\/\/)/i.test(ref)
const inside = (base, file) => file === base || file.startsWith(base + path.sep)
const posix = file => file.split(path.sep).join('/')

export async function collectFiles(directory, projectRoot) {
  const found = []
  async function walk(dir) {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      if (entry.name.startsWith('.') || ignored.has(entry.name)) continue
      if (dir === projectRoot && entry.name !== 'content') continue
      const file = path.join(dir, entry.name)
      if (entry.isDirectory()) await walk(file)
      else if (entry.isFile()) found.push(file)
    }
  }
  await walk(directory)
  return found
}

export async function publishVault({ vaultRoot, projectRoot, target, manifestPath, persist = true }) {
  const contentRoot = path.join(projectRoot, 'content')
  const all = await collectFiles(vaultRoot, projectRoot)
  const fileSet = new Set(all)
  const notes = []
  const today = new Intl.DateTimeFormat('en-CA', { timeZone: 'Africa/Nairobi', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date())
  for (const file of all.filter(f => f.endsWith('.md'))) {
    const handle = await fs.open(file)
    const buffer = Buffer.alloc(16384)
    let header
    try { const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0); header = buffer.subarray(0, bytesRead).toString('utf8') }
    finally { await handle.close() }
    const frontmatter = header.match(/^---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/)?.[1]
    if (!frontmatter || !/^publish:\s*true\s*(?:#.*)?$/m.test(frontmatter)) continue
    const raw = await fs.readFile(file, 'utf8')
    const parsed = matter(raw)
    if (parsed.data.publish !== true || parsed.data.draft === true) continue
    const managed = inside(contentRoot, file)
    const relative = managed ? posix(path.relative(contentRoot, file)) : null
    const meta = { ...parsed.data }
    const additions = {}
    if (!meta.permalink) {
      meta.permalink = '/' + (managed ? slugifyFilePath(relative) : 'notes/' + slugifyFilePath(path.basename(file)))
      additions.permalink = meta.permalink
    }
    const route = String(meta.permalink).replace(/^\//, '').replace(/\.html$/, '').replace(/\/$/, '/index')
    if (!route || route.split('/').some(p => !p || p === '.' || p === '..') || /[?#\\]/.test(route)) throw new Error(`Invalid permalink in ${file}`)
    const slug = slugifyFilePath(route + '.md')
    if (slug !== route) throw new Error(`Permalink must use lowercase URL-safe segments: ${file}: /${slug}`)
    if (!meta.section) {
      meta.section = relative?.startsWith('til/tils/') ? 'til' : relative?.startsWith('blog/') && !meta.listing ? 'blog' : managed ? 'page' : 'blog'
      additions.section = meta.section
    }
    if (!meta.series && relative?.startsWith('blog/posts/nlp/embedding_world/') && !meta.listing) { meta.series = 'embedding'; additions.series = meta.series }
    if (!meta.date && meta.section !== 'page') { meta.date = meta.published ?? today; additions.date = meta.date }
    for (const key of ['date', 'modified', 'published']) {
      if (meta[key] instanceof Date) meta[key] = meta[key].toISOString().slice(0, 10)
      else if (/^\d{4}-\d{1,2}-\d{1,2}$/.test(String(meta[key]))) meta[key] = String(meta[key]).split('-').map((n, i) => i ? n.padStart(2, '0') : n).join('-')
    }
    meta.title ||= path.basename(file, '.md')
    meta.lang ||= /[\u0600-\u06ff]/.test(meta.title) ? 'ar' : 'en'
    meta.dir ||= meta.lang.startsWith('ar') ? 'rtl' : 'ltr'
    notes.push({ file, raw, content: parsed.content, meta, additions, staged: slug + '.md' })
  }
  const byFile = new Map(notes.map(n => [n.file, n]))
  const byRoute = new Map()
  for (const note of [...notes].sort((a, b) => Number(Boolean(a.additions.permalink)) - Number(Boolean(b.additions.permalink)))) {
    if (byRoute.has(note.staged) && note.additions.permalink) {
      const suffix = createHash('sha256').update(posix(path.relative(vaultRoot, note.file))).digest('hex').slice(0, 8)
      note.staged = note.staged.replace(/\.md$/, `-${suffix}.md`)
      note.meta.permalink = '/' + note.staged.replace(/\.md$/, '')
      note.additions.permalink = note.meta.permalink
    }
    if (byRoute.has(note.staged)) throw new Error(`Duplicate permalink /${note.staged}: ${byRoute.get(note.staged).file} and ${note.file}. Give one note a different permalink.`)
    byRoute.set(note.staged, note)
  }
  if (!byRoute.has('index.md')) throw new Error('No published homepage at permalink: /index')
  const assets = new Map()
  const issues = []
  const lookup = (ref, note, wiki = false) => {
    let name = decodeURIComponent(ref.split(/[?#]/)[0]).replace(/\\\|/g, '|')
    if (!name) return { file: note.file, suffix: ref }
    const suffix = ref.slice(ref.search(/[?#]/) === -1 ? ref.length : ref.search(/[?#]/))
    const isLegacy = name.endsWith('.html') || ref.startsWith('https://kareemai.com/')
    name = name.replace(/^https:\/\/kareemai\.com\//, '/').replace(/\.html$/, '.md')
    const route = path.posix.normalize(path.posix.join(path.posix.dirname(note.staged), name))
    if (isLegacy || name.startsWith('/')) {
      const routeName = (name.startsWith('/') ? name.slice(1) : route).replace(/\.md$/, '') + '.md'
      const old = byRoute.get(routeName)
      if (old) return { file: old.file, suffix }
    }
    const bases = name.startsWith('/') ? [vaultRoot, contentRoot] : wiki ? [vaultRoot, path.dirname(note.file), contentRoot] : [path.dirname(note.file), vaultRoot, contentRoot]
    for (const base of bases) {
      const candidate = path.resolve(base, name.replace(/^\//, ''))
      if (!inside(vaultRoot, candidate)) continue
      for (const test of [candidate, candidate + '.md', path.join(candidate, 'index.md')]) if (fileSet.has(test)) return { file: test, suffix }
    }
    const clean = name.replace(/^(?:\.\.\/|\.\/|\/)+/g, '')
    let matches = all.filter(file => {
      const rel = posix(path.relative(vaultRoot, file))
      return rel === clean || rel.endsWith('/' + clean) || (!path.extname(clean) && rel.endsWith('/' + clean + '.md'))
    })
    if (!matches.length) {
      const basename = path.posix.basename(clean)
      matches = all.filter(file => path.basename(file) === basename || (!path.extname(basename) && path.basename(file) === basename + '.md'))
    }
    if (matches.length === 1) return { file: matches[0], suffix }
    if (matches.length > 1) {
      const nearby = matches.filter(f => path.dirname(f) === path.dirname(note.file))
      if (nearby.length === 1) return { file: nearby[0], suffix }
      issues.push(`${posix(path.relative(vaultRoot, note.file))}: ambiguous link "${ref}"; use the full vault path.`)
    }
    return { missing: true, suffix }
  }
  async function link(ref, note, wiki = false) {
    if (external(ref) && !ref.startsWith('https://kareemai.com/')) return { url: ref }
    const result = lookup(ref, note, wiki)
    if (result.missing) {
      // Generated Quartz folder pages have no physical Markdown source.
      if (!wiki && !media.test(ref.split(/[?#]/)[0])) return { url: ref }
      issues.push(`${posix(path.relative(vaultRoot, note.file))}: cannot find "${ref}"`)
      return { url: ref }
    }
    if (result.file.endsWith('.md')) {
      const published = byFile.get(result.file)
      return published ? { url: '/' + published.staged + result.suffix, note: true } : { private: true }
    }
    if (!media.test(result.file)) throw new Error(`Unsupported published attachment: ${ref}`)
    if (!assets.has(result.file)) {
      const bytes = await fs.readFile(result.file)
      const hash = createHash('sha256').update(bytes).digest('hex').slice(0, 24)
      assets.set(result.file, { staged: 'assets/' + hash + path.extname(result.file).toLowerCase(), bytes })
    }
    return { url: '/' + assets.get(result.file).staged + result.suffix }
  }
  for (const note of notes) {
    const nodes = []
    const edits = []
    visit(unified().use(remarkParse).parse(note.content), n => { if (['image', 'link', 'definition', 'html', 'text'].includes(n.type)) nodes.push(n) })
    for (const node of nodes) {
      const start = node.position.start.offset, end = node.position.end.offset
      const fragment = note.content.slice(start, end)
      if (['image', 'link', 'definition'].includes(node.type)) {
        const resolved = await link(node.url, note)
        if (resolved.private) edits.push({ start, end, value: node.children?.map(c => c.value ?? '').join('') ?? node.alt ?? '' })
        else {
          let rawUrl = node.url
          let at = fragment.lastIndexOf(rawUrl)
          if (at < 0) { rawUrl = node.url.replace(/([\\()[\]])/g, '\\$1'); at = fragment.lastIndexOf(rawUrl) }
          if (at >= 0) edits.push({ start: start + at, end: start + at + rawUrl.length, value: resolved.url })
        }
      } else if (node.type === 'html') {
        for (const match of fragment.matchAll(/(?:src|href)=["']([^"']+)["']/g)) {
          const resolved = await link(match[1], note)
          const at = start + match.index + match[0].indexOf(match[1])
          edits.push({ start: at, end: at + match[1].length, value: resolved.private ? '#' : resolved.url })
        }
      } else {
        for (const match of fragment.matchAll(/(!?)\[\[([^\]]+)\]\]/g)) {
          const [ref, ...aliases] = match[2].replace(/\\\|/g, '|').split('|')
          const resolved = await link(ref, note, true)
          const label = aliases.join('|') || ref.split('#')[0].split('/').at(-1)
          const pipe = match[2].includes('\\|') ? '\\|' : '|'
          const value = resolved.private ? label : `${match[1]}[[${resolved.url.replace(/^\//, '').replace(/\.md(?=#|$)/, '')}${pipe}${aliases.length ? aliases.join(pipe) : label}]]`
          edits.push({ start: start + match.index, end: start + match.index + match[0].length, value })
        }
      }
    }
    for (const key of ['socialImage', 'image', 'cover']) {
      if (typeof note.meta[key] === 'string' && note.meta[key]) {
        const resolved = await link(note.meta[key].replace(/^!\[\[|\]\]$/g, ''), note, true)
        note.meta.socialImage = resolved.url
      }
    }
    note.output = note.content
    let last = Infinity
    for (const edit of edits.sort((a, b) => b.start - a.start)) {
      if (edit.end > last) continue
      note.output = note.output.slice(0, edit.start) + edit.value + note.output.slice(edit.end)
      last = edit.start
    }
  }
  if (issues.length) throw new Error(issues.join('\n'))
  await fs.rm(target, { recursive: true, force: true })
  await fs.mkdir(target, { recursive: true })
  for (const note of notes) {
    const destination = path.join(target, note.staged)
    await fs.mkdir(path.dirname(destination), { recursive: true })
    const stagedMeta = { ...note.meta }
    // Quartz treats permalink as a redirect alias. The snapshot already lives at
    // that URL, so keep this identity only in the editable source note.
    delete stagedMeta.permalink
    await fs.writeFile(destination, matter.stringify(note.output, stagedMeta))
    if (persist && Object.keys(note.additions).length) {
      const fields = Object.entries(note.additions).map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join('\n')
      // Preserve the author's Markdown and existing YAML formatting.
      await fs.writeFile(note.file, note.raw.replace(/^---\r?\n/, `---\n${fields}\n`))
    }
  }
  for (const asset of assets.values()) {
    const destination = path.join(target, asset.staged)
    await fs.mkdir(path.dirname(destination), { recursive: true })
    await fs.writeFile(destination, asset.bytes)
  }
  const manifest = {
    notes: notes.map(n => n.staged),
    attachments: [...new Set([...assets.values()].map(a => a.staged))],
    records: notes.map(n => ({ source: posix(path.relative(vaultRoot, n.file)), note: n.staged, permalink: n.meta.permalink, section: n.meta.section })),
  }
  await fs.mkdir(path.dirname(manifestPath), { recursive: true })
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2))
  console.log(`Publishing ${notes.length} approved vault notes and ${manifest.attachments.length} referenced attachments.`)
  return manifest
}
