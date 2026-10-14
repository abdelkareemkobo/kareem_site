import fs from 'node:fs/promises'
import path from 'node:path'
import { createHash } from 'node:crypto'
import sharp from 'sharp'
import { fromHtml } from 'hast-util-from-html'
import { toHtml } from 'hast-util-to-html'
import { visit } from 'unist-util-visit'

async function allFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  return (await Promise.all(entries.map(e => e.isDirectory() ? allFiles(path.join(dir, e.name)) : path.join(dir, e.name)))).flat()
}

export async function optimizeOutput(output) {
  const paths = new Set((await allFiles(output)).map(file => path.relative(output, file).split(path.sep).join('/')))
  const localURL = (ref, file) => {
    try {
      const url = new URL(ref, `https://kareemai.com/${file}`)
      if (url.origin !== 'https://kareemai.com') return null
      const relative = decodeURIComponent(url.pathname.slice(1))
      if (relative.split('/').includes('..')) return null
      return { url, relative }
    } catch { return null }
  }
  const media = new Map()
  async function imageInfo(relative) {
    if (media.has(relative)) return media.get(relative)
    const bytes = await fs.readFile(path.join(output, relative))
    let meta
    try { meta = await sharp(bytes, { animated: true }).metadata() } catch { return null }
    if (!meta.width || !meta.height) return null
    const info = { width: meta.autoOrient?.width ?? meta.width, height: meta.pageHeight ?? meta.autoOrient?.height ?? meta.height, variants: [] }
    media.set(relative, info)
    if (meta.pages > 1 || !['png', 'jpeg', 'webp', 'avif'].includes(meta.format) || bytes.length < 80000) return info
    const hash = createHash('sha256').update(bytes).digest('hex').slice(0, 24)
    for (const width of [480, 960, 1440]) {
      if (width > info.width) continue
      const encoded = await sharp(bytes).rotate().resize({ width, withoutEnlargement: true }).webp({ quality: 88 }).toBuffer()
      if (encoded.length >= bytes.length * 0.85) continue
      const asset = `assets/responsive/${hash}-${width}.webp`
      await fs.mkdir(path.dirname(path.join(output, asset)), { recursive: true })
      await fs.writeFile(path.join(output, asset), encoded)
      info.variants.push({ file: asset, width })
    }
    return info
  }
  // A small, correctly typed favicon replaces the old SVG stored in an .ico file.
  try { await sharp(path.join(output, 'static/favicon.ico')).resize(48, 48).png().toFile(path.join(output, 'static/favicon.png')) } catch {}
  const searchFile = path.join(output, 'static/contentIndex.json')
  const search = JSON.parse(await fs.readFile(searchFile, 'utf8'))
  const navigation = {}
  for (const [slug, data] of Object.entries(search)) {
    delete data.richContent
    navigation[slug] = { slug: data.slug, title: data.title, links: data.links, tags: data.tags }
  }
  await fs.writeFile(searchFile, JSON.stringify(search))
  await fs.writeFile(path.join(output, 'static/navigationIndex.json'), JSON.stringify(navigation))

  for (const file of paths) {
    if (!file.endsWith('.html')) continue
    const target = path.join(output, file)
    const tree = fromHtml(await fs.readFile(target, 'utf8'))
    const images = []
    let hasMath = false
    let hasMermaid = false
    let hasCode = false
    visit(tree, 'element', node => {
      const p = node.properties
      if (Array.isArray(p.className) && p.className.includes('katex')) hasMath = true
      if (Array.isArray(p.className) && p.className.includes('mermaid')) hasMermaid = true
      if (node.tagName === 'pre') hasCode = true
      if (node.tagName === 'img' && typeof p.src === 'string') images.push(node)
      if (node.tagName === 'iframe') p.loading ??= 'lazy'
      // Point server-rendered links directly to the same URLs used by canonicals.
      if (node.tagName === 'a' && typeof p.href === 'string' && !p.href.startsWith('#')) {
        const local = localURL(p.href, file)
        if (!local) return
        let page = local.relative
        if (!page || page === 'index' || page === 'index.html') page = ''
        else if (paths.has(page + '.html')) page += '.html'
        else if (page.endsWith('/') && paths.has(page + 'index.html')) page += 'index.html'
        else if (!paths.has(page) && paths.has(page + '/index.html')) page += '/index.html'
        if (page === '' || page.endsWith('.html') && paths.has(page)) p.href = '/' + page + local.url.search + local.url.hash
      }
    })
    if (!hasMath) visit(tree, 'element', node => {
      if (!Array.isArray(node.children)) return
      node.children = node.children.filter(child => !(child.type === 'element' && ['link', 'script'].includes(child.tagName) && /katex@/.test(String(child.properties.href ?? child.properties.src ?? ''))))
    })
    visit(tree, 'element', node => {
      if (node.tagName === 'script' && !node.properties.src) for (const child of node.children) if (child.type === 'text' && child.value.includes('clipboard-button')) child.value = child.value.replaceAll('"Copy source"', '"نسخ الكود"')
      if (!Array.isArray(node.children)) return
      node.children = node.children.filter(child => {
        if (child.type !== 'element' || child.tagName !== 'script' || child.properties.src) return true
        const script = child.children.map(c => c.value ?? '').join('')
        return !(!hasMermaid && script.includes('mermaid-controls') || !hasCode && script.includes('clipboard-button'))
      })
    })
    for (const [index, node] of images.entries()) {
      const p = node.properties
      p.decoding ??= 'async'
      p.loading ??= index === 0 ? 'eager' : 'lazy'
      if (file === 'index.html' && index === 0) p.fetchPriority = 'high'
      const local = localURL(p.src, file)
      if (!local || !paths.has(local.relative)) continue
      const info = await imageInfo(local.relative)
      if (!info) continue
      if (!p.width && !p.height) { p.width = info.width; p.height = info.height }
      else if (p.width && !p.height && Number(p.width)) p.height = Math.round(Number(p.width) * info.height / info.width)
      else if (p.height && !p.width && Number(p.height)) p.width = Math.round(Number(p.height) * info.width / info.height)
      if (info.variants.length && !p.srcSet) {
        p.srcSet = [...info.variants.map(v => `/${v.file} ${v.width}w`), ...(info.variants.some(v => v.width === info.width) ? [] : [`${p.src} ${info.width}w`])].join(', ')
        p.sizes = '(max-width: 800px) calc(100vw - 32px), (max-width: 1200px) calc(100vw - 290px), 750px'
      }
    }
    // Supply dimensions for social images without guessing remote image sizes.
    let social
    visit(tree, 'element', node => { if (node.tagName === 'meta' && node.properties.property === 'og:image') social = node.properties.content })
    const socialFile = social && localURL(social, file)
    if (socialFile && paths.has(socialFile.relative)) {
      const info = await imageInfo(socialFile.relative)
      if (info) visit(tree, 'element', node => {
        if (node.tagName === 'head') for (const [key, value] of [['width', info.width], ['height', info.height]]) node.children.push({ type: 'element', tagName: 'meta', properties: { property: 'og:image:' + key, content: String(value) }, children: [] })
      })
    }
    await fs.writeFile(target, toHtml(tree, { allowDangerousHtml: true }))
  }
  console.log(`Optimized images and metadata; ${Object.keys(navigation).length} notes in the lightweight navigation index.`)
}
