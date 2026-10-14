import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import matter from 'gray-matter'
import { publishVault } from './publish-vault.mjs'
import { finish, root } from './site.mjs'
import { spawnSync } from 'node:child_process'

async function fixture(t) {
  const vaultRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'quartz-publishing-'))
  t.after(() => fs.rm(vaultRoot, { recursive: true, force: true }))
  const projectRoot = path.join(vaultRoot, 'Website')
  const target = path.join(projectRoot, '.publish-content')
  const manifestPath = path.join(projectRoot, '.quartz/manifest.json')
  async function write(relative, text) {
    const destination = path.join(vaultRoot, relative)
    await fs.mkdir(path.dirname(destination), { recursive: true })
    await fs.writeFile(destination, text)
    return destination
  }
  await write('Website/content/index.md', '---\npublish: true\npermalink: /index\nsection: page\n---\nHomepage\n')
  return { vaultRoot, projectRoot, target, manifestPath, write, build: () => publishVault({ vaultRoot, projectRoot, target, manifestPath }) }
}

test('publishes notes anywhere; moving notes and images preserves URLs and media identities', async t => {
  const f = await fixture(t)
  const note = await f.write('Knowledge/Learning.md', '---\npublish: true\ntitle: Learning\n---\n![[Attachments/photo.png|240]]\n')
  const photo = await f.write('Attachments/photo.png', Buffer.from('image-fixture'))
  const first = await f.build()
  assert(first.notes.includes('notes/learning.md'))
  const identity = matter(await fs.readFile(note, 'utf8')).data
  assert.equal(identity.permalink, '/notes/learning')
  assert(identity.date)
  const rendered = await fs.readFile(path.join(f.target, 'notes/learning.md'), 'utf8')
  assert.match(rendered, /!\[\[assets\/\w+\.png\|240\]\]/)
  await fs.mkdir(path.join(f.vaultRoot, 'Completely/Different'), { recursive: true })
  await fs.rename(note, path.join(f.vaultRoot, 'Completely/Different/New name.md'))
  await fs.rename(photo, path.join(f.vaultRoot, 'Completely/Different/photo.png'))
  const second = await f.build()
  assert.deepEqual(second.notes, first.notes)
  assert.deepEqual(second.attachments, first.attachments)
  const moved = matter(await fs.readFile(path.join(f.target, 'notes/learning.md'), 'utf8'))
  assert.deepEqual(moved.data, matter(rendered).data)
  assert.equal(moved.content, matter(rendered).content)
})

test('resolves Markdown images, vault paths, note aliases, headings and published transclusions', async t => {
  const f = await fixture(t)
  await f.write('Files/Pasted image.png', 'pixels')
  await f.write('Ideas/Target.md', '---\npublish: true\n---\n## A heading\nPublic text\n')
  await f.write('Other/Source.md', '---\npublish: true\n---\n![Alt](../Files/Pasted%20image.png)\n![[Files/Pasted image.png|My picture]]\n[[Target#A heading|Read more]]\n![[Target#A heading]]\n```text\n![[Not a real file.png]]\n```\n')
  const manifest = await f.build()
  assert.equal(manifest.attachments.length, 1)
  const source = await fs.readFile(path.join(f.target, 'notes/source.md'), 'utf8')
  assert.match(source, /!\[Alt\]\(\/assets\/\w+\.png\)/)
  assert.match(source, /\[\[notes\/target#A heading\|Read more\]\]/)
  assert.match(source, /!\[\[notes\/target#A heading\|Target\]\]/)
  assert(source.includes('![[Not a real file.png]]'))
})

test('private notes, private transclusions and unrelated attachments never enter the snapshot', async t => {
  const f = await fixture(t)
  await f.write('Private/Secret.md', '---\npublish: false\n---\nSENSITIVE TEXT\n![[private.png]]\n')
  await f.write('Private/Draft.md', '---\npublish: true\ndraft: true\n---\nDRAFT TEXT\n')
  await f.write('Private/private.png', 'private pixels')
  await f.write('Public.md', '---\npublish: true\n---\n![[Secret]]\n[[Secret|Private link]]\n[Private link](Private/Secret.md)\n')
  const manifest = await f.build()
  assert.equal(manifest.notes.length, 2)
  assert.equal(manifest.attachments.length, 0)
  const output = await fs.readFile(path.join(f.target, 'notes/public.md'), 'utf8')
  assert(!output.includes('SENSITIVE TEXT'))
  assert(!output.includes('[[Secret'))
  assert(!output.includes('Private/Secret'))
})

test('duplicate basenames get separate stable URLs, and precise image paths resolve correctly', async t => {
  const f = await fixture(t)
  await f.write('A/Same.md', '---\npublish: true\n---\n![[A/photo.png]]\n')
  await f.write('B/Same.md', '---\npublish: true\n---\n![Photo](photo.png)\n')
  await f.write('A/photo.png', 'A pixels')
  await f.write('B/photo.png', 'B pixels')
  const manifest = await f.build()
  assert.equal(new Set(manifest.notes).size, 3)
  assert.equal(manifest.attachments.length, 2)
  assert.deepEqual((await f.build()).notes, manifest.notes)
})

test('ambiguous and missing image references stop publication with an actionable error', async t => {
  const f = await fixture(t)
  await f.write('Elsewhere/Post.md', '---\npublish: true\n---\n![[photo.png]]\n')
  await f.write('A/photo.png', 'A')
  await f.write('B/photo.png', 'B')
  await assert.rejects(f.build(), /ambiguous link "photo.png"/)
  await f.write('Elsewhere/Post.md', '---\npublish: true\n---\n![[missing.png]]\n')
  await assert.rejects(f.build(), /cannot find "missing.png"/)
})

test('Quartz renders vault images and transclusions; RSS keeps the same GUID after moving files', async t => {
  const f = await fixture(t)
  const note = await f.write('Research/Moving note.md', '---\npublish: true\ntitle: Moving note\n---\n![[photo.png|240]]\n![[Target#Public heading]]\n![[Secret]]\n')
  await f.write('Research/Target.md', '---\npublish: true\ntitle: Target\n---\n## Public heading\nPublished transclusion text.\n')
  await f.write('Secret.md', '---\npublish: false\n---\nSECRET MUST NOT LEAK\n')
  const photo = await f.write('Pictures/photo.png', Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jBz8AAAAASUVORK5CYII=', 'base64'))
  const output = path.join(f.projectRoot, 'public')
  async function render() {
    const manifest = await f.build()
    const build = spawnSync(process.execPath, ['quartz/bootstrap-cli.mjs', 'build', '-d', f.target, '-o', output], { cwd: root, encoding: 'utf8' })
    assert.equal(build.status, 0, build.stdout + build.stderr)
    await finish(manifest, { input: f.target, output, migration: path.join(f.projectRoot, 'migration') })
    const html = await fs.readFile(path.join(output, 'notes/moving-note.html'), 'utf8')
    assert(html.includes('Published transclusion text.'))
    assert(!html.includes('SECRET MUST NOT LEAK'))
    assert.match(html, /src="[^" ]*assets\/\w+\.png"/)
    const rss = await fs.readFile(path.join(output, 'blog/feed.xml'), 'utf8')
    assert(rss.includes('<guid isPermaLink="true">https://kareemai.com/notes/moving-note.html</guid>'))
    assert(rss.includes('https://kareemai.com/assets/'))
    assert(!rss.includes('SECRET MUST NOT LEAK'))
    return rss.match(/<guid[^>]*>[^<]+<\/guid>/g).sort()
  }
  const guids = await render()
  await fs.mkdir(path.join(f.vaultRoot, 'Reorganized'), { recursive: true })
  await fs.rename(note, path.join(f.vaultRoot, 'Reorganized/Renamed.md'))
  await fs.rename(photo, path.join(f.vaultRoot, 'Reorganized/photo.png'))
  assert.deepEqual(await render(), guids)
})
