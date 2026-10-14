// Use the official, compiled npm releases, pinned by package-lock.json.
// This also avoids Quartz 5's Git installer failing on vault paths with spaces.
import fs from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
process.chdir(root)
const pkg = JSON.parse(await fs.readFile('package.json', 'utf8'))
await fs.mkdir('.quartz/plugins', { recursive: true })
for (const entry of await fs.readdir('.quartz/plugins', { withFileTypes: true })) {
  if (entry.isSymbolicLink()) {
    try { await fs.access(path.join('.quartz/plugins', entry.name)) }
    catch { await fs.unlink(path.join('.quartz/plugins', entry.name)) }
  }
}
for (const name of Object.keys(pkg.dependencies).filter(n => n.startsWith('@quartz-community/') && !n.endsWith('/types') && !n.endsWith('/utils'))) {
  const dir = path.resolve('node_modules', name)
  const link = path.resolve('.quartz/plugins', name.split('/')[1])
  await fs.rm(link, { recursive: true, force: true })
  await fs.symlink(dir, link, 'dir')
}
const result = spawnSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', 'await import("./scripts/prepare-plugins.ts"); const { regeneratePluginIndex } = await import("./quartz/plugins/loader/gitLoader.ts"); await regeneratePluginIndex()'], { stdio: 'inherit' })
if (result.status) process.exit(result.status)
