// Plugins bundle their own locales. Build local copies from pinned npm releases,
// using the site's Arabic dictionary; never edit installed npm packages.
import fs from "node:fs/promises"
import path from "node:path"
import arabic, { arabicNotes } from "../quartz/i18n/locales/ar-SA"

function js(value: unknown): string {
  if (typeof value === "function") return value.toString()
  if (Array.isArray(value)) return `[${value.map(js).join(",")}]`
  if (value && typeof value === "object") return `{${Object.entries(value).map(([key, v]) => `${JSON.stringify(key)}:${js(v)}`).join(",")}}`
  return JSON.stringify(value)
}

const locale = {
  ...arabic,
  components: {
    ...arabic.components,
    search: { ...arabic.components.search, noResults: "لا توجد نتائج", noResultsHint: "جرّب كلمات أخرى", tagFilterHint: "تصفية حسب الوسم", noTagsFound: "لا توجد وسوم مطابقة" },
  },
}
for (const name of await fs.readdir("node_modules/@quartz-community")) {
  const installed = path.resolve("node_modules/@quartz-community", name)
  let source: string
  try { source = await fs.readFile(path.join(installed, "dist/index.js"), "utf8") } catch { continue }
  const marker = /\/\/ src\/i18n\/locales\/ar-SA\.ts\n[\s\S]*?(?=\n\/\/ src\/)/
  if (!marker.test(source)) continue
  const local = path.resolve(".quartz/overrides", name)
  await fs.rm(local, { recursive: true, force: true })
  await fs.cp(installed, local, { recursive: true })
  async function patchDirectory(dir: string) {
    for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name)
      if (entry.isDirectory()) await patchDirectory(file)
      else if (entry.name.endsWith(".js")) {
        let text = await fs.readFile(file, "utf8")
        text = text.replace(marker, `// Site Arabic locale\nconst arabicNotes = ${arabicNotes.toString()};\nvar ar_SA_default = ${js(locale)};\n`)
        if (name === "graph") text = text.replaceAll('"Global Graph"', '"عرض خريطة الروابط كاملة"')
        if (name === "search") text = text.replaceAll('children: "Search"', 'children: "بحث"')
        await fs.writeFile(file, text)
      }
    }
  }
  await patchDirectory(path.join(local, "dist"))
  const link = path.resolve(".quartz/plugins", name)
  await fs.rm(link, { recursive: true, force: true })
  await fs.symlink(local, link, "dir")
}
