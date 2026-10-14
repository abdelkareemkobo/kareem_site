import { QuartzComponent, QuartzComponentConstructor } from "./types"
import { pathToRoot } from "../util/path"

export default (() => {
  const Listing: QuartzComponent = ({ fileData, allFiles }) => {
    if (fileData.slug === "index") {
      const recent = allFiles.filter(note => note.frontmatter?.publish === true && !note.frontmatter?.listing && ["blog", "til"].includes(String(note.frontmatter?.section)))
        .sort((a, b) => String(b.frontmatter?.modified ?? b.frontmatter?.updated ?? b.frontmatter?.date ?? "").localeCompare(String(a.frontmatter?.modified ?? a.frontmatter?.updated ?? a.frontmatter?.date ?? ""))).slice(0, 8)
      return <section class="garden-recent"><h2>أحدث الكتابات والملاحظات</h2><ul>{recent.map(note => <li><a href={`/${note.slug}.html`} lang={String(note.frontmatter?.lang ?? "ar")} dir={note.frontmatter?.dir === "ltr" ? "ltr" : "rtl"}>{note.frontmatter?.title}</a><time>{String(note.frontmatter?.modified ?? note.frontmatter?.updated ?? note.frontmatter?.date ?? "")}</time></li>)}</ul><p><a href="/blog/feed.html">جميع المقالات ←</a> · <a href="/til/index.html">الملاحظات اليومية ←</a></p></section>
    }
    const kind = fileData.frontmatter?.listing
    if (!kind) return null
    const notes = allFiles.filter(f => !f.frontmatter?.listing && f.frontmatter?.publish === true && (kind === "embedding" ? f.frontmatter?.series === "embedding" : f.frontmatter?.section === (kind === "til" ? "til" : "blog")))
      .sort((a, b) => String(b.frontmatter?.date ?? "").localeCompare(String(a.frontmatter?.date ?? "")))
    const root = pathToRoot(fileData.slug!)
    const english = String(fileData.frontmatter?.lang ?? "ar").startsWith("en")
    const heading = kind === "embedding" ? (english ? "Articles in this series" : "مقالات هذه السلسلة") : kind === "til" ? "أحدث الملاحظات" : "أحدث المقالات"
    return <section class="kareem-listing" aria-label={english ? "Articles" : "المقالات"}><h2>{heading}</h2><div class="note-grid">{notes.map(note => <article class="note-card" lang={String(note.frontmatter?.lang ?? "ar")} dir={note.frontmatter?.dir === "ltr" ? "ltr" : "rtl"}>
      {note.frontmatter?.socialImage && <a class="note-cover" href={`${root}/${note.slug}.html`} tabIndex={-1} aria-hidden="true"><img src={String(note.frontmatter.socialImage)} alt="" loading="lazy" decoding="async" /></a>}
      <div class="note-card-body">
      <time dateTime={String(note.frontmatter?.date ?? "")}>{String(note.frontmatter?.date ?? "")}</time>
      <h3><a href={`${root}/${note.slug}.html`}>{note.frontmatter?.title}</a></h3>
      <p>{String(note.frontmatter?.description ?? "")}</p>
      </div>
    </article>)}</div></section>
  }
  return Listing
}) satisfies QuartzComponentConstructor
