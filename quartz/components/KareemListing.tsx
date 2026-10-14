import { QuartzComponent, QuartzComponentConstructor } from "./types"
import { pathToRoot } from "../util/path"

export default (() => {
  const Listing: QuartzComponent = ({ fileData, allFiles }) => {
    const kind = fileData.frontmatter?.listing
    if (!kind) return null
    const notes = allFiles.filter(f => !f.frontmatter?.listing && f.frontmatter?.publish === true && (kind === "embedding" ? f.frontmatter?.series === "embedding" : f.frontmatter?.section === (kind === "til" ? "til" : "blog")))
      .sort((a, b) => String(b.frontmatter?.date ?? "").localeCompare(String(a.frontmatter?.date ?? "")))
    const root = pathToRoot(fileData.slug!)
    const english = String(fileData.frontmatter?.lang ?? "ar").startsWith("en")
    const heading = kind === "embedding" ? (english ? "Articles in this series" : "مقالات هذه السلسلة") : kind === "til" ? "أحدث الملاحظات" : "أحدث المقالات"
    return <section class="kareem-listing" aria-label={english ? "Articles" : "المقالات"}><h2>{heading}</h2>{notes.map(note => <article class="note-card" lang={String(note.frontmatter?.lang ?? "ar")} dir={note.frontmatter?.dir === "ltr" ? "ltr" : "rtl"}>
      <time dateTime={String(note.frontmatter?.date ?? "")}>{String(note.frontmatter?.date ?? "")}</time>
      <h3><a href={`${root}/${note.slug}.html`}>{note.frontmatter?.title}</a></h3>
      <p>{String(note.frontmatter?.description ?? "")}</p>
    </article>)}</section>
  }
  return Listing
}) satisfies QuartzComponentConstructor
