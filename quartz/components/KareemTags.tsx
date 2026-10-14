import { QuartzComponent, QuartzComponentConstructor } from "./types"

const internalTags = new Set(["publish", "note", "projects", "idea_forge", "blogging", "til"])
const labels: Record<string, string> = { embedding: "نماذج التضمين", nlp: "معالجة اللغة الطبيعية", arabic: "اللغة العربية", qdrant: "Qdrant", sparse: "البحث المتناثر", transformers: "المحوّلات", vector_db: "قواعد البيانات المتجهية", benchmark: "اختبارات الأداء", papers: "أوراق بحثية", architecture: "بنية النماذج", llm: "النماذج اللغوية", "fine-tuning": "الضبط الدقيق", "tool-calling": "استدعاء الأدوات" }
export const tagLabel = (tag: string) => tag === "index" || !tag ? "تصفّح المواضيع" : labels[tag] ?? tag.replaceAll("_", " ")
const noteCount = (count: number) => count === 1 ? "ملاحظة واحدة" : count === 2 ? "ملاحظتان" : `${count} ${count <= 10 ? "ملاحظات" : "ملاحظة"}`
const tagURL = (tag: string) => `/tags/${tag.split("/").map(encodeURIComponent).join("/")}.html`

export default (() => {
  const Tags: QuartzComponent = ({ fileData, allFiles }) => {
    const tag = String(fileData.slug).replace(/^tags\/?/, "")
    const publicNotes = allFiles.filter(note => note.frontmatter?.publish === true && note.unlisted !== true && !note.frontmatter?.listing)
    const counts = new Map<string, number>()
    for (const note of publicNotes) for (const topic of new Set(note.frontmatter?.tags ?? [])) if (topic && !internalTags.has(topic)) counts.set(topic, (counts.get(topic) ?? 0) + 1)
    if (!tag || tag === "index") return <section class="tag-directory" dir="rtl"><p class="tag-intro">اختر موضوعاً لاستكشاف المقالات والتجارب والملاحظات المرتبطة به.</p><label class="topic-filter-label">ابحث عن موضوع<input class="topic-filter" type="search" placeholder="مثل: التضمين، الصوت، embedding…" /></label><p class="topic-filter-status" aria-live="polite"></p><div class="topic-grid">{[...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).map(([topic, count]) => <a href={tagURL(topic)} data-topic={topic}><span>{tagLabel(topic)}</span><small>{noteCount(count)}</small></a>)}</div></section>
    const notes = publicNotes.filter(note => (note.frontmatter?.tags ?? []).some(topic => topic === tag || topic.startsWith(tag + "/")))
      .sort((a, b) => String(b.frontmatter?.date ?? "").localeCompare(String(a.frontmatter?.date ?? "")))
    return <section class="tag-collection" dir="rtl"><div class="tag-intro"><p>مقالات وتجارب وملاحظات حول <strong>{tagLabel(tag)}</strong>.</p><div class="tag-collection-meta"><span>{noteCount(notes.length)}</span><span dir="ltr">#{tag}</span><a href="/tags/index.html">جميع المواضيع ←</a></div></div><div class="note-grid">{notes.map(note => {
      const href = `/${note.slug}.html`
      const topics = (note.frontmatter?.tags ?? []).filter(topic => topic && topic !== tag && !internalTags.has(topic)).slice(0, 3)
      const date = String(note.frontmatter?.date ?? "")
      return <article class="note-card" lang={String(note.frontmatter?.lang ?? "ar")} dir={note.frontmatter?.dir === "ltr" ? "ltr" : "rtl"}>
        <a class="note-cover" href={href} tabIndex={-1} aria-hidden="true">{note.frontmatter?.socialImage ? <img src={String(note.frontmatter.socialImage)} alt="" loading="lazy" decoding="async" /> : <span class="tag-cover-placeholder">{tagLabel(tag)}</span>}</a>
        <div class="note-card-body">{date && <time dateTime={date}>{date}</time>}<h2><a href={href}>{note.frontmatter?.title}</a></h2><p>{String(note.frontmatter?.description ?? note.description ?? "").slice(0, 240)}</p><div class="topic-links">{topics.map(topic => <a href={tagURL(topic)}>{tagLabel(topic)}</a>)}</div></div>
      </article>
    })}</div></section>
  }
  Tags.afterDOMLoaded = `(() => { const setup = () => { const input = document.querySelector('.topic-filter'); if (!input || input.dataset.bound) return; input.dataset.bound = 'true'; const cards = [...document.querySelectorAll('.topic-grid > a')]; const status = document.querySelector('.topic-filter-status'); const filter = () => { const query = input.value.trim().toLocaleLowerCase(); let count = 0; for (const card of cards) { card.hidden = !(card.textContent + ' ' + card.dataset.topic).toLocaleLowerCase().includes(query); if (!card.hidden) count++; } status.textContent = count === 1 ? 'موضوع واحد' : count === 2 ? 'موضوعان' : count ? count + ' موضوعاً' : 'لا توجد مواضيع مطابقة'; }; input.addEventListener('input', filter); filter(); }; document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', setup) : setup(); document.addEventListener('nav', setup); })();`
  return Tags
}) satisfies QuartzComponentConstructor
