import { QuartzComponent, QuartzComponentConstructor } from "./types"

export default (() => {
  const Contents: QuartzComponent = ({ fileData }) => {
    if (!fileData.toc?.length || fileData.frontmatter?.toc === false || fileData.frontmatter?.listing || fileData.slug === "index") return null
    return <details class="kareem-contents" dir="rtl">
      <summary>في هذه الصفحة</summary>
      <ol>{fileData.toc.map(entry => <li class={`depth-${entry.depth}`}><a href={`#${entry.slug}`}>{entry.text}</a></li>)}</ol>
    </details>
  }
  return Contents
}) satisfies QuartzComponentConstructor
