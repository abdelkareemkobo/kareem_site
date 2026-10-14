import { QuartzComponent, QuartzComponentConstructor } from "./types"
import { pathToRoot } from "../util/path"

export default (() => {
  const Navigation: QuartzComponent = ({ fileData }) => {
    const root = pathToRoot(fileData.slug!)
    const links = [["الرئيسية", "index"], ["المدونة", "blog/feed"], ["ملاحظات", "til/index"], ["أبحاث", "papers"], ["عنّي", "about"], ["كتب", "publish"]]
    return <nav class="kareem-nav" dir="rtl" aria-label="التنقل الرئيسي">{links.map(([label, slug]) => <a href={`${root}/${slug}.html`} aria-current={fileData.slug === slug ? "page" : undefined}>{label}</a>)}</nav>
  }
  return Navigation
}) satisfies QuartzComponentConstructor
