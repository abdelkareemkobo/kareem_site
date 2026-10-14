import { loadQuartzConfig, loadQuartzLayout } from "./quartz/plugins/loader/config-loader"
import { PageTypeDispatcher } from "./quartz/plugins/pageTypes/dispatcher"
import Navigation from "./quartz/components/KareemNavigation"
import Listing from "./quartz/components/KareemListing"
import Contents from "./quartz/components/KareemContents"
import { QuartzComponent } from "./quartz/components/types"
import { h, cloneElement, toChildArray } from "preact"
import { createHash } from "node:crypto"
import { transform } from "lightningcss"
import { write } from "./quartz/plugins/emitters/helpers"
import { FullSlug } from "./quartz/util/path"
import { BuildCtx } from "./quartz/util/ctx"
import { StaticResources } from "./quartz/util/resources"

const config = await loadQuartzConfig({ analytics: process.env.KAREEM_PRODUCTION === "1" ? { provider: "google", tagId: "G-XFPC3M9TMG" } : null })
export const layout = await loadQuartzLayout()
const navigation = Navigation()
navigation.afterDOMLoaded = `(() => {
  if (window.kareemHeaderReady) return;
  window.kareemHeaderReady = true;
  document.addEventListener("click", event => {
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest("button.darkmode")) {
      const theme = document.documentElement.getAttribute("saved-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("saved-theme", theme);
      localStorage.setItem("theme", theme);
      document.body.classList.remove("theme-light", "theme-dark");
      document.body.classList.add("theme-" + theme);
      document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
    }
    const toggle = target?.closest(".explorer-toggle");
    const explorer = toggle?.closest(".explorer");
    if (explorer) {
      const closed = explorer.classList.toggle("collapsed");
      explorer.setAttribute("aria-expanded", String(!closed));
      toggle.setAttribute("aria-expanded", String(!closed));
      document.documentElement.classList.toggle("mobile-no-scroll", !closed);
    }
  });
  document.addEventListener("keydown", event => {
    if (event.key !== "Escape") return;
    for (const explorer of document.querySelectorAll('.explorer[aria-expanded="true"]')) {
      explorer.classList.add("collapsed"); explorer.setAttribute("aria-expanded", "false");
      explorer.querySelector(".explorer-toggle")?.setAttribute("aria-expanded", "false");
    }
    document.documentElement.classList.remove("mobile-no-scroll");
  });
})();`
const listing = Listing()
const contents = Contents()
// Serve the full Things port once as a fingerprinted, cacheable stylesheet.
const theme = config.plugins.transformers.find(plugin => plugin.name === "QuartzTheme")
if (!theme?.externalResources) throw new Error("Things theme plugin is not enabled")
const originalThemeResources = theme.externalResources
let themeAssets: { resources: Partial<StaticResources>; css: Buffer; slug: FullSlug } | undefined
function getThemeAssets(ctx: BuildCtx) {
  if (themeAssets) return themeAssets
  const resources = originalThemeResources!(ctx) ?? {}
  let source = (resources.css ?? []).map(resource => resource.content).join("\n")
  // The port splits selector commas inside this special Obsidian task marker.
  for (const mode of ["dark", "light"]) source = source.replaceAll(`data-task=", html[saved-theme="${mode}"] "]`, 'data-task=","]')
  // The Things 1.0.1 export also has one extra closing brace at this boundary.
  source = source.replace(/\n}\n@layer quartz-themes-base \{/, "\n@layer quartz-themes-base {")
  // Keep the port's trailing unlayered compatibility rules below site layout fixes.
  const css = Buffer.from(transform({ filename: "things.css", code: Buffer.from(`@layer obsidian-theme { ${source} }`), minify: true }).code)
  const slug = `static/things-${createHash("sha256").update(css).digest("hex").slice(0, 16)}` as FullSlug
  return themeAssets = { resources, css, slug }
}
theme.externalResources = ctx => { const assets = getThemeAssets(ctx); return { ...assets.resources, css: [{ content: `/${assets.slug}.css` }] } }
config.plugins.emitters.push({ name: "KareemThingsTheme", async *emit(ctx) { const assets = getThemeAssets(ctx); yield await write({ ctx, slug: assets.slug, ext: ".css", content: assets.css }) } })
const metadataWrappers = new Map<QuartzComponent, QuartzComponent>()
const uiWrappers = new Map<QuartzComponent, QuartzComponent>()
function adaptUI(component: QuartzComponent): QuartzComponent {
  if (uiWrappers.has(component)) return uiWrappers.get(component)!
  const earlyScripts = Array.isArray(component.beforeDOMLoaded) ? component.beforeDOMLoaded : component.beforeDOMLoaded ? [component.beforeDOMLoaded] : []
  component.beforeDOMLoaded = earlyScripts.map(script => script.replace(/for\(let e of document\.getElementsByClassName\("darkmode"\)\)e\.addEventListener\("click",n\),window\.addCleanup\(\(\)=>e\.removeEventListener\("click",n\)\);/g, ""))
  const scripts = Array.isArray(component.afterDOMLoaded) ? component.afterDOMLoaded : component.afterDOMLoaded ? [component.afterDOMLoaded] : []
  component.afterDOMLoaded = scripts.map(original => {
    if (original.includes("[Graph]")) {
      const replacements = [
        ['text:iu.text,style:{fontSize:We*15,fill:ze,fontFamily:Ne}', 'text:_.classList.contains("global-graph-container")&&iu.text.length>44?iu.text.slice(0,43)+"…":iu.text,style:{fontSize:We*15,fill:ze,fontFamily:"Arial",wordWrap:true,wordWrapWidth:170,align:"center"}'],
        ['return 2+Math.sqrt(l)}).iterations(3)', 'return _.classList.contains("global-graph-container")?48:2+Math.sqrt(l)}).iterations(3)'],
        ['var nu=new a.Container', `await document.fonts.load('14px "Kareem UI"');var nu=new a.Container`],
        ['ou.on("tick",function(){})', 'ou.on("tick",function(){if(_.classList.contains("global-graph-container")){for(var node of ru){node.x=Math.max(-O/2+100,Math.min(O/2-100,node.x));node.y=Math.max(-G/2+160,Math.min(G/2-35,node.y))}}})'],
        ['var le=!1;function se()', 'var labelFrame=0;var le=!1;function se()'],
        ['for(var i=0;i<I.length;i++){var C=I[i]', 'if(_.classList.contains("global-graph-container")&&++labelFrame%4===0){var occupied=[];for(var visible of P){if(_u===visible.simulationData.id){visible.label.alpha=1;continue}var box=visible.label.getBounds();var overlaps=occupied.some(function(other){return box.x<other.x+other.width+8&&box.x+box.width+8>other.x&&box.y<other.y+other.height+6&&box.y+box.height+6>other.y});visible.label.alpha=overlaps?0:.9;if(!overlaps)occupied.push(box)}}for(var i=0;i<I.length;i++){var C=I[i]'],
        ['ee=h(Z.getPropertyValue("--lightgray").trim(),"#d4d4d4")', 'ee=h(document.documentElement.getAttribute("saved-theme")==="dark"?"#536576":"#c8d2dd","#c8d2dd")'],
        ['Du.alpha=0', 'Du.alpha=_.classList.contains("global-graph-container")?.9:0'],
        ['F=Math.max((l-1)/3.75,0)', 'F=_.classList.contains("global-graph-container")?Math.min(1,Math.max(.75,(l-1)/3.75)):Math.max((l-1)/3.75,0)'],
        ['Wu(F.id),T=C.alpha,Eu||Au()', 'Wu(F.id),T=C.alpha,C.text=F.text,Eu||Au()'],
        ['Wu(null),C.alpha=T,Eu||Au()', 'Wu(null),C.text=_.classList.contains("global-graph-container")&&F.text.length>44?F.text.slice(0,43)+"…":F.text,C.alpha=T,Eu||Au()'],
      ]
      let graphScript = original.replaceAll("await fetchData", "await fetchNavigationData")
      for (const [before, after] of replacements) {
        if (!graphScript.includes(before)) throw new Error("Graph plugin changed; review readable labels")
        graphScript = graphScript.replaceAll(before, after)
      }
      return graphScript + `;(() => { const enhance = () => { for (const panel of document.querySelectorAll('.global-graph-container')) { if (!panel.querySelector('canvas') || panel.querySelector('.graph-help')) continue; const help = document.createElement('div'); help.className = 'graph-help'; help.innerHTML = '<strong>استكشف الملاحظات</strong><span>اسحب للتحريك · مرّر للتكبير · اضغط على ملاحظة لفتحها</span>'; const close = document.createElement('button'); close.className = 'graph-close'; close.type = 'button'; close.setAttribute('aria-label', 'إغلاق خريطة الروابط'); close.textContent = '×'; close.addEventListener('click', () => document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))); panel.append(help, close); } }; enhance(); new MutationObserver(enhance).observe(document.body, { childList: true, subtree: true }); document.addEventListener('nav', enhance); })();`
    }
    if (original.includes("[Explorer]") || original.includes("[Graph]")) return original.replaceAll("await fetchData", "await fetchNavigationData")
      .replace('l.addEventListener("click",o),s.push(()=>l.removeEventListener("click",o))', "")
      .replace('n.checkVisibility&&n.checkVisibility()&&(t.classList.add("collapsed")', 'n.checkVisibility&&n.checkVisibility()&&t.getAttribute("aria-expanded")!=="true"&&(t.classList.add("collapsed")')
    // Flex groups copy their children's scripts when the layout is constructed.
    if (!original.includes('async function Mn(){Ai(),await Ti(),await wi(),Li()}')) return original
    const replacements = [
      ["Ai(),await Ti(),await wi(),Li()", "Ai(),await wi(),Li()"],
      ["W=_=>{Me=_", "W=_=>{Ti().catch(console.error);Me=_"],
      ["nt=async _=>{let A=_.target.value", "nt=async _=>{await Ti();let A=_.target.value"],
      ['"Search results"', '"نتائج البحث"'], ['"Tag suggestions"', '"وسوم مقترحة"'],
      ['"No results."', '"لا توجد نتائج"'], ['"Try another search term?"', '"جرّب كلمات أخرى"'],
      ['"No matching tags"', '"لا توجد وسوم مطابقة"'], ['"Filter by tag"', '"تصفية حسب الوسم"'],
    ]
    let script = original
    for (const [before, after] of replacements) {
      if (!script.includes(before) && before.includes("=")) throw new Error(`Search plugin changed; review site adaptation: ${before}`)
      script = script.replaceAll(before, after)
    }
    // Indexing is shared by clicks and keystrokes, including slow first searches.
    const initialization = "var Rt=!1;async function Ti(){Rt||(Z=await bi(),await ki(),Rt=!0)}"
    if (!script.includes(initialization)) throw new Error("Search plugin changed; review deferred indexing")
    script = script.replace(initialization, "var Rt=!1,siteSearchReady;function Ti(){return siteSearchReady??=(async()=>{Z=await bi();await ki();Rt=!0})().catch(e=>{siteSearchReady=undefined;throw e})}")
    return script
  })
  let adapted = component
  if (component.name === "Footer") {
    adapted = Object.assign(((props) => {
      const footer = component(props)
      if (!footer) return null
      return cloneElement(footer, {}, [h("p", { dir: "rtl", class: "kareem-support" }, "يمكنك دعمي بدعوة صادقة!"), ...toChildArray(footer.props.children).slice(1)])
    }) as QuartzComponent, component)
  }
  if (component.name === "Backlinks") {
    adapted = Object.assign(((props) => {
      const links = component(props)
      if (!links) return null
      const english = String(props.fileData.frontmatter?.lang ?? "ar").startsWith("en")
      return h("section", { class: "kareem-backlinks" }, h("h2", null, english ? "Pages linking here" : "صفحات تشير إلى هذه الملاحظة"), links)
    }) as QuartzComponent, component)
  }
  if (component.name === "Graph" && component.afterDOMLoaded) {
    const script = (component.afterDOMLoaded as string[]).join("\n")
    adapted = Object.assign(((props) => h("section", { class: "kareem-graph" }, component(props))) as QuartzComponent, component)
    adapted.afterDOMLoaded = script
    component.afterDOMLoaded = adapted.afterDOMLoaded
  }
  uiWrappers.set(component, adapted)
  return adapted
}
for (const slots of [layout.defaults, ...Object.values(layout.byPageType)]) {
  if (slots.footer) slots.footer = adaptUI(slots.footer)
  for (const position of ["left", "right", "beforeBody", "afterBody", "header"] as const) slots[position] = slots[position]?.map(adaptUI)
  slots.header = [...(slots.header ?? []), navigation]
  slots.right = slots.right?.filter(component => component.name !== "TableOfContents")
  slots.beforeBody = [...(slots.beforeBody ?? []), contents]
  slots.afterBody = [listing, ...(slots.afterBody ?? [])]
  slots.beforeBody = slots.beforeBody?.map(component => {
    if (component.name !== "ContentMetadata") return component
    let wrapped = metadataWrappers.get(component)
    if (!wrapped) {
      wrapped = Object.assign(((props) => props.fileData.frontmatter?.date ? component(props) : null) as QuartzComponent, component)
      metadataWrappers.set(component, wrapped)
    }
    return wrapped
  })
}
config.plugins.emitters = config.plugins.emitters.filter(e => e.name !== "PageTypeDispatcher")
config.plugins.emitters.push(PageTypeDispatcher(layout))
export default config
