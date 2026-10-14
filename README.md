# Kareem's personal website — Quartz

The personal website has been imported into Obsidian-readable Markdown. The original `Karem Site` Quarto checkout is untouched. The Arabic NLP book and `Abdelkareem-NNs` website are outside this migration.

## Write anywhere in your Obsidian vault

A Markdown note can live **anywhere in this vault**. Add only this property to publish it:

```yaml
publish: true
```

Normal `![[photo.png]]`, `![[Attachments/photo.png|320]]`, Markdown image links, `[[Note|Alias]]`, heading links and published note embeds are resolved against the vault. Only images/media referenced by approved notes are included. Links or embeds to unpublished notes become plain labels; their content and attachments are never exported. If two files have the same name and the link is ambiguous, use the full Obsidian vault path; the build reports the exact problem.

The first successful build adds a stable **permalink**, **section**, and (for a new article without one) **date** to the source note. Keep these properties when moving or renaming it. They preserve its public URL and RSS GUID independently of its current folder. These additions preserve the rest of the note and its YAML formatting. Existing imported URLs are retained. New notes get `/notes/<name>.html`; duplicate names get separate URLs automatically.

Move notes and images through Obsidian so it updates relative links. `alwaysUpdateLinks` has been enabled in this vault's `.obsidian/app.json`. Moving an image with a unique filename also resolves by filename, but ambiguous duplicate filenames need an updated path.

New notes default to `section: blog`. Use `section: til` for the TIL listing/feed, and optionally `series: embedding` for that series feed. These values follow the note when you move it. A missing publish flag, `publish: false`, or `draft: true` excludes it.

Hidden folders, `.obsidian`, `.trash`, dependency/build folders, and Quartz runtime files are excluded. `content/` remains a convenient home for the imported website notes; it is no longer a required publishing location. The original Quarto copies remain unchanged.

**A property change affects the next build/deployment.** It does not independently upload anything. While `npm run preview` is running, it watches notes and attachments throughout the vault and rebuilds on changes; refresh the browser to see them. Commit the refreshed approved export to the website repository to trigger deployment.

## Preview locally

From this project's folder, with Node 22+ installed:

```sh
npm ci
npm run preview
```

Open http://localhost:8088. Content edits rebuild the preview; refresh the browser afterward. Stop with Ctrl+C. To use another port: `PORT=8090 npm run preview`.

```sh
npm run build
npm run verify
npm run typecheck
```

The static website is generated in `public/`. Successful local builds also update `published/`, a portable copy containing only approved notes and their referenced media. This is generated automatically; edit the original Obsidian notes, not these copies. Verification checks every imported note's local links and embedded media, SEO metadata, JSON-LD, Arabic/English direction, RSS/sitemap presence, and exclusion of an unpublished note and unrelated attachment.

## What was migrated

- 68 published pages/articles/notes, including 6 notebook articles converted into Markdown.
- 11 draft articles left in the original site, listed in `migration/report.json`.
- Saved notebook code/output is preserved without execution. Two notebooks retain visualizations in `notebook-outputs/*.htm` embeds. New articles do not need notebooks or Quarto.
- Arabic fonts are self-hosted. Each note has its own `lang`/`dir`; code remains left-to-right.
- Blog, TIL, and embedding listings are generated from approved notes using section/series metadata, independently of folder locations.
- Titles, descriptions, dates, social images, Person/BlogPosting structured data, and canonical `.html` URLs are retained or reconstructed.
- `/blog/feed.xml`, `/til/index.xml`, and `/blog/posts/nlp/embedding_world/index.xml` retain their separate feeds. `/index.xml` combines dated writing.
- Filename normalization (case/spaces) produces permanent redirect rules in `public/_redirects` for Netlify. `migration/legacy-urls.json` retains the original public paths, and `migration/urls.json` records the applicable redirects.
- The existing Giscus repository/category and pathname mapping are configured. Confirm discussion continuity for normalized URLs before the live switch. Google Analytics uses the existing tag only when `KAREEM_PRODUCTION=1`.
- Two broken image filenames and incorrect internal links were repaired in the imported copies. Links to missing open-source pages now point to the author's GitHub repositories. Links to private drafts were removed or rendered as draft text.

This is a working Quartz preview, not an exact copy of the Quarto theme. Portfolio pages use Markdown; Bootstrap wrappers and Quarto-specific listing settings were removed. The 404 page currently uses Quartz's default design. The imported content is not re-executed or fact-checked.

## Publish to GitHub Pages

The website repository is `abdelkareemkobo/kareem_site`. Its `master` branch builds and deploys `public/` through `.github/workflows/deploy.yml` to the existing `kareemai.com` domain. Pull requests run verification without publishing.

Before pushing, run `npm run build`, `npm run verify`, and `npm run typecheck`, then commit the updated `published/` export. Cloud builds use `KAREEM_PUBLISHED_ONLY=1` and never read the private vault. Only approved notes and referenced media belong in this public repository; keep unpublished notes in the vault outside this checkout.

The Things theme uses pinned Quartz theme packages, a cached stylesheet, Arabic fonts, and responsive site navigation. Graph and backlinks panels are collapsed by default. GitHub Pages retains legacy `.html` addresses with canonical, noindex redirect pages; these are browser redirects because Pages does not support custom HTTP redirect rules. RSS keeps the original article identities.

The original Quarto history and `gh-pages` branch are retained for rollback. `netlify.toml` remains an optional alternative host configuration.

## Implementation notes

Quartz is based on the official v5.0.0 tag. Community plugins use official compiled npm releases pinned by `package-lock.json`; `scripts/setup-plugins.mjs` connects them to Quartz. This avoids the upstream Git installer's problem with spaces in vault paths. Small core adjustments add canonical/social metadata, per-page direction, shared frontmatter types, and support for the ignored publishing snapshot. Keep these changes when updating Quartz.

`scripts/migrate-quarto.py` is an import record, not part of the publishing workflow. It refuses to overwrite existing notes. Publishing requires only Node/npm; Pandoc, Python, and Quarto are not required.

## Regression checks

`npm run test:publishing` exercises arbitrary vault locations, moving/renaming notes and images, duplicate basenames, image paths with spaces, aliases/headings/transclusions, and privacy. It also renders a real Quartz fixture before and after a move and confirms that full-content RSS keeps the same GUID and working image URLs. `npm run verify` runs these checks and audits the generated personal site.

## SEO, performance and Arabic review — 14 October 2026

The comparison uses all 68 imported pages in the saved Quarto `_site` and generated Quartz output. It measures metadata and file sizes, not search rankings or field Core Web Vitals. Run `node scripts/audit-seo.mjs` to refresh `migration/seo-after.json`; the original Quartz baseline is `migration/seo-before.json`. `migration/seo-metadata-comparison.json` records the page metadata from both versions.

| Check | Old Quarto | Quartz before this review | Quartz after fixes |
| --- | ---: | ---: | ---: |
| Pages with canonical, description and social image | 68 | 68 | 68 |
| Dated posts with Article/BlogPosting schema | 61 | 61 | 61 |
| Pages with structured data | 62 | 62 | 68 |
| Homepage with WebSite name schema | 0 | 0 | 1 |
| Search index, uncompressed | 731 KB | 2,852 KB | 543 KB, loaded on demand |
| Navigation index, uncompressed | — | shared full search index | 22.5 KB (about 5 KB gzip) |
| Combined HTML for the 68 pages | 11.21 MB | 3.69 MB | 3.61 MB |
| Homepage HTML | 54.9 KB | 48.3 KB | 45.6 KB |

All 68 descriptions retain their original wording, allowing for smart quotation marks. Existing social pictures are carried through the attachment system; pages without a specific picture use the author's default card. Added WebSite/Person/WebPage relationships, article dates in ISO format, author profile URLs, breadcrumbs, correct Open Graph site-name attributes, social image dimensions, and per-section RSS discovery. The sitemap lists canonical note URLs and uses explicit `modified`/`updated` dates, falling back to publication dates; it never stamps every page with the build time. Set `modified: YYYY-MM-DD` in Obsidian after a significant content update.

Restored the historical `zaraah.html` redirect and consolidated `/index.html` and extensionless plugin links with permanent redirects. Netlify Pretty URLs are disabled to preserve the existing `.html` canonical URLs. Both deploy previews and branch previews receive a noindex header. Other hosts need equivalent redirect and preview-indexing settings.

Full-content RSS and exact Quarto GUIDs are retained at all four endpoints. The old blog, TIL and embedding feed GUID sets were compared against the new feeds with no missing identities. Arbitrary vault locations, moved images, stable permalinks and publication isolation remain covered by the publishing tests.

Removed duplicated HTML from the search index and stopped downloading/indexing article text on initial page load. Explorer and graph use the small navigation index. The graph and its external libraries initialize when expanded. Only resources for components used in page layouts are bundled. Pages without math, Mermaid diagrams or code omit the corresponding unnecessary resources. Local images get dimensions and async decoding; later images load lazily. Large raster images get responsive WebP variants while their original files and note links remain intact. In the browser, the Sakr article selected a 69 KB WebP instead of the original 1.84 MB JPEG at the desktop viewport.

Hash-addressed media has long-lived caching. HTML, scripts, CSS, indexes and feeds revalidate; unversioned fonts no longer have an immutable one-year cache. The favicon is a small PNG instead of the old SVG stored under an `.ico` extension.

Corrected Arabic interface labels, search results/empty states, reading-time plurals, folder names, copy-code labels and error text. Local plugin copies use the site's Arabic locale, since community plugins bundle separate dictionaries. The books page is now consistently Arabic, research-page headings and technical translations were corrected, and the main pages use “الوكلاء الأذكياء” and “التعلّم الاتحادي”. Official paper/book titles and the original language of blog articles are preserved. Table-of-contents navigation is enabled for notes that request it.

Browser checks covered desktop/mobile layout, mobile menu open/close, Arabic and English search, localized empty results, search-result redirects to canonical URLs, graph expansion, and responsive image selection. The mobile pages had equal document and viewport widths. Three remotely hosted article images still lack intrinsic dimensions; local images all have dimensions. No remote images were downloaded or modified. Screenshots are saved as `migration/seo-desktop.jpg` and `migration/seo-mobile.jpg`.

The SEO additions follow Google's documentation for [article structured data](https://developers.google.com/search/docs/appearance/structured-data/article), [site names](https://developers.google.com/search/docs/appearance/site-names), [canonical URLs](https://developers.google.com/search/docs/crawling-indexing/consolidate-duplicate-urls), and [accurate sitemap modification dates](https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap). Image work follows [web.dev's LCP guidance](https://web.dev/articles/optimize-lcp). URL hosting behavior follows [Netlify's post-processing documentation](https://docs.netlify.com/build/post-processing/overview/).

These changes are local. The live Quarto deployment and the other vault websites have not been changed. Search Console indexing, real Core Web Vitals, Giscus behavior and redirects on the actual host still need checking when the reviewed site is deployed.
