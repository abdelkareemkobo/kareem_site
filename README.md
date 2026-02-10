# Kareem Elkhateb Blog

<!-- - Basic changes -->
- move your Content from AstroAI
- Rss Feed
- Google Analytics
- Deployement live
<!-- - Change the Talk page into Papers -->
- My Github Projects
- Fix the website SEO and locale

## Level up

1. Adding Comment system
2. Buy me a coffee part
3. Add the View of post with Images in a different view :) 
   1. the secret page 

## Level 2

1. Sync with Obsidian Files
2. Quarto extension
3. Understand all Hamel Codebase

## Level 3

1. Obsidian shimmering focus
2. My own Newsletter with substack and hodhod

## Shimmering Focus Features (SF)

The site now matches the Shimmering Focus visual language and behavior:

- Typography: longform reading font for body, SF-like headings, improved line-height.
- Local fonts: Recursive + iA Writer Quattro + Amiri + JetBrains Mono bundled locally.
- Color system: SF accent applied to links, headings, bold text, and highlight.
- Highlight syntax: `==highlight==` renders with SF-style mark background.
- Foldable headings: h2-h4 are collapsible with a "Collapse all" control.
- Listings: small thumbnails, compact cards, SF-style hover and borders.
- Post hero image: frontmatter image appears under title on posts.
- Callouts: note/tip/warning/important styled like SF panels.
- Blockquotes: inset card style with accent border.
- Tables: soft borders and zebra striping.
- Code: inline code + code blocks styled to match SF contrast.
- Tags: category pills with subtle borders and hover tint.
- TOC: clean rail with active state highlight.
- UI polish: selection color, section dividers, soft card shadows.
- Performance: local fonts + link prefetch for faster navigation.

Main files to tweak:

- `styles.css` (visual system and SF components)
- `shimmering-fonts.css` (font faces)
- `_includes/fold-headings.html` (fold logic + collapse all)
- `_includes/post-hero-image.html` (post image)
