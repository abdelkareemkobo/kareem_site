-- jsonld.lua - Auto-inject Article schema for blog posts
function Meta(meta)
  local title = pandoc.utils.stringify(meta.title or "")
  local date = pandoc.utils.stringify(meta.date or "")
  local description = pandoc.utils.stringify(meta.description or "")
  local author = pandoc.utils.stringify(meta.author or "Kareem Elkhateb")
  
  -- Skip if no title (not an article)
  if title == "" then return end
  
  -- Escape quotes in strings for JSON
  title = title:gsub('"', '\\"')
  description = description:gsub('"', '\\"')
  author = author:gsub('"', '\\"')
  
  local jsonld = string.format([[
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "Article",
  "headline": "%s",
  "description": "%s",
  "author": {"@type": "Person", "name": "%s"},
  "datePublished": "%s",
  "publisher": {
    "@type": "Person",
    "name": "Kareem Elkhateb",
    "url": "https://kareemai.com"
  }
}
</script>
]], title, description, author, date)
  
  -- Use header-includes which is the standard pandoc way
  local header_includes = meta["header-includes"]
  if header_includes == nil then
    header_includes = pandoc.List({})
  elseif pandoc.utils.type(header_includes) ~= "List" then
    header_includes = pandoc.List({header_includes})
  end
  
  header_includes:insert(pandoc.RawBlock("html", jsonld))
  meta["header-includes"] = header_includes
  
  return meta
end
