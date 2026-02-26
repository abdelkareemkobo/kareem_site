local stringify = pandoc.utils.stringify

local type_map = {
  note = "note",
  info = "note",
  abstract = "note",
  summary = "note",
  tldr = "note",
  quote = "note",
  cite = "note",
  example = "note",
  question = "note",
  todo = "note",
  tip = "tip",
  hint = "tip",
  success = "tip",
  warning = "warning",
  attention = "warning",
  caution = "caution",
  important = "important",
  danger = "warning",
  failure = "important",
  bug = "warning",
  error = "important",
}

local function pretty_type(t)
  if t == "tldr" then
    return "TL;DR"
  end
  t = t:gsub("_", "-")
  local parts = {}
  for part in t:gmatch("[^%-]+") do
    if #part > 0 then
      table.insert(parts, part:sub(1, 1):upper() .. part:sub(2))
    end
  end
  if #parts == 0 then
    return "Callout"
  end
  return table.concat(parts, " ")
end

local function split_inlines_at_break(inlines)
  local head = {}
  local tail = {}
  local found = false
  for _, inline in ipairs(inlines) do
    if not found and (inline.t == "LineBreak" or inline.t == "SoftBreak") then
      found = true
    elseif not found then
      table.insert(head, inline)
    else
      table.insert(tail, inline)
    end
  end
  return head, tail, found
end

function BlockQuote(el)
  if #el.content == 0 then
    return nil
  end

  local first = el.content[1]
  if first.t ~= "Para" then
    return nil
  end

  local head, tail, has_break = split_inlines_at_break(first.content)
  local header_text = stringify(head)

  local raw_type, fold, title = header_text:match("^%[!([%w%-]+)%]([%+%-]?)%s*(.*)$")
  if not raw_type then
    return nil
  end

  local callout_type = raw_type:lower()
  local mapped_type = type_map[callout_type] or callout_type

  if title == nil or title == "" then
    title = pretty_type(callout_type)
  end

  local body_blocks = {}
  if has_break and #tail > 0 then
    table.insert(body_blocks, pandoc.Para(tail))
  end
  for i = 2, #el.content do
    table.insert(body_blocks, el.content[i])
  end

  local icon_span = pandoc.Span({}, pandoc.Attr("", { "callout-icon" }))
  local icon_container = pandoc.Span({ icon_span }, pandoc.Attr("", { "callout-icon-container" }))
  local title_span = pandoc.Span({ pandoc.Str(title) }, pandoc.Attr("", { "callout-title" }))
  local title_para = pandoc.Para({ icon_container, title_span })
  local title_div = pandoc.Div({ title_para }, pandoc.Attr("", { "callout-title-container" }))
  local body_div = pandoc.Div(body_blocks, pandoc.Attr("", { "callout-body" }))

  local classes = { "callout", "callout-" .. mapped_type }
  if mapped_type ~= callout_type then
    table.insert(classes, "callout-" .. callout_type)
  end

  local attrs = { ["data-callout"] = callout_type }
  if fold ~= "" then
    attrs["data-callout-fold"] = fold
  end

  return pandoc.Div({ title_div, body_div }, pandoc.Attr("", classes, attrs))
end
