--- Shared formatting utilities for debug views
local M = {}

--- Define theme-safe highlight groups (using default=true so user themes can override)
function M.setup_highlights()
  vim.api.nvim_set_hl(0, "NeuralOpenTitle", { default = true, link = "Title" })
  vim.api.nvim_set_hl(0, "NeuralOpenLabel", { default = true, link = "Special" })
  vim.api.nvim_set_hl(0, "NeuralOpenTableHeader", { default = true, link = "DiagnosticError" })
end

--- Add a section title line with Title highlighting
---@param lines string[]
---@param hl table[]
---@param text string
function M.add_title(lines, hl, text)
  table.insert(lines, text)
  table.insert(hl, { row = #lines, col = 0, end_col = #text, group = "NeuralOpenTitle" })
end

--- Add a "  label: value" line with Label highlighting on "label:"
---@param lines string[]
---@param hl table[]
---@param label string
---@param value string
---@param indent? number Number of leading spaces (default 2)
function M.add_label(lines, hl, label, value, indent)
  local prefix = string.rep(" ", indent or 2)
  local text = prefix .. label .. ": " .. value
  table.insert(lines, text)
  table.insert(hl, {
    row = #lines,
    col = #prefix,
    end_col = #prefix + #label + 1,
    group = "NeuralOpenLabel",
  })
end

--- Format a snake_case feature name for display (e.g. "not_current" -> "Not current")
---@param name string
---@return string
function M.format_feature_name(name)
  local formatted = name:gsub("_", " "):gsub("(%l)(%u)", "%1 %2")
  return formatted:sub(1, 1):upper() .. formatted:sub(2)
end

--- Format and append an auto-aligned table.
---
--- Takes a 2D array of cell strings. The first row is the header.
--- Measures column widths automatically and pads accordingly.
---
--- Header row gets NeuralOpenTableHeader highlighting.
--- First column of data rows gets NeuralOpenLabel highlighting.
---
---@param lines string[] Output line buffer to append to
---@param hl table[] Output highlight buffer to append to
---@param rows string[][] 2D array of cell strings; rows[1] is the header
---@param opts? { indent?: number, align?: string[], gap?: number }
---   indent: leading spaces per line (default 2)
---   align: per-column "left" or "right" (default: first col "left", rest "right")
---   gap: spaces between columns (default 2)
function M.format_table(lines, hl, rows, opts)
  if #rows == 0 then
    return
  end

  opts = opts or {}
  local indent = opts.indent or 2
  local gap = opts.gap or 2
  local prefix = string.rep(" ", indent)
  local gap_str = string.rep(" ", gap)

  -- Determine number of columns from widest row
  local num_cols = 0
  for _, row in ipairs(rows) do
    if #row > num_cols then
      num_cols = #row
    end
  end

  -- Measure max width per column
  local col_widths = {}
  for c = 1, num_cols do
    col_widths[c] = 0
  end
  for _, row in ipairs(rows) do
    for c = 1, num_cols do
      local cell = row[c] or ""
      if #cell > col_widths[c] then
        col_widths[c] = #cell
      end
    end
  end

  -- Determine alignment per column
  local align = opts.align or {}
  for c = 1, num_cols do
    if not align[c] then
      align[c] = c == 1 and "left" or "right"
    end
  end

  -- Format each row
  for r, row in ipairs(rows) do
    local parts = {}
    for c = 1, num_cols do
      local cell = row[c] or ""
      local w = col_widths[c]
      if align[c] == "right" then
        parts[c] = string.rep(" ", w - #cell) .. cell
      else
        -- Don't pad the last column
        if c == num_cols then
          parts[c] = cell
        else
          parts[c] = cell .. string.rep(" ", w - #cell)
        end
      end
    end

    local text = prefix .. table.concat(parts, gap_str)
    table.insert(lines, text)

    if r == 1 then
      -- Header row
      table.insert(hl, { row = #lines, col = 0, end_col = #text, group = "NeuralOpenTableHeader" })
    else
      -- Data row: highlight first column as label
      local label_end = indent + col_widths[1]
      table.insert(hl, { row = #lines, col = 0, end_col = label_end, group = "NeuralOpenLabel" })
    end
  end
end

--- Shorten a path for display: fnamemodify(":~:.") + "..." prefix if too long
---@param path string
---@param max_width number
---@return string
function M.truncate_path(path, max_width)
  local display = vim.fn.fnamemodify(path, ":~:.")
  if #display > max_width then
    display = "..." .. display:sub(-(max_width - 3))
  end
  return display
end

--- Append a titled, ranked list section (top 10, sorted by .score descending).
--- Skipped entirely when items is empty.
---@param lines string[]
---@param hl table[]
---@param title string Section title
---@param items table[] Items; sorted by .score desc when first item has .score
---@param format_entry fun(i: number, item: table): string Row formatter
function M.append_ranked_list(lines, hl, title, items, format_entry)
  if #items == 0 then
    return
  end

  if items[1].score then
    table.sort(items, function(a, b)
      return a.score > b.score
    end)
  end

  M.add_title(lines, hl, title)
  table.insert(lines, "")
  for i = 1, math.min(10, #items) do
    table.insert(lines, format_entry(i, items[i]))
  end
  table.insert(lines, "")
end

--- Append a "Features: / Value" table sorted by value descending.
--- Shared by naive and nn (fallback) debug views.
---@param lines string[]
---@param hl table[]
---@param features_map table<string, number> Feature name -> normalized value
function M.append_feature_value_table(lines, hl, features_map)
  local sorted = {}
  for name, value in pairs(features_map) do
    table.insert(sorted, { name = name, value = value })
  end
  table.sort(sorted, function(a, b)
    return a.value > b.value
  end)

  local rows = { { "Features:", "Value" } }
  for _, f in ipairs(sorted) do
    table.insert(rows, { M.format_feature_name(f.name), string.format("%.4f", f.value) })
  end

  table.insert(lines, "")
  M.format_table(lines, hl, rows)
  table.insert(lines, "")
end

return M
