local M = {}

local ns = vim.api.nvim_create_namespace("neural-open.debug")
local fmt = require("neural-open.debug_fmt")

--- Apply collected highlights as extmarks to the preview buffer
---@param buf number Buffer handle
---@param highlights table[] Array of {row, col, end_col, group}
local function apply_highlights(buf, highlights)
  for _, h in ipairs(highlights) do
    pcall(vim.api.nvim_buf_set_extmark, buf, ns, h.row - 1, h.col, {
      end_col = h.end_col,
      hl_group = h.group,
    })
  end
end

--- Generate a detailed debug preview for a picker item
--- Shows raw features, normalized features, weighted components, and scoring calculations
---@param ctx table The Snacks picker context containing item and preview
function M.debug_preview(ctx)
  local item = ctx.item
  if not item or not item.file then
    ctx.preview:reset()
    ctx.preview:set_lines({ "No item selected" })
    return
  end

  fmt.setup_highlights()
  ctx.preview:reset()
  ctx.preview:minimal()

  local lines = {}
  local hl = {}

  -- File preview at top (first 10 lines with syntax highlighting)
  fmt.add_title(lines, hl, "Preview")
  table.insert(lines, "")

  local file_content_lines = {} -- raw lines for syntax highlighting
  local file_start_row = #lines -- 0-indexed row where file content starts
  local file_line_count = 0

  local ok, file_handle = pcall(io.open, item.file, "r")
  if ok and file_handle then
    for line in file_handle:lines() do
      file_line_count = file_line_count + 1
      table.insert(file_content_lines, line)
      table.insert(lines, string.format("  %3d  %s", file_line_count, line))
      if file_line_count >= 10 then
        break
      end
    end
    file_handle:close()
  else
    table.insert(lines, "  [Unable to read file]")
  end
  table.insert(lines, "")

  -- Get the algorithm from the item's context if available,
  -- otherwise load from config
  local algorithm
  if item.nos and item.nos.ctx and item.nos.ctx.algorithm then
    algorithm = item.nos.ctx.algorithm
  else
    local registry = require("neural-open.algorithms.registry")
    algorithm = registry.get_algorithm()
  end

  -- Get all ranked items from the picker
  local all_items = ctx.picker:items()

  -- Get the context from the item
  local ctx_data = nil
  if item and item.nos and item.nos.ctx then
    ctx_data = item.nos.ctx
  end

  -- Use algorithm-specific debug view (may return lines, highlights or just lines)
  local algorithm_lines, algorithm_hl = algorithm.debug_view(item, all_items)
  local row_offset = #lines
  for _, line in ipairs(algorithm_lines) do
    table.insert(lines, line)
  end
  -- Merge algorithm highlights with row offset
  if algorithm_hl then
    for _, h in ipairs(algorithm_hl) do
      table.insert(hl, {
        row = h.row + row_offset,
        col = h.col,
        end_col = h.end_col,
        group = h.group,
      })
    end
  end
  table.insert(lines, "")

  -- Additional metadata
  local metadata = {}
  if item.nos and item.nos.is_open_buffer then
    table.insert(metadata, "Open Buffer")
  end
  if item.nos and item.nos.is_alternate then
    table.insert(metadata, "Alternate Buffer")
  end
  if item.nos and item.nos.recent_rank then
    table.insert(metadata, "Recent Rank: " .. item.nos.recent_rank)
  end

  if #metadata > 0 then
    fmt.add_title(lines, hl, "Metadata")
    table.insert(lines, "")
    for _, meta in ipairs(metadata) do
      table.insert(lines, "  " .. meta)
    end
    table.insert(lines, "")
  end

  -- Trigram similarity details
  if item.nos and item.nos.raw_features and item.nos.raw_features.trigram and item.nos.raw_features.trigram > 0 then
    fmt.add_title(lines, hl, "Trigram Similarity")
    table.insert(lines, "")

    -- Get the virtual names from stored data
    local virtual_name = item.nos.virtual_name
    local current_virtual_name = ctx_data and ctx_data.current_file_virtual_name or ""

    if not virtual_name or not current_virtual_name then
      table.insert(lines, "  [Trigram data incomplete - virtual names not available]")
      table.insert(lines, "")
    else
      -- Use pre-computed trigrams from context if available
      local current_trigrams = ctx_data and ctx_data.current_file_trigrams

      if not current_trigrams then
        table.insert(lines, "  [Trigram data incomplete - current file trigrams not available]")
        table.insert(lines, "")
      else
        -- Compute target trigrams (since we don't store them)
        local trigrams_module = require("neural-open.trigrams")
        local target_trigrams = trigrams_module.compute_trigrams(virtual_name)

        -- Find common trigrams and decode packed integer keys for display
        local common_trigrams = {}
        for trigram in pairs(target_trigrams) do
          if current_trigrams[trigram] then
            table.insert(common_trigrams, trigrams_module.unpack_trigram(trigram))
          end
        end
        table.sort(common_trigrams)

        fmt.add_label(lines, hl, "Current file", current_virtual_name)
        fmt.add_label(lines, hl, "Target file", virtual_name)
        fmt.add_label(lines, hl, "Dice coefficient", string.format("%.4f", item.nos.raw_features.trigram))
        fmt.add_label(
          lines,
          hl,
          string.format("Common trigrams (%d)", #common_trigrams),
          #common_trigrams > 0
              and table.concat(common_trigrams, ", "):sub(1, 60) .. (#common_trigrams > 10 and "..." or "")
            or "none"
        )
        table.insert(lines, "")
      end
    end
  end

  -- Transitions (Current File)
  if ctx_data and ctx_data.current_file and ctx_data.current_file ~= "" then
    local transitions = require("neural-open.transitions")
    local scores = transitions.compute_scores_from(ctx_data.current_file)

    local items = {}
    for dest, score in pairs(scores) do
      table.insert(items, { dest = dest, score = score })
    end

    fmt.append_ranked_list(lines, hl, "Transitions (Current File)", items, function(i, entry)
      return string.format("  %2d. %-50s  %.4f", i, fmt.truncate_path(entry.dest, 50), entry.score)
    end)
  end

  -- Transitions (All Files)
  local db = require("neural-open.db")
  local all_weights = db.get_weights() or {}
  local transition_frecency = all_weights.transition_frecency
  if transition_frecency then
    local now = os.time()
    local half_life = 30 * 24 * 3600
    local lambda = math.log(2) / half_life

    local all_pairs = {}
    for source, destinations in pairs(transition_frecency) do
      for dest, deadline in pairs(destinations) do
        local raw_score = math.exp(lambda * (deadline - now))
        local normalized = 1 - 1 / (1 + raw_score / 4)
        table.insert(all_pairs, { source = source, dest = dest, score = normalized })
      end
    end

    fmt.append_ranked_list(lines, hl, "Transitions (All Files)", all_pairs, function(i, entry)
      return string.format(
        "  %2d. %-25s -> %-25s  %.4f",
        i,
        fmt.truncate_path(entry.source, 25),
        fmt.truncate_path(entry.dest, 25),
        entry.score
      )
    end)
  end

  -- Frecent files (from Snacks frecency database, normalized to 0-1)
  local frecency_ok, frecency_mod = pcall(require, "snacks.picker.core.frecency")
  if frecency_ok then
    local inst_ok, frecency_inst = pcall(frecency_mod.new)
    if inst_ok and frecency_inst and frecency_inst.cache then
      local frecent_files = {}
      for path, deadline in pairs(frecency_inst.cache) do
        local raw = frecency_inst:to_score(deadline)
        if raw > 0 then
          table.insert(frecent_files, { path = path, score = 1 - 1 / (1 + raw / 8) })
        end
      end

      fmt.append_ranked_list(lines, hl, "Frecent Files", frecent_files, function(i, entry)
        return string.format("  %2d. %-55s  %.4f", i, fmt.truncate_path(entry.path, 55), entry.score)
      end)
    end
  end

  -- Recent files list
  local recent_module = require("neural-open.recent")
  local recent_list = recent_module.get_recency_list()

  local recent_items = {}
  for _, path in ipairs(recent_list) do
    table.insert(recent_items, { path = path })
  end

  fmt.append_ranked_list(lines, hl, "Recent Files", recent_items, function(i, entry)
    return string.format("  %2d. %s", i, fmt.truncate_path(entry.path, 60))
  end)

  ctx.preview:set_lines(lines)
  ctx.preview:set_title("Neural Open Debug")

  -- Apply extmarks only if we have a real buffer (not in test mocks)
  local buf = ctx.preview.win and ctx.preview.win.buf
  if not buf then
    return
  end

  -- Apply syntax highlighting to file preview lines
  if file_line_count > 0 then
    local raw_code = table.concat(file_content_lines, "\n")
    local snacks_hl = require("snacks.picker.util.highlight")
    local hl_ok, extmarks = pcall(snacks_hl.get_highlights, {
      code = raw_code,
      file = item.file,
    })
    if hl_ok and extmarks then
      local col_offset = 7 -- "  NNN  " prefix is 7 chars
      for row, marks in pairs(extmarks) do
        local buf_row = file_start_row + row - 1 -- extmarks are 1-indexed, buf is 0-indexed
        for _, mark in ipairs(marks) do
          pcall(vim.api.nvim_buf_set_extmark, buf, ns, buf_row, mark.col + col_offset, {
            end_col = mark.end_col + col_offset,
            hl_group = mark.hl_group,
            priority = mark.priority,
          })
        end
      end
    end
  end

  -- Apply all collected highlights
  apply_highlights(buf, hl)
end

return M
