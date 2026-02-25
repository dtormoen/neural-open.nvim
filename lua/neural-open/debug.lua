local M = {}

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

  ctx.preview:reset()
  ctx.preview:minimal()

  local lines = {}

  -- File header
  table.insert(lines, "📄 " .. (item.text or item.file))
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

  -- Use algorithm-specific debug view
  local algorithm_lines = algorithm.debug_view(item, all_items)
  for _, line in ipairs(algorithm_lines) do
    table.insert(lines, line)
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
    table.insert(lines, "ℹ️  Metadata:")
    table.insert(lines, "")
    for _, meta in ipairs(metadata) do
      table.insert(lines, "  " .. meta)
    end
    table.insert(lines, "")
  end

  -- Trigram similarity details
  if item.nos and item.nos.raw_features and item.nos.raw_features.trigram and item.nos.raw_features.trigram > 0 then
    table.insert(lines, "🔤 Trigram Similarity:")
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

        -- Find common trigrams
        local common_trigrams = {}
        for trigram in pairs(target_trigrams) do
          if current_trigrams[trigram] then
            table.insert(common_trigrams, trigram)
          end
        end
        table.sort(common_trigrams)

        -- Display info
        table.insert(lines, string.format("  Current file: %s", current_virtual_name))
        table.insert(lines, string.format("  Target file:  %s", virtual_name))
        table.insert(lines, string.format("  Dice coefficient: %.4f", item.nos.raw_features.trigram))
        table.insert(
          lines,
          string.format(
            "  Common trigrams (%d): %s",
            #common_trigrams,
            #common_trigrams > 0
                and table.concat(common_trigrams, ", "):sub(1, 60) .. (#common_trigrams > 10 and "..." or "")
              or "none"
          )
        )
        table.insert(lines, "")
      end
    end
  end

  -- Recent files list
  table.insert(lines, "🕐 10 Most Recent Files:")
  table.insert(lines, "")

  local recent_module = require("neural-open.recent")
  local recent_list = recent_module.get_recency_list()

  for i = 1, math.min(10, #recent_list) do
    local display_path = vim.fn.fnamemodify(recent_list[i], ":~:.")
    if #display_path > 60 then
      display_path = "..." .. display_path:sub(-57)
    end
    table.insert(lines, string.format("  %2d. %s", i, display_path))
  end
  table.insert(lines, "")

  -- File preview (first few lines)
  table.insert(lines, "👁️  File Preview:")
  table.insert(lines, "")

  local file_lines = {}
  local ok, file_handle = pcall(io.open, item.file, "r")
  if ok and file_handle then
    local count = 0
    for line in file_handle:lines() do
      count = count + 1
      table.insert(file_lines, string.format("%3d  %s", count, line))
      if count >= 10 then -- Show first 10 lines
        break
      end
    end
    file_handle:close()
  else
    table.insert(file_lines, "  [Unable to read file]")
  end

  for _, line in ipairs(file_lines) do
    table.insert(lines, line)
  end

  ctx.preview:set_lines(lines)
  ctx.preview:set_title("Neural Open Debug")
  ctx.preview:highlight({ ft = "text" })
end

return M
