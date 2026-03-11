--- Finder that yields recent and frecent files for the neural_recent source.
local M = {}

local path_mod = require("neural-open.path")

---@param _opts table Finder options (unused)
---@param _ctx table Finder context (unused)
---@return table[] items Array of {file, text} items
function M.finder(_opts, _ctx)
  local recent = require("neural-open.recent")
  local recency_map = recent.get_recency_map()

  -- Collect recent files ordered by rank (ascending = most recent first)
  local ranked = {}
  for path, info in pairs(recency_map) do
    ranked[#ranked + 1] = { path = path, rank = info.recent_rank }
  end
  table.sort(ranked, function(a, b)
    return a.rank < b.rank
  end)

  local seen = {}
  local items = {}

  -- Add recent files first (most recent = highest priority)
  for _, entry in ipairs(ranked) do
    if not seen[entry.path] and vim.uv.fs_stat(entry.path) then
      seen[entry.path] = true
      items[#items + 1] = { file = entry.path, text = entry.path }
    end
  end

  -- Add frecent files from snacks frecency DB
  local frecency_ok, snacks_frecency = pcall(require, "snacks.picker.core.frecency")
  if frecency_ok then
    local inst_ok, frecency_inst = pcall(snacks_frecency.new)
    if inst_ok and frecency_inst and frecency_inst.cache then
      -- Collect and sort by score descending
      local frecent = {}
      for raw_path, deadline in pairs(frecency_inst.cache) do
        local score = frecency_inst:to_score(deadline)
        if score > 0 then
          frecent[#frecent + 1] = { path = path_mod.normalize(raw_path), score = score }
        end
      end
      table.sort(frecent, function(a, b)
        return a.score > b.score
      end)

      for _, entry in ipairs(frecent) do
        if not seen[entry.path] and vim.uv.fs_stat(entry.path) then
          seen[entry.path] = true
          items[#items + 1] = { file = entry.path, text = entry.path }
        end
      end
    end
  end

  return items
end

return M
