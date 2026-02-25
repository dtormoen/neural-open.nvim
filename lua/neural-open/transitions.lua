--- File transition tracking module
--- Manages history of file transitions (from -> to) for scoring
local M = {}

--- Normalize a file path for consistent storage
---@param path string
---@return string
local function normalize_path(path)
  -- Use pcall to safely check for vim.fs.normalize
  local has_normalize = false
  local ok = pcall(function()
    has_normalize = vim.fs ~= nil and vim.fs.normalize ~= nil
  end)

  if ok and has_normalize then
    return vim.fs.normalize(path, { expand_env = false })
  else
    return vim.fn.fnamemodify(path, ":p")
  end
end

--- Record a transition from source file to destination file with optional latency tracking
--- Appends to transition_history array, removes oldest if over limit
---@param from_path string Source file path
---@param to_path string Destination file path
---@param latency_ctx? table Optional latency context
function M.record_transition(from_path, to_path, latency_ctx)
  local latency = require("neural-open.latency")

  -- Normalize paths for consistent storage (measure if it involves filesystem)
  local result, ok = latency.measure(latency_ctx, "transitions.normalize_paths", function()
    return { normalize_path(from_path), normalize_path(to_path) }
  end, "async.transition_record")

  local normalized_from, normalized_to
  if ok then
    normalized_from, normalized_to = result[1], result[2]
  else
    -- Fallback if measure failed
    normalized_from = normalize_path(from_path)
    normalized_to = normalize_path(to_path)
  end

  -- Load current weights data (includes transition_history)
  latency.start(latency_ctx, "transitions.get_weights", "async.transition_record")
  local db = require("neural-open.db")
  local all_weights = db.get_weights(latency_ctx) or {}
  latency.finish(latency_ctx, "transitions.get_weights")

  -- Get or initialize transition_history array
  local history = all_weights.transition_history or {}

  -- Append new transition
  table.insert(history, {
    from = normalized_from,
    to = normalized_to,
    timestamp = os.time(),
  })

  -- Enforce ring buffer limit
  local config = require("neural-open").config
  local max_size = config.transition_history_size or 200

  -- Remove oldest entries if over limit
  while #history > max_size do
    table.remove(history, 1) -- Remove from front (oldest)
  end

  -- Save back to weights file
  latency.start(latency_ctx, "transitions.save_weights", "async.transition_record")
  all_weights.transition_history = history
  db.save_weights(all_weights, latency_ctx)
  latency.finish(latency_ctx, "transitions.save_weights")
end

--- Compute transition scores from a source file to all destinations
--- Returns map of {destination_path -> score} where score = 1-1/(1+count)
---@param source_path string Source file path
---@return table<string, number> Map of destination paths to scores
function M.compute_scores_from(source_path)
  -- Normalize source path
  local normalized_source = normalize_path(source_path)

  -- Load transition history from weights
  local db = require("neural-open.db")
  local all_weights = db.get_weights() or {}
  local history = all_weights.transition_history or {}

  -- Count transitions: source -> each destination
  local counts = {}
  for _, entry in ipairs(history) do
    if entry.from == normalized_source then
      local dest = entry.to
      counts[dest] = (counts[dest] or 0) + 1
    end
  end

  -- Convert counts to scores: 1 - 1/(1 + count)
  local scores = {}
  for dest, count in pairs(counts) do
    scores[dest] = 1 - 1 / (1 + count)
  end

  return scores
end

return M
