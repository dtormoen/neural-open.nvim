--- File transition tracking with frecency-based scoring
local M = {}

local path_mod = require("neural-open.path")
local frecency_mod = require("neural-open.frecency")

-- Exposed for testing
M.MAX_DESTINATIONS_PER_SOURCE = 50
M.MAX_SOURCES = 200

--- Record a transition from source file to destination file.
--- Each visit adds 1 to the frecency score, stored as a deadline for automatic decay.
---@param from_path string Source file path
---@param to_path string Destination file path
---@param latency_ctx? table Optional latency context
function M.record_transition(from_path, to_path, latency_ctx)
  local latency = require("neural-open.latency")

  local result, ok = latency.measure(latency_ctx, "transitions.normalize_paths", function()
    return { path_mod.normalize(from_path), path_mod.normalize(to_path) }
  end, "async.transition_record")

  local normalized_from, normalized_to
  if ok then
    normalized_from, normalized_to = result[1], result[2]
  else
    normalized_from = path_mod.normalize(from_path)
    normalized_to = path_mod.normalize(to_path)
  end

  latency.start(latency_ctx, "transitions.get_tracking", "async.transition_record")
  local db = require("neural-open.db")
  local tracking = db.get_tracking("files", latency_ctx) or {}
  latency.finish(latency_ctx, "transitions.get_tracking")

  local now = os.time()
  local frecency = tracking.transition_frecency or {}
  local destinations = frecency[normalized_from] or {}

  destinations[normalized_to] = frecency_mod.bump(destinations[normalized_to], now)

  frecency_mod.prune_map(destinations, M.MAX_DESTINATIONS_PER_SOURCE, now)
  frecency[normalized_from] = destinations
  frecency_mod.prune_nested(frecency, M.MAX_SOURCES, now)

  latency.start(latency_ctx, "transitions.save_tracking", "async.transition_record")
  tracking.transition_frecency = frecency
  db.save_tracking("files", tracking, latency_ctx)
  latency.finish(latency_ctx, "transitions.save_tracking")
end

--- Compute transition scores from a source file to all destinations.
--- Returns normalized scores in [0,1] via 1 - 1/(1+score/4).
---@param source_path string Source file path
---@return table<string, number> Map of destination paths to normalized scores
function M.compute_scores_from(source_path)
  local normalized_source = path_mod.normalize(source_path)

  local db = require("neural-open.db")
  local tracking = db.get_tracking("files") or {}

  local frecency = tracking.transition_frecency or {}
  local destinations = frecency[normalized_source]

  if not destinations then
    return {}
  end

  local now = os.time()
  local scores = {}
  for dest, deadline in pairs(destinations) do
    local score = frecency_mod.deadline_to_score(deadline, now)
    scores[dest] = frecency_mod.normalize_transition(score, 4)
  end

  return scores
end

return M
