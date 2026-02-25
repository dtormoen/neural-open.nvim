--- File transition tracking with frecency-based scoring
local M = {}

local HALF_LIFE = 30 * 24 * 3600 -- 30 days in seconds
local LAMBDA = math.log(2) / HALF_LIFE

-- Exposed for testing
M.MAX_DESTINATIONS_PER_SOURCE = 50
M.MAX_SOURCES = 200

---@param path string
---@return string
local function normalize_path(path)
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

--- Convert a deadline timestamp to a current score: s = e^(λ*(deadline-now))
---@param deadline number
---@param now number
---@return number
local function deadline_to_score(deadline, now)
  return math.exp(LAMBDA * (deadline - now))
end

--- Convert a score to a deadline timestamp: t = now + ln(s)/λ
---@param score number
---@param now number
---@return number
local function score_to_deadline(score, now)
  return now + math.log(score) / LAMBDA
end

--- Prune a destinations table in-place, keeping only top entries by score.
---@param destinations table<string, number> Map of dest -> deadline
---@param max_count number
---@param now number
local function prune_destinations(destinations, max_count, now)
  local entries = {}
  for dest, deadline in pairs(destinations) do
    entries[#entries + 1] = { dest = dest, score = deadline_to_score(deadline, now) }
  end

  if #entries <= max_count then
    return
  end

  table.sort(entries, function(a, b)
    return a.score > b.score
  end)

  for i = max_count + 1, #entries do
    destinations[entries[i].dest] = nil
  end
end

--- Prune the sources table in-place, keeping only top entries by total score.
---@param frecency table<string, table<string, number>>
---@param max_count number
---@param now number
local function prune_sources(frecency, max_count, now)
  local sources = {}
  for source, destinations in pairs(frecency) do
    local total_score = 0
    for _, deadline in pairs(destinations) do
      total_score = total_score + deadline_to_score(deadline, now)
    end
    sources[#sources + 1] = { source = source, score = total_score }
  end

  if #sources <= max_count then
    return
  end

  table.sort(sources, function(a, b)
    return a.score > b.score
  end)

  for i = max_count + 1, #sources do
    frecency[sources[i].source] = nil
  end
end

--- Remove legacy transition_history if present.
---@param all_weights table
---@return boolean removed
local function migrate_legacy(all_weights)
  if all_weights.transition_history then
    all_weights.transition_history = nil
    return true
  end
  return false
end

--- Record a transition from source file to destination file.
--- Each visit adds 1 to the frecency score, stored as a deadline for automatic decay.
---@param from_path string Source file path
---@param to_path string Destination file path
---@param latency_ctx? table Optional latency context
function M.record_transition(from_path, to_path, latency_ctx)
  local latency = require("neural-open.latency")

  local result, ok = latency.measure(latency_ctx, "transitions.normalize_paths", function()
    return { normalize_path(from_path), normalize_path(to_path) }
  end, "async.transition_record")

  local normalized_from, normalized_to
  if ok then
    normalized_from, normalized_to = result[1], result[2]
  else
    normalized_from = normalize_path(from_path)
    normalized_to = normalize_path(to_path)
  end

  latency.start(latency_ctx, "transitions.get_weights", "async.transition_record")
  local db = require("neural-open.db")
  local all_weights = db.get_weights(latency_ctx) or {}
  latency.finish(latency_ctx, "transitions.get_weights")

  migrate_legacy(all_weights)

  local now = os.time()
  local frecency = all_weights.transition_frecency or {}
  local destinations = frecency[normalized_from] or {}

  local current_score = 0
  if destinations[normalized_to] then
    current_score = deadline_to_score(destinations[normalized_to], now)
  end

  destinations[normalized_to] = score_to_deadline(current_score + 1, now)

  prune_destinations(destinations, M.MAX_DESTINATIONS_PER_SOURCE, now)
  frecency[normalized_from] = destinations
  prune_sources(frecency, M.MAX_SOURCES, now)

  latency.start(latency_ctx, "transitions.save_weights", "async.transition_record")
  all_weights.transition_frecency = frecency
  db.save_weights(all_weights, latency_ctx)
  latency.finish(latency_ctx, "transitions.save_weights")
end

--- Compute transition scores from a source file to all destinations.
--- Returns normalized scores in [0,1] via 1 - 1/(1+score/4).
---@param source_path string Source file path
---@return table<string, number> Map of destination paths to normalized scores
function M.compute_scores_from(source_path)
  local normalized_source = normalize_path(source_path)

  local db = require("neural-open.db")
  local all_weights = db.get_weights() or {}

  if migrate_legacy(all_weights) then
    db.save_weights(all_weights)
  end

  local frecency = all_weights.transition_frecency or {}
  local destinations = frecency[normalized_source]

  if not destinations then
    return {}
  end

  local now = os.time()
  local scores = {}
  for dest, deadline in pairs(destinations) do
    local score = deadline_to_score(deadline, now)
    scores[dest] = 1 - 1 / (1 + score / 4)
  end

  return scores
end

return M
