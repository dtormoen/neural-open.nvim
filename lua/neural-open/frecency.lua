--- Shared deadline-based frecency math.
--- Pure functions: no I/O, no mutable state, no external requires.
local M = {}

--- Half-life for exponential decay (30 days in seconds).
M.HALF_LIFE = 30 * 24 * 3600

--- Decay constant: ln(2) / half_life.
M.LAMBDA = math.log(2) / M.HALF_LIFE

--- Convert a deadline timestamp to a current score: s = e^(lambda*(deadline-now))
---@param deadline number Deadline timestamp
---@param now number Current timestamp
---@return number score Current frecency score
function M.deadline_to_score(deadline, now)
  return math.exp(M.LAMBDA * (deadline - now))
end

--- Convert a score to a deadline timestamp: t = now + ln(s)/lambda
---@param score number Frecency score (must be > 0)
---@param now number Current timestamp
---@return number deadline Deadline timestamp
function M.score_to_deadline(score, now)
  return now + math.log(score) / M.LAMBDA
end

--- Bump a frecency entry: compute current score from deadline (0 if nil), add 1, return new deadline.
---@param deadline_or_nil number? Existing deadline timestamp, or nil for new entry
---@param now number Current timestamp
---@return number deadline New deadline timestamp
function M.bump(deadline_or_nil, now)
  local current_score = 0
  if deadline_or_nil then
    current_score = M.deadline_to_score(deadline_or_nil, now)
  end
  return M.score_to_deadline(current_score + 1, now)
end

--- Normalize a raw score to [0,1] via 1 - 1/(1 + raw/divisor).
---@param raw_score number Raw frecency score
---@param divisor number Divisor controlling the curve shape
---@return number normalized Score in [0,1]
function M.normalize_transition(raw_score, divisor)
  return 1 - 1 / (1 + raw_score / divisor)
end

--- Prune a flat deadline map in-place, keeping only the top entries by score.
---@param map table<string, number> Map of key -> deadline
---@param max_count number Maximum entries to keep
---@param now number Current timestamp
function M.prune_map(map, max_count, now)
  local entries = {}
  for key, deadline in pairs(map) do
    entries[#entries + 1] = { key = key, score = M.deadline_to_score(deadline, now) }
  end

  if #entries <= max_count then
    return
  end

  table.sort(entries, function(a, b)
    return a.score > b.score
  end)

  for i = max_count + 1, #entries do
    map[entries[i].key] = nil
  end
end

--- Prune a source-to-destinations nested map in-place, keeping only the top sources by total score.
---@param nested_map table<string, table<string, number>> Map of source -> { dest -> deadline }
---@param max_count number Maximum sources to keep
---@param now number Current timestamp
function M.prune_nested(nested_map, max_count, now)
  local sources = {}
  for source, destinations in pairs(nested_map) do
    local total_score = 0
    for _, deadline in pairs(destinations) do
      total_score = total_score + M.deadline_to_score(deadline, now)
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
    nested_map[sources[i].source] = nil
  end
end

return M
