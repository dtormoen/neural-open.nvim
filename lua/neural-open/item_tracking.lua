--- Item tracking module for non-file pickers.
--- Provides frecency and recency tracking for arbitrary items, keyed by picker name + item identity.
--- Supports both global and CWD-scoped tracking. Data persisted under "item_tracking" key
--- in each picker's JSON file via the db module.
local M = {}

local HALF_LIFE = 30 * 24 * 3600 -- 30 days in seconds
local LAMBDA = math.log(2) / HALF_LIFE
local DEBOUNCE_MS = 5000

-- Pruning limits
M.MAX_FRECENCY_ITEMS = 500
M.MAX_CWD_FRECENCY_ITEMS = 200
M.MAX_TRANSITION_DESTINATIONS = 50
M.MAX_TRANSITION_SOURCES = 200

--- In-memory cache keyed by picker_name.
--- Each entry holds the tracking stores, loaded/dirty flags, and save timer.
---@type table<string, { frecency: table<string, number>, cwd_frecency: table<string, table<string, number>>, recency_list: string[], cwd_recency: table<string, string[]>, transition_frecency: table<string, table<string, number>>, loaded: boolean, dirty: boolean, save_timer: uv.uv_timer_t? }>
local cache = {}

--- Get the maximum recency list size from config.
---@return number
local function get_max_size()
  local config = require("neural-open").config
  return config.recency_list_size or 100
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

--- Prune a frecency table in-place, keeping only top entries by score.
---@param frecency_map table<string, number> Map of item_id -> deadline
---@param max_count number
---@param now number
local function prune_frecency(frecency_map, max_count, now)
  local entries = {}
  for id, deadline in pairs(frecency_map) do
    entries[#entries + 1] = { id = id, score = deadline_to_score(deadline, now) }
  end

  if #entries <= max_count then
    return
  end

  table.sort(entries, function(a, b)
    return a.score > b.score
  end)

  for i = max_count + 1, #entries do
    frecency_map[entries[i].id] = nil
  end
end

--- Prune a destinations table in-place, keeping only top entries by score.
---@param destinations table<string, number> Map of dest_id -> deadline
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

--- Ensure tracking data is loaded for a picker.
---@param picker_name string
---@return table Cache entry for this picker
local function ensure_loaded(picker_name)
  local entry = cache[picker_name]
  if entry and entry.loaded then
    return entry
  end

  local db = require("neural-open.db")
  local all_weights = db.get_weights(picker_name) or {}
  local tracking = all_weights.item_tracking or {}

  entry = {
    frecency = tracking.frecency or {},
    cwd_frecency = tracking.cwd_frecency or {},
    recency_list = tracking.recency_list or {},
    cwd_recency = tracking.cwd_recency or {},
    transition_frecency = tracking.transition_frecency or {},
    loaded = true,
    dirty = false,
    save_timer = nil,
  }
  cache[picker_name] = entry
  return entry
end

--- Schedule a debounced save for a picker.
---@param picker_name string
local function schedule_save(picker_name)
  local entry = cache[picker_name]
  if not entry then
    return
  end

  if entry.save_timer then
    entry.save_timer:stop()
  else
    entry.save_timer = vim.loop.new_timer()
  end
  if not entry.save_timer then
    return
  end

  entry.save_timer:start(
    DEBOUNCE_MS,
    0,
    vim.schedule_wrap(function()
      M.flush(picker_name)
    end)
  )
end

--- Move an item to the front of a recency list, trimming to max size.
---@param list string[]
---@param item_id string
---@param max_size number
local function push_recency(list, item_id, max_size)
  -- Remove existing entry if present
  for i = #list, 1, -1 do
    if list[i] == item_id then
      table.remove(list, i)
      break
    end
  end

  -- Insert at front
  table.insert(list, 1, item_id)

  -- Trim to max size
  while #list > max_size do
    list[#list] = nil
  end
end

--- Initialize tracking data for a picker (loads from disk if not already loaded).
---@param picker_name string
function M.init(picker_name)
  ensure_loaded(picker_name)
end

--- Record that an item was selected. Updates all five tracking stores.
---@param picker_name string
---@param item_id string Item identity (caller resolves item.value or item.text)
---@param cwd string Current working directory
function M.record_selection(picker_name, item_id, cwd)
  local entry = ensure_loaded(picker_name)
  local now = os.time()
  local max_size = get_max_size()

  -- Capture "from" item before recency lists are updated
  local from_item = entry.cwd_recency[cwd] and entry.cwd_recency[cwd][1]

  -- 1. Global frecency: add 1 to score
  local current_score = 0
  if entry.frecency[item_id] then
    current_score = deadline_to_score(entry.frecency[item_id], now)
  end
  entry.frecency[item_id] = score_to_deadline(current_score + 1, now)
  prune_frecency(entry.frecency, M.MAX_FRECENCY_ITEMS, now)

  -- 2. CWD frecency: add 1 to score
  local cwd_map = entry.cwd_frecency[cwd] or {}
  local cwd_score = 0
  if cwd_map[item_id] then
    cwd_score = deadline_to_score(cwd_map[item_id], now)
  end
  cwd_map[item_id] = score_to_deadline(cwd_score + 1, now)
  prune_frecency(cwd_map, M.MAX_CWD_FRECENCY_ITEMS, now)
  entry.cwd_frecency[cwd] = cwd_map

  -- 3. Global recency
  push_recency(entry.recency_list, item_id, max_size)

  -- 4. CWD recency
  local cwd_recency = entry.cwd_recency[cwd] or {}
  push_recency(cwd_recency, item_id, max_size)
  entry.cwd_recency[cwd] = cwd_recency

  -- 5. Transition frecency: record from_item -> item_id
  if from_item then
    local destinations = entry.transition_frecency[from_item] or {}
    local current_transition_score = 0
    if destinations[item_id] then
      current_transition_score = deadline_to_score(destinations[item_id], now)
    end
    destinations[item_id] = score_to_deadline(current_transition_score + 1, now)
    prune_destinations(destinations, M.MAX_TRANSITION_DESTINATIONS, now)
    entry.transition_frecency[from_item] = destinations
    prune_sources(entry.transition_frecency, M.MAX_TRANSITION_SOURCES, now)
  end

  entry.dirty = true
  schedule_save(picker_name)
end

--- Get computed tracking data for feature computation.
---@param picker_name string
---@param cwd string Current working directory
---@return NosItemTrackingData
function M.get_tracking_data(picker_name, cwd)
  local entry = ensure_loaded(picker_name)
  local now = os.time()

  -- Compute global frecency scores
  local frecency = {}
  for id, deadline in pairs(entry.frecency) do
    frecency[id] = deadline_to_score(deadline, now)
  end

  -- Compute CWD frecency scores
  local cwd_frecency = {}
  local cwd_map = entry.cwd_frecency[cwd]
  if cwd_map then
    for id, deadline in pairs(cwd_map) do
      cwd_frecency[id] = deadline_to_score(deadline, now)
    end
  end

  -- Compute global recency ranks (1-based)
  local recency_rank = {}
  for i, id in ipairs(entry.recency_list) do
    recency_rank[id] = i
  end

  -- Compute CWD recency ranks (1-based)
  local cwd_recency_rank = {}
  local cwd_recency = entry.cwd_recency[cwd]
  if cwd_recency then
    for i, id in ipairs(cwd_recency) do
      cwd_recency_rank[id] = i
    end
  end

  -- Last selected is position 1 of global recency list
  local last_selected = entry.recency_list[1]

  -- Last CWD-scoped selected is position 1 of CWD recency list
  local last_cwd_selected = cwd_recency and cwd_recency[1] or nil

  return {
    frecency = frecency,
    cwd_frecency = cwd_frecency,
    recency_rank = recency_rank,
    cwd_recency_rank = cwd_recency_rank,
    last_selected = last_selected,
    last_cwd_selected = last_cwd_selected,
  }
end

--- Compute transition scores from a source item to all destinations.
--- Returns normalized scores in [0,1] via 1 - 1/(1+score/4).
---@param picker_name string
---@param source_item_id string Source item identity
---@return table<string, number> Map of destination item ids to normalized scores
function M.compute_transition_scores(picker_name, source_item_id)
  local entry = ensure_loaded(picker_name)
  local destinations = entry.transition_frecency[source_item_id]

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

--- Get the raw transition frecency table for a picker (for debug views).
---@param picker_name string
---@return table<string, table<string, number>> Raw transition_frecency table (source -> {dest -> deadline})
function M.get_transition_frecency(picker_name)
  local entry = ensure_loaded(picker_name)
  return entry.transition_frecency
end

--- Immediately persist tracking data for a picker. No-op if not dirty.
---@param picker_name string
function M.flush(picker_name)
  local entry = cache[picker_name]
  if not entry or not entry.dirty or not entry.loaded then
    return
  end

  -- Cancel pending timer
  if entry.save_timer then
    entry.save_timer:stop()
  end

  local db = require("neural-open.db")
  local all_weights = db.get_weights(picker_name) or {}
  all_weights.item_tracking = {
    frecency = entry.frecency,
    cwd_frecency = entry.cwd_frecency,
    recency_list = entry.recency_list,
    cwd_recency = entry.cwd_recency,
    transition_frecency = entry.transition_frecency,
  }
  db.save_weights(picker_name, all_weights)

  entry.dirty = false
end

--- Reset all in-memory caches. Used by tests to ensure clean state.
function M.reset()
  for _, entry in pairs(cache) do
    if entry.save_timer then
      entry.save_timer:stop()
    end
  end
  cache = {}
end

return M
