--- Item tracking module for non-file pickers.
--- Provides frecency and recency tracking for arbitrary items, keyed by picker name + item identity.
--- Supports both global and CWD-scoped tracking. Data persisted under "item_tracking" key
--- in each picker's tracking JSON file via the db module.
---
--- Stateless module: every operation reads from disk, modifies in-place, and writes back
--- immediately. No module-level mutable state.
local M = {}

local frecency_mod = require("neural-open.frecency")

-- Pruning limits
M.MAX_FRECENCY_ITEMS = 500
M.MAX_CWD_FRECENCY_ITEMS = 200
M.MAX_TRANSITION_DESTINATIONS = 50
M.MAX_TRANSITION_SOURCES = 200

--- Get the maximum recency list size from config.
---@return number
local function get_max_size()
  local config = require("neural-open").config
  return config.recency_list_size or 100
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

--- Read the item_tracking store from disk for a picker.
---@param picker_name string
---@return NosItemTrackingStore
local function get_store(picker_name)
  local db = require("neural-open.db")
  local all = db.get_tracking(picker_name) or {}
  return all.item_tracking or {}
end

--- Write the item_tracking store back to disk for a picker.
---@param picker_name string
---@param store NosItemTrackingStore
local function save_store(picker_name, store)
  local db = require("neural-open.db")
  local tracking = db.get_tracking(picker_name) or {}
  tracking.item_tracking = store
  db.save_tracking(picker_name, tracking)
end

--- Record that an item was selected. Reads store from disk, applies all five
--- tracking updates, and writes back immediately.
---@param picker_name string
---@param item_id string Item identity (caller resolves item.value or item.text)
---@param cwd string Current working directory
function M.record_selection(picker_name, item_id, cwd)
  local store = get_store(picker_name)
  local now = os.time()
  local max_size = get_max_size()

  -- Ensure store tables exist
  store.frecency = store.frecency or {}
  store.cwd_frecency = store.cwd_frecency or {}
  store.recency_list = store.recency_list or {}
  store.cwd_recency = store.cwd_recency or {}
  store.transition_frecency = store.transition_frecency or {}

  -- Capture "from" item before recency lists are updated
  local from_item = store.cwd_recency[cwd] and store.cwd_recency[cwd][1]

  -- 1. Global frecency: add 1 to score
  store.frecency[item_id] = frecency_mod.bump(store.frecency[item_id], now)
  frecency_mod.prune_map(store.frecency, M.MAX_FRECENCY_ITEMS, now)

  -- 2. CWD frecency: add 1 to score
  local cwd_map = store.cwd_frecency[cwd] or {}
  cwd_map[item_id] = frecency_mod.bump(cwd_map[item_id], now)
  frecency_mod.prune_map(cwd_map, M.MAX_CWD_FRECENCY_ITEMS, now)
  store.cwd_frecency[cwd] = cwd_map

  -- 3. Global recency
  push_recency(store.recency_list, item_id, max_size)

  -- 4. CWD recency
  local cwd_recency = store.cwd_recency[cwd] or {}
  push_recency(cwd_recency, item_id, max_size)
  store.cwd_recency[cwd] = cwd_recency

  -- 5. Transition frecency: record from_item -> item_id
  if from_item then
    local destinations = store.transition_frecency[from_item] or {}
    destinations[item_id] = frecency_mod.bump(destinations[item_id], now)
    frecency_mod.prune_map(destinations, M.MAX_TRANSITION_DESTINATIONS, now)
    store.transition_frecency[from_item] = destinations
    frecency_mod.prune_nested(store.transition_frecency, M.MAX_TRANSITION_SOURCES, now)
  end

  -- Write back immediately
  save_store(picker_name, store)
end

--- Get computed tracking data for feature computation.
--- Reads from disk unless a pre-loaded store is provided.
---@param picker_name string
---@param cwd string Current working directory
---@param store? NosItemTrackingStore Optional pre-loaded store to avoid double-read
---@return NosItemTrackingData
function M.get_tracking_data(picker_name, cwd, store)
  if not store then
    store = get_store(picker_name)
  end
  local now = os.time()

  local store_frecency = store.frecency or {}
  local store_cwd_frecency = store.cwd_frecency or {}
  local store_recency_list = store.recency_list or {}
  local store_cwd_recency = store.cwd_recency or {}

  -- Compute global frecency scores
  local frecency = {}
  for id, deadline in pairs(store_frecency) do
    frecency[id] = frecency_mod.deadline_to_score(deadline, now)
  end

  -- Compute CWD frecency scores
  local cwd_frecency = {}
  local cwd_map = store_cwd_frecency[cwd]
  if cwd_map then
    for id, deadline in pairs(cwd_map) do
      cwd_frecency[id] = frecency_mod.deadline_to_score(deadline, now)
    end
  end

  -- Compute global recency ranks (1-based)
  local recency_rank = {}
  for i, id in ipairs(store_recency_list) do
    recency_rank[id] = i
  end

  -- Compute CWD recency ranks (1-based)
  local cwd_recency_rank = {}
  local cwd_recency = store_cwd_recency[cwd]
  if cwd_recency then
    for i, id in ipairs(cwd_recency) do
      cwd_recency_rank[id] = i
    end
  end

  -- Last selected is position 1 of global recency list
  local last_selected = store_recency_list[1]

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
--- Reads from disk unless a pre-loaded store is provided.
---@param picker_name string
---@param source_item_id string Source item identity
---@param store? NosItemTrackingStore Optional pre-loaded store to avoid double-read
---@return table<string, number> Map of destination item ids to normalized scores
function M.compute_transition_scores(picker_name, source_item_id, store)
  if not store then
    store = get_store(picker_name)
  end

  local transition_frecency = store.transition_frecency or {}
  local destinations = transition_frecency[source_item_id]

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

--- Get the raw transition frecency table for a picker (for debug views).
--- Reads from disk unless a pre-loaded store is provided.
---@param picker_name string
---@param store? NosItemTrackingStore Optional pre-loaded store to avoid double-read
---@return table<string, table<string, number>> Raw transition_frecency table (source -> {dest -> deadline})
function M.get_transition_frecency(picker_name, store)
  if not store then
    store = get_store(picker_name)
  end
  return store.transition_frecency or {}
end

--- Get the set of item_ids this picker has ever tracked globally.
--- Unions ids from global frecency and recency_list. Intended for finders that
--- want to re-emit previously-selected items (e.g. recipe invocations with
--- custom args) so they participate in scoring.
---@param picker_name string
---@return table<string, true> Set of known item_ids
function M.get_known_item_ids(picker_name)
  local store = get_store(picker_name)
  local seen = {}
  for id in pairs(store.frecency or {}) do
    seen[id] = true
  end
  for _, id in ipairs(store.recency_list or {}) do
    seen[id] = true
  end
  return seen
end

--- Get the set of item_ids this picker has tracked within a specific cwd.
--- Unions ids from cwd_frecency[cwd] and cwd_recency[cwd]. Use this when you
--- only want to re-emit items that were selected while working in this cwd.
---@param picker_name string
---@param cwd string Current working directory
---@return table<string, true> Set of known item_ids for this cwd
function M.get_cwd_known_item_ids(picker_name, cwd)
  local store = get_store(picker_name)
  local seen = {}
  local cwd_frec = store.cwd_frecency and store.cwd_frecency[cwd]
  if cwd_frec then
    for id in pairs(cwd_frec) do
      seen[id] = true
    end
  end
  local cwd_rec = store.cwd_recency and store.cwd_recency[cwd]
  if cwd_rec then
    for _, id in ipairs(cwd_rec) do
      seen[id] = true
    end
  end
  return seen
end

--- No-op reset (kept for test compatibility). No module-level state to clear.
function M.reset() end

return M
