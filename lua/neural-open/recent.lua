--- Recency tracking module with in-memory cache and debounced persistence
--- Maintains an ordered list of recently accessed files, persisted to the weights file.
--- BufEnter events update the in-memory list; disk writes are debounced to avoid
--- excessive I/O since the weights file may contain large NN state.
local M = {}

local path_mod = require("neural-open.path")

--- In-memory ordered array of normalized paths (index 1 = most recent)
---@type string[]
local recency_list = {}

--- Whether the recency list has been loaded from disk
---@type boolean
local loaded = false

--- Debounce timer handle for deferred persistence
---@type uv.uv_timer_t?
local save_timer = nil

--- Whether the in-memory list has unsaved changes
---@type boolean
local dirty = false

--- Get the maximum recency list size from config
---@return number
local function get_max_size()
  local config = require("neural-open").config
  return config.recency_list_size or 100
end

--- Seed the recency list from vim.v.oldfiles and buffer lastused timestamps.
--- Used as a fallback when no persisted recency list exists yet.
---@return string[] Ordered array of normalized file paths (most recent first)
local function seed_from_vim_sources()
  local limit = get_max_size()
  local result = {}
  local added = {}

  -- Collect listed buffers with lastused timestamps
  local buffers = {}
  for _, buf in ipairs(vim.api.nvim_list_bufs()) do
    if vim.api.nvim_buf_is_valid(buf) and vim.bo[buf].buflisted and vim.bo[buf].buftype == "" then
      local buf_name = vim.api.nvim_buf_get_name(buf)
      if buf_name and buf_name ~= "" then
        local buf_info = vim.fn.getbufinfo(buf)[1]
        if buf_info and buf_info.lastused then
          table.insert(buffers, {
            path = path_mod.normalize(buf_name),
            lastused = buf_info.lastused,
          })
        end
      end
    end
  end

  -- Sort buffers by lastused (most recent first)
  table.sort(buffers, function(a, b)
    return a.lastused > b.lastused
  end)

  -- Add recently used buffers first
  for _, buf in ipairs(buffers) do
    if #result >= limit then
      break
    end
    if not added[buf.path] then
      table.insert(result, buf.path)
      added[buf.path] = true
    end
  end

  -- Then add files from oldfiles that are not already included
  local oldfiles = vim.v.oldfiles or {}
  for _, file in ipairs(oldfiles) do
    if #result >= limit then
      break
    end
    local abs_path = path_mod.normalize(file)
    if not added[abs_path] and vim.fn.filereadable(file) == 1 then
      table.insert(result, abs_path)
      added[abs_path] = true
    end
  end

  return result
end

--- Load the recency list from disk if not already loaded.
--- Falls back to seeding from vim sources when no persisted list exists.
local function ensure_loaded()
  if loaded then
    return
  end
  loaded = true

  local db = require("neural-open.db")
  local all_weights = db.get_weights() or {}
  local persisted = all_weights.recency_list

  if persisted and type(persisted) == "table" and #persisted > 0 then
    recency_list = persisted
  else
    recency_list = seed_from_vim_sources()
    -- Do NOT set dirty; fallback data should not be persisted immediately.
    -- Real BufEnter events will push entries to the top, and the debounced
    -- save (or VimLeavePre flush) will eventually write a blended list.
  end
end

--- Record that a buffer was focused, moving it to the top of the recency list.
--- Schedules a debounced flush to persist the change after 5 seconds of inactivity.
---@param path string File path of the focused buffer
function M.record_buffer_focus(path)
  ensure_loaded()

  local normalized = path_mod.normalize(path)

  -- Remove existing entry if present
  for i = #recency_list, 1, -1 do
    if recency_list[i] == normalized then
      table.remove(recency_list, i)
      break
    end
  end

  -- Insert at the front (most recent)
  table.insert(recency_list, 1, normalized)

  -- Trim to max size
  local max_size = get_max_size()
  while #recency_list > max_size do
    recency_list[#recency_list] = nil
  end

  dirty = true

  -- Schedule debounced save (reuse timer to avoid handle leaks)
  if save_timer then
    save_timer:stop()
  else
    save_timer = vim.loop.new_timer()
  end
  if not save_timer then
    return
  end
  save_timer:start(
    5000,
    0,
    vim.schedule_wrap(function()
      M.flush()
    end)
  )
end

--- Build a recency map from the in-memory list for use by the scoring pipeline.
--- Returns a table mapping normalized paths to their recency metadata.
---@param limit? number Maximum number of entries to include (defaults to recency_list_size)
---@return table<string, {recent_rank: number}> Map of path to recency info
function M.get_recency_map(limit)
  ensure_loaded()

  limit = limit or get_max_size()
  local count = math.min(limit, #recency_list)
  local map = {}

  for i = 1, count do
    map[recency_list[i]] = { recent_rank = i }
  end

  return map
end

--- Return the raw ordered recency list array.
--- Index 1 is the most recently accessed file.
---@return string[] Ordered array of normalized file paths
function M.get_recency_list()
  ensure_loaded()
  return recency_list
end

--- Immediately persist the in-memory recency list to the weights file.
--- Cancels any pending debounce timer. No-op if there are no unsaved changes.
function M.flush()
  if not dirty or not loaded then
    return
  end

  -- Cancel any pending debounce timer
  if save_timer then
    save_timer:stop()
  end

  local db = require("neural-open.db")
  local all_weights = db.get_weights() or {}
  all_weights.recency_list = recency_list
  db.save_weights(all_weights)

  dirty = false
end

return M
