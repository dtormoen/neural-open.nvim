--- Recency tracking module with pending touches and debounced persistence.
--- Maintains a pending list of recently accessed files in memory. On flush or
--- get_recency_map, merges pending touches with the on-disk recency list.
--- BufEnter events update only the in-memory pending list; disk writes are
--- debounced to avoid excessive I/O.
local M = {}

local path_mod = require("neural-open.path")

--- Pending touches: ordered list of normalized paths (index 1 = most recent).
--- Only contains files touched since last flush; merged with disk on read/flush.
---@type string[]
local pending_touches = {}

--- Dedup set for pending_touches: path -> true
---@type table<string, boolean>
local pending_set = {}

--- Debounce timer handle for deferred persistence
---@type uv.uv_timer_t?
local save_timer = nil

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

--- Merge pending touches with the on-disk recency list.
--- Returns the merged list and the tracking table without modifying pending state.
---@param limit? number Maximum entries in result (defaults to recency_list_size)
---@return string[] merged Merged ordered list (most recent first)
---@return table tracking The tracking table read from disk (reusable by caller)
local function merge_with_disk(limit)
  limit = limit or get_max_size()

  -- Read on-disk recency list
  local db = require("neural-open.db")
  local tracking = db.get_tracking("files") or {}
  local disk_list = tracking.recency_list

  -- Seed from vim sources if disk is empty
  if not disk_list or type(disk_list) ~= "table" or #disk_list == 0 then
    disk_list = seed_from_vim_sources()
  end

  -- Build merged list: pending_touches first, then disk entries skipping dupes
  local merged = {}
  local seen = {}

  -- Add all pending touches (most recent first)
  for _, path in ipairs(pending_touches) do
    if #merged >= limit then
      break
    end
    if not seen[path] then
      table.insert(merged, path)
      seen[path] = true
    end
  end

  -- Append disk entries, skipping those already in pending_set
  for _, path in ipairs(disk_list) do
    if #merged >= limit then
      break
    end
    if not seen[path] then
      table.insert(merged, path)
      seen[path] = true
    end
  end

  return merged, tracking
end

--- Record that a buffer was focused, adding it to the pending touches list.
--- Schedules a debounced flush to persist the change after 5 seconds of inactivity.
--- No disk I/O is performed by this function.
---@param path string File path of the focused buffer
function M.record_buffer_focus(path)
  local normalized = path_mod.normalize(path)

  -- If already in pending_set, remove from current position
  if pending_set[normalized] then
    for i = #pending_touches, 1, -1 do
      if pending_touches[i] == normalized then
        table.remove(pending_touches, i)
        break
      end
    end
  end

  -- Insert at the front (most recent)
  table.insert(pending_touches, 1, normalized)
  pending_set[normalized] = true

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

--- Build a recency map by merging pending touches with the on-disk list.
--- Returns a table mapping normalized paths to their recency metadata.
--- Read-only: does NOT clear pending_touches.
---@param limit? number Maximum number of entries to include (defaults to recency_list_size)
---@return table<string, {recent_rank: number}> Map of path to recency info
function M.get_recency_map(limit)
  local merged = merge_with_disk(limit)
  local map = {}

  for i, path in ipairs(merged) do
    map[path] = { recent_rank = i }
  end

  return map
end

--- Immediately persist pending touches by merging with disk and writing back.
--- Cancels any pending debounce timer. No-op if there are no pending touches.
function M.flush()
  if #pending_touches == 0 then
    return
  end

  -- Cancel any pending debounce timer
  if save_timer then
    save_timer:stop()
  end

  local merged, tracking = merge_with_disk()
  tracking.recency_list = merged

  local db = require("neural-open.db")
  db.save_tracking("files", tracking)

  -- Clear pending state
  pending_touches = {}
  pending_set = {}
end

return M
