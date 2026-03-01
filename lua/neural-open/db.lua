local M = {}

local cached_weights_dir = nil

--- Resolve and cache the weights directory path.
--- Handles backward compatibility for .json file paths and auto-migration from weights.json to files.json.
---@return string The weights directory path
local function ensure_weights_dir()
  if cached_weights_dir then
    return cached_weights_dir
  end

  local init = require("neural-open")
  local configured_path = init.config.weights_path or (vim.fn.stdpath("data") .. "/neural-open/")
  configured_path = vim.fn.expand(configured_path)

  -- Backward compat: if path ends in .json or is an existing file, use parent dir
  if configured_path:match("%.json$") or (vim.fn.filereadable(configured_path) == 1) then
    vim.notify(
      "neural-open: weights_path should be a directory, not a file. Using parent directory.",
      vim.log.levels.WARN
    )
    cached_weights_dir = vim.fn.fnamemodify(configured_path, ":h")
  else
    -- Strip trailing slashes for consistency
    cached_weights_dir = configured_path:gsub("/+$", "")
  end

  vim.fn.mkdir(cached_weights_dir, "p")

  -- Auto-migration: weights.json -> files.json
  local old_path = cached_weights_dir .. "/weights.json"
  local new_path = cached_weights_dir .. "/files.json"
  if vim.fn.filereadable(old_path) == 1 and vim.fn.filereadable(new_path) == 0 then
    -- Create backup before rename so original is preserved if anything fails
    local f = io.open(old_path, "r")
    if f then
      local content = f:read("*all")
      f:close()
      local bak = io.open(old_path .. ".bak", "w")
      if bak then
        bak:write(content)
        bak:close()
      end
    end
    os.rename(old_path, new_path)
  end

  return cached_weights_dir
end

--- Resolve the file path for a specific picker name.
---@param picker_name string
---@param suffix? string Optional suffix (e.g. ".tracking") inserted before ".json"
---@return string
local function picker_path(picker_name, suffix)
  local dir = ensure_weights_dir()
  return dir .. "/" .. picker_name .. (suffix or "") .. ".json"
end

--- Read and decode a JSON file from disk with optional latency tracking.
---@param picker_name string The picker name (e.g. "files")
---@param suffix string File suffix ("" for weights, ".tracking" for tracking)
---@param latency_ctx? table Optional latency context
---@return table data
local function read_json(picker_name, suffix, latency_ctx)
  local latency = require("neural-open.latency")
  local path = picker_path(picker_name, suffix)
  local parent = suffix == "" and "db.get_weights" or "db.get_tracking"

  local content, read_ok = latency.measure(latency_ctx, "db.get.file_read", function()
    local file = io.open(path, "r")
    if not file then
      return ""
    end
    local c = file:read("*all")
    file:close()
    return c
  end, parent)

  if not read_ok or content == "" then
    return {}
  end

  latency.add_metadata(latency_ctx, "db.get.file_read", { bytes = #content })

  local data, decode_ok = latency.measure(latency_ctx, "db.get.json_decode", function()
    return vim.json.decode(content)
  end, parent)

  if not decode_ok then
    vim.notify("neural-open: Failed to decode JSON file: " .. tostring(data), vim.log.levels.ERROR)
    return {}
  end

  return data or {}
end

--- Encode and write a table as JSON to disk with optional latency tracking.
---@param picker_name string The picker name (e.g. "files")
---@param suffix string File suffix ("" for weights, ".tracking" for tracking)
---@param data table
---@param latency_ctx? table Optional latency context
---@return boolean success
local function write_json(picker_name, suffix, data, latency_ctx)
  local latency = require("neural-open.latency")
  local path = picker_path(picker_name, suffix)
  local parent = suffix == "" and "db.save_weights" or "db.save_tracking"

  local encoded, ok = latency.measure(latency_ctx, "db.save.json_encode", function()
    return vim.json.encode(data)
  end, parent)

  if not ok then
    vim.notify("neural-open: Failed to encode data: " .. tostring(encoded), vim.log.levels.ERROR)
    return false
  end

  latency.add_metadata(latency_ctx, "db.save.json_encode", {
    bytes = #encoded,
    keys = vim.tbl_count(data),
  })

  local temp_path = path .. ".tmp." .. vim.fn.getpid() .. "." .. vim.loop.hrtime()

  local write_result, write_ok = latency.measure(latency_ctx, "db.save.file_write", function()
    local file = io.open(temp_path, "w")
    if not file then
      error("Failed to open temp file for writing")
    end

    file:write(encoded)
    file:flush()
    file:close()
    return true
  end, parent)

  if not write_ok then
    pcall(os.remove, temp_path)
    vim.notify("neural-open: Failed to write data: " .. tostring(write_result), vim.log.levels.ERROR)
    return false
  end

  local rename_result, rename_ok = latency.measure(latency_ctx, "db.save.atomic_rename", function()
    local ok_rename = os.rename(temp_path, path)
    if not ok_rename then
      vim.fn.writefile({ encoded }, path)
    end
    return true
  end, parent)

  pcall(os.remove, temp_path)

  if not rename_ok then
    vim.notify("neural-open: Failed to move file: " .. tostring(rename_result), vim.log.levels.ERROR)
    return false
  end

  return true
end

--- Migrate tracking keys from weight file to tracking file on first access.
---@param picker_name string
---@param latency_ctx? table
---@return table tracking
local function migrate_tracking(picker_name, latency_ctx)
  local weights = read_json(picker_name, "", latency_ctx)
  if vim.tbl_isempty(weights) then
    return {}
  end

  local tracking_keys = { "recency_list", "transition_frecency", "item_tracking" }
  local tracking = {}
  local found = false

  for _, key in ipairs(tracking_keys) do
    if weights[key] ~= nil then
      tracking[key] = weights[key]
      weights[key] = nil
      found = true
    end
  end

  -- Also handle legacy transition_history key
  if weights.transition_history then
    weights.transition_history = nil
    found = true
  end

  if not found then
    return {}
  end

  -- Write tracking file first (crash safety: data in both is harmless)
  write_json(picker_name, ".tracking", tracking, latency_ctx)
  -- Re-save weights without tracking keys
  write_json(picker_name, "", weights, latency_ctx)

  return tracking
end

--- Save weights to disk with optional latency tracking
---@param picker_name string The picker name (e.g. "files")
---@param weights table
---@param latency_ctx? table Optional latency context for tracking
---@return boolean success
function M.save_weights(picker_name, weights, latency_ctx)
  return write_json(picker_name, "", weights, latency_ctx)
end

--- Get weights from disk with optional latency tracking
---@param picker_name string The picker name (e.g. "files")
---@param latency_ctx? table Optional latency context
---@return table weights
function M.get_weights(picker_name, latency_ctx)
  return read_json(picker_name, "", latency_ctx)
end

--- Save tracking data to disk with optional latency tracking
---@param picker_name string The picker name (e.g. "files")
---@param data table
---@param latency_ctx? table Optional latency context
---@return boolean success
function M.save_tracking(picker_name, data, latency_ctx)
  return write_json(picker_name, ".tracking", data, latency_ctx)
end

--- Get tracking data from disk with optional latency tracking.
--- On first access, migrates tracking keys from the weight file if needed.
---@param picker_name string The picker name (e.g. "files")
---@param latency_ctx? table Optional latency context
---@return table tracking
function M.get_tracking(picker_name, latency_ctx)
  local tracking = read_json(picker_name, ".tracking", latency_ctx)
  if vim.tbl_isempty(tracking) then
    tracking = migrate_tracking(picker_name, latency_ctx)
  end
  return tracking
end

--- Reset the cached weights directory path. Used by tests to ensure clean state.
function M.reset_cache()
  cached_weights_dir = nil
end

return M
