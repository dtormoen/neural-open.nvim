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
---@return string
local function picker_path(picker_name)
  local dir = ensure_weights_dir()
  return dir .. "/" .. picker_name .. ".json"
end

--- Save weights to disk with optional latency tracking
---@param picker_name string The picker name (e.g. "files")
---@param weights table
---@param latency_ctx? table Optional latency context for tracking
---@return boolean success
function M.save_weights(picker_name, weights, latency_ctx)
  local latency = require("neural-open.latency")
  local weights_path = picker_path(picker_name)

  -- Encode with timing and metadata
  local encoded, ok = latency.measure(latency_ctx, "db.save.json_encode", function()
    return vim.json.encode(weights)
  end, "db.save_weights")

  if not ok then
    vim.notify("neural-open: Failed to encode weights: " .. tostring(encoded), vim.log.levels.ERROR)
    return false
  end

  -- Add metadata about size
  latency.add_metadata(latency_ctx, "db.save.json_encode", {
    bytes = #encoded,
    keys = vim.tbl_count(weights),
  })

  -- Generate unique temp file name
  local temp_path = weights_path .. ".tmp." .. vim.fn.getpid() .. "." .. vim.loop.hrtime()

  -- Write to temp file first with timing
  local write_result, write_ok = latency.measure(latency_ctx, "db.save.file_write", function()
    local file = io.open(temp_path, "w")
    if not file then
      error("Failed to open temp weights file for writing")
    end

    -- Write and flush to ensure data is on disk
    file:write(encoded)
    file:flush()
    file:close()
    return true
  end, "db.save_weights")

  if not write_ok then
    pcall(os.remove, temp_path) -- Clean up temp file
    vim.notify("neural-open: Failed to write weights: " .. tostring(write_result), vim.log.levels.ERROR)
    return false
  end

  -- Atomic rename operation with timing
  local rename_result, rename_ok = latency.measure(latency_ctx, "db.save.atomic_rename", function()
    local ok_rename = os.rename(temp_path, weights_path)
    if not ok_rename then
      -- Fallback for cross-filesystem moves
      vim.fn.writefile({ encoded }, weights_path)
    end
    return true
  end, "db.save_weights")

  pcall(os.remove, temp_path) -- Clean up temp file if it still exists

  if not rename_ok then
    vim.notify("neural-open: Failed to move weights file: " .. tostring(rename_result), vim.log.levels.ERROR)
    return false
  end

  return true
end

--- Get weights from disk with optional latency tracking
---@param picker_name string The picker name (e.g. "files")
---@param latency_ctx? table Optional latency context
---@return table weights
function M.get_weights(picker_name, latency_ctx)
  local latency = require("neural-open.latency")
  local weights_path = picker_path(picker_name)

  -- Read file with timing
  local content, read_ok = latency.measure(latency_ctx, "db.get.file_read", function()
    local file = io.open(weights_path, "r")
    if not file then
      return ""
    end
    local c = file:read("*all")
    file:close()
    return c
  end, "db.get_weights")

  if not read_ok or content == "" then
    return {}
  end

  latency.add_metadata(latency_ctx, "db.get.file_read", { bytes = #content })

  -- Decode with timing
  local weights, decode_ok = latency.measure(latency_ctx, "db.get.json_decode", function()
    return vim.json.decode(content)
  end, "db.get_weights")

  if not decode_ok then
    vim.notify("neural-open: Failed to decode weights file: " .. tostring(weights), vim.log.levels.ERROR)
    return {}
  end

  return weights or {}
end

--- Reset the cached weights directory path. Used by tests to ensure clean state.
function M.reset_cache()
  cached_weights_dir = nil
end

return M
