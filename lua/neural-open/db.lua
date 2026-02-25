local M = {}

local cached_weights_path = nil

local function ensure_weights_file()
  -- Return cached path if already initialized
  if cached_weights_path then
    return cached_weights_path
  end

  -- Check config for weights_path, fall back to default
  local init = require("neural-open")
  local configured_path = init.config.weights_path or vim.fn.stdpath("data") .. "/neural-open/weights.json"
  -- Expand ~ and environment variables in the path
  cached_weights_path = vim.fn.expand(configured_path)

  -- Ensure the directory exists
  local dir = vim.fn.fnamemodify(cached_weights_path, ":h")
  vim.fn.mkdir(dir, "p")

  return cached_weights_path
end

--- Save weights to disk with optional latency tracking
---@param weights table
---@param latency_ctx? table Optional latency context for tracking
---@return boolean success
function M.save_weights(weights, latency_ctx)
  local latency = require("neural-open.latency")
  local weights_path = ensure_weights_file()

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
---@param latency_ctx? table Optional latency context
---@return table weights
function M.get_weights(latency_ctx)
  local latency = require("neural-open.latency")
  local weights_path = ensure_weights_file()

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

return M
