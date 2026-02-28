--- Weight management module for neural-open
--- Handles loading, saving, and managing algorithm weights
local M = {}

local weights_db = nil

--- Ensure database is initialized
---@return table
local function ensure_db()
  if not weights_db then
    weights_db = require("neural-open.db")
  end
  return weights_db
end

--- Get default weights for an algorithm from main config.
--- Uses item_algorithm_config for non-file pickers when available.
---@param algorithm_name AlgorithmName
---@param picker_name? string Picker name (default "files")
---@return table?
function M.get_default_weights(algorithm_name, picker_name)
  local config = require("neural-open").config
  local algo_config = config.algorithm_config
  if picker_name and picker_name ~= "files" and config.item_algorithm_config then
    algo_config = config.item_algorithm_config
  end
  local entry = algo_config and algo_config[algorithm_name]
  return entry and entry.default_weights
end

--- Load weights for a specific algorithm from database
---@param algorithm_name string
---@param picker_name? string Picker name for storage isolation (default "files")
---@return table
function M.get_weights(algorithm_name, picker_name)
  local db = ensure_db()
  picker_name = picker_name or "files"

  local stored_data = db.get_weights(picker_name)

  -- Get weights for this specific algorithm
  local algorithm_weights = {}

  if stored_data and stored_data[algorithm_name] then
    algorithm_weights = stored_data[algorithm_name]
  end

  -- If weights are empty, initialize with defaults
  if vim.tbl_isempty(algorithm_weights) then
    local defaults = M.get_default_weights(algorithm_name, picker_name)
    if defaults then
      algorithm_weights = vim.deepcopy(defaults)
    end
  end

  return algorithm_weights
end

--- Save weights for a specific algorithm to database with optional latency tracking
---@param algorithm_name string
---@param weights table
---@param latency_ctx? table Optional latency context
---@param picker_name? string Picker name for storage isolation (default "files")
function M.save_weights(algorithm_name, weights, latency_ctx, picker_name)
  local latency = require("neural-open.latency")
  local db = ensure_db()
  picker_name = picker_name or "files"

  -- Load all current weights (measured in db.get_weights)
  local all_weights, ok = latency.measure(latency_ctx, "weights.get_all", function()
    return db.get_weights(picker_name, latency_ctx) or {}
  end, "nn.save_weights")

  if not ok then
    all_weights = {}
  end

  -- Update the specific algorithm's weights
  all_weights[algorithm_name] = weights

  -- Save back to database (measured in db.save_weights)
  latency.measure(latency_ctx, "weights.save_all", function()
    db.save_weights(picker_name, all_weights, latency_ctx)
  end, "nn.save_weights")
end

--- Reset weights for a specific algorithm
---@param algorithm_name string?
---@param defaults table?
---@param picker_name? string Picker name for storage isolation (default "files")
---@return table
function M.reset_weights(algorithm_name, defaults, picker_name)
  algorithm_name = algorithm_name or "classic"

  local reset_weights = vim.deepcopy(defaults or {})
  M.save_weights(algorithm_name, reset_weights, nil, picker_name)

  return reset_weights
end

return M
