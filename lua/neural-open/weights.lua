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

--- Get default weights for an algorithm from main config
---@param algorithm_name AlgorithmName
---@return table
function M.get_default_weights(algorithm_name)
  local config = require("neural-open").config
  return config.algorithm_config[algorithm_name].default_weights
end

--- Load weights for a specific algorithm from database
---@param algorithm_name string
---@return table
function M.get_weights(algorithm_name)
  local db = ensure_db()

  local stored_data = db.get_weights()

  -- Get weights for this specific algorithm
  local algorithm_weights = {}

  if stored_data and stored_data[algorithm_name] then
    algorithm_weights = stored_data[algorithm_name]
  end

  -- If weights are empty, initialize with defaults
  if vim.tbl_isempty(algorithm_weights) then
    local defaults = M.get_default_weights(algorithm_name)
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
function M.save_weights(algorithm_name, weights, latency_ctx)
  local latency = require("neural-open.latency")
  local db = ensure_db()

  -- Load all current weights (measured in db.get_weights)
  local all_weights, ok = latency.measure(latency_ctx, "weights.get_all", function()
    return db.get_weights(latency_ctx) or {}
  end, "nn.save_weights")

  if not ok then
    all_weights = {}
  end

  -- Update the specific algorithm's weights
  all_weights[algorithm_name] = weights

  -- Save back to database (measured in db.save_weights)
  latency.measure(latency_ctx, "weights.save_all", function()
    db.save_weights(all_weights, latency_ctx)
  end, "nn.save_weights")
end

--- Reset weights for a specific algorithm
---@param algorithm_name string?
---@param defaults table?
---@return table
function M.reset_weights(algorithm_name, defaults)
  algorithm_name = algorithm_name or "classic"

  local reset_weights = vim.deepcopy(defaults or {})
  M.save_weights(algorithm_name, reset_weights)

  return reset_weights
end

return M
