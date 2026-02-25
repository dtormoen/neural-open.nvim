--- Simplified algorithm registry that returns the appropriate algorithm
local M = {}

--- Get an algorithm based on configuration
---@return Algorithm The algorithm instance
function M.get_algorithm()
  local config = require("neural-open").config
  local algorithm_name = config.algorithm

  -- Validate algorithm name
  if algorithm_name ~= "classic" and algorithm_name ~= "naive" and algorithm_name ~= "nn" then
    vim.notify(string.format("Invalid algorithm '%s', falling back to 'classic'", algorithm_name), vim.log.levels.ERROR)
    algorithm_name = "classic"
  end

  -- Load the algorithm module
  local algorithm
  if algorithm_name == "classic" then
    algorithm = require("neural-open.algorithms.classic")
  elseif algorithm_name == "naive" then
    algorithm = require("neural-open.algorithms.naive")
  elseif algorithm_name == "nn" then
    algorithm = require("neural-open.algorithms.nn")
  end

  -- Initialize the algorithm with its config if provided
  if algorithm.init then
    local algo_config = config.algorithm_config and config.algorithm_config[algorithm_name] or {}
    algorithm.init(algo_config)
  end

  return algorithm
end

return M
