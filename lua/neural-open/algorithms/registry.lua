--- Simplified algorithm registry that returns the appropriate algorithm
local M = {}

--- Load an algorithm module by name, with validation.
---@param algorithm_name AlgorithmName
---@return Algorithm
local function load_algorithm(algorithm_name)
  if algorithm_name ~= "classic" and algorithm_name ~= "naive" and algorithm_name ~= "nn" then
    vim.notify(string.format("Invalid algorithm '%s', falling back to 'classic'", algorithm_name), vim.log.levels.ERROR)
    algorithm_name = "classic"
  end

  if algorithm_name == "classic" then
    return require("neural-open.algorithms.classic")
  elseif algorithm_name == "naive" then
    return require("neural-open.algorithms.naive")
  else
    return require("neural-open.algorithms.nn")
  end
end

--- Get an algorithm based on global configuration (for the default file picker).
---@return Algorithm The algorithm instance
function M.get_algorithm()
  local config = require("neural-open").config
  local algorithm_name = config.algorithm
  local algorithm = load_algorithm(algorithm_name)

  if algorithm.init then
    local algo_config = config.algorithm_config and config.algorithm_config[algorithm_name] or {}
    algorithm.init(algo_config)
  end

  return algorithm
end

--- Get an algorithm configured for a specific picker.
--- Allows passing a different algorithm_config and picker_name for weight isolation.
---@param algorithm_name AlgorithmName Algorithm to use
---@param algorithm_config NosAlgorithmConfig Algorithm config table (e.g., item_algorithm_config)
---@param picker_name string Picker name for weight file isolation
---@param extra_config? table Additional config fields to inject (e.g., feature_names)
---@return Algorithm The algorithm instance
function M.get_algorithm_for_picker(algorithm_name, algorithm_config, picker_name, extra_config)
  local algorithm = load_algorithm(algorithm_name)

  if algorithm.init then
    local algo_config = vim.deepcopy(algorithm_config[algorithm_name] or {})
    algo_config.picker_name = picker_name
    if extra_config then
      for k, v in pairs(extra_config) do
        algo_config[k] = v
      end
    end
    algorithm.init(algo_config)
  end

  return algorithm
end

return M
