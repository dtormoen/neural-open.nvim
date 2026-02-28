--- Simplified algorithm registry that returns the appropriate algorithm
local M = {}

--- Load an algorithm module by name, with validation.
---@param algorithm_name AlgorithmName
---@return Algorithm|{create_instance: fun(config: table): Algorithm}
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

--- Initialize an algorithm module with the given config.
--- Uses create_instance() when available, falls back to init() + module.
---@param algorithm Algorithm|{create_instance: fun(config: table): Algorithm} The algorithm module
---@param algo_config table Algorithm-specific configuration
---@return Algorithm The algorithm instance (may be the module itself or a new instance)
local function init_algorithm(algorithm, algo_config)
  if algorithm.create_instance then
    return algorithm.create_instance(algo_config)
  end
  if algorithm.init then
    algorithm.init(algo_config)
  end
  return algorithm
end

--- Get an algorithm based on global configuration (for the default file picker).
---@return Algorithm The algorithm instance
function M.get_algorithm()
  local config = require("neural-open").config
  local algorithm_name = config.algorithm
  local algorithm = load_algorithm(algorithm_name)
  local algo_config = config.algorithm_config and config.algorithm_config[algorithm_name] or {}
  return init_algorithm(algorithm, algo_config)
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
  local algo_config = vim.deepcopy(algorithm_config[algorithm_name] or {})
  algo_config.picker_name = picker_name
  if extra_config then
    for k, v in pairs(extra_config) do
      algo_config[k] = v
    end
  end
  return init_algorithm(algorithm, algo_config)
end

return M
