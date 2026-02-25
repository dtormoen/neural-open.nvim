-- Test helper module for neural-open.nvim
-- Provides isolation verification and common test utilities

local M = {}

-- Verify test isolation is active
local function verify_isolation()
  -- Check testing flag
  if not vim.env.NEURAL_OPEN_TESTING then
    error(
      [[

FATAL: Tests are running without isolation!

Your real Neovim environment could be affected.
Please run tests using: just test

If you need to run without isolation for debugging,
use: just test-no-isolation
]],
      2
    )
  end

  -- Verify XDG directories are isolated
  local data_home = vim.fn.stdpath("data")
  local config_home = vim.fn.stdpath("config")
  local cache_home = vim.fn.stdpath("cache")
  local state_home = vim.fn.stdpath("state")

  local home = vim.env.HOME
  if
    home
    and (
      data_home:find(home, 1, true)
      or config_home:find(home, 1, true)
      or cache_home:find(home, 1, true)
      or state_home:find(home, 1, true)
    )
  then
    -- Only fail if it's actually in the user's home directory
    -- (not in a temp directory that happens to be under home)
    if not data_home:match("/tmp/") and not data_home:match("/var/folders/") then
      error([[

FATAL: Test isolation check failed!

XDG directories are pointing to your real home directory.
This should not happen when using 'just test'.

Data home: ]] .. data_home .. [[

Config home: ]] .. config_home .. [[

Please ensure you're running tests with: just test
]], 2)
    end
  end
end

-- Setup function to be called at the beginning of test files
function M.setup()
  verify_isolation()

  -- Return useful test info
  return {
    data_dir = vim.fn.stdpath("data"),
    config_dir = vim.fn.stdpath("config"),
    cache_dir = vim.fn.stdpath("cache"),
    state_dir = vim.fn.stdpath("state"),
    is_isolated = true,
  }
end

-- Clear all package.loaded entries for the plugin
function M.clear_plugin_modules()
  local modules_to_clear = {
    "neural-open",
    "neural-open.scorer",
    "neural-open.weights",
    "neural-open.db",
    "neural-open.source",
    "neural-open.algorithms.registry",
    "neural-open.algorithms.classic",
    "neural-open.algorithms.naive",
    "neural-open.algorithms.nn",
    "neural-open.algorithms.nn_core",
    "neural-open.recent",
  }

  for _, module in ipairs(modules_to_clear) do
    package.loaded[module] = nil
  end
end

-- Create a test context with common defaults
function M.create_test_context(opts)
  opts = opts or {}
  return vim.tbl_extend("force", {
    cwd = "/test/project",
    current_file = "/test/project/current.lua",
    query = "",
  }, opts)
end

-- Create a test item with common defaults
---@param opts table?
---@return NeuralOpenItem
function M.create_test_item(opts)
  opts = opts or {}
  local file = opts.file or "/test/project/file.lua"

  -- If nos field is provided, use it directly
  if opts.nos then
    return vim.tbl_extend("force", {
      file = file,
      text = opts.text or file,
      nos = opts.nos,
    }, {
      frecency = opts.frecency,
      score = opts.score,
      buf = opts.buf,
      neural_rank = opts.neural_rank,
    })
  end

  -- Otherwise, create the new structure from provided fields
  local normalized_path = opts.normalized_path or file
  local is_open_buffer = opts.is_open_buffer or false
  local is_alternate = opts.is_alternate or false

  return {
    file = file,
    text = opts.text or file,
    nos = {
      normalized_path = normalized_path,
      is_open_buffer = is_open_buffer,
      is_alternate = is_alternate,
      raw_features = opts.raw_features or {},
      normalized_features = opts.normalized_features or {},
      neural_score = opts.neural_score or 0,
      recent_rank = opts.recent_rank,
      virtual_name = opts.virtual_name,
    },
    frecency = opts.frecency,
    score = opts.score,
    buf = opts.buf,
    neural_rank = opts.neural_rank,
  }
end

-- Utility to safely create and cleanup test buffers
function M.with_test_buffer(name, callback)
  local buf = vim.api.nvim_create_buf(false, true)
  vim.api.nvim_buf_set_name(buf, name)

  local ok, result = pcall(callback, buf)

  -- Always cleanup
  if vim.api.nvim_buf_is_valid(buf) then
    vim.api.nvim_buf_delete(buf, { force = true })
  end

  if not ok then
    error(result)
  end

  return result
end

-- Create a temporary directory for test files
function M.create_temp_dir()
  local temp_dir

  -- Try different temp directory locations
  if vim.fn.has("unix") == 1 then
    temp_dir = os.tmpname()
    -- os.tmpname() returns a file, we want a directory
    os.remove(temp_dir)
    temp_dir = temp_dir .. "_dir"
  else
    temp_dir = vim.fn.tempname() .. "_dir"
  end

  vim.fn.mkdir(temp_dir, "p")
  return temp_dir
end

-- Clean up temporary directory and all its contents
function M.cleanup_temp_dir(temp_dir)
  if temp_dir and vim.fn.isdirectory(temp_dir) == 1 then
    vim.fn.delete(temp_dir, "rf")
  end
end

-- Utility to run tests with a temporary database
function M.with_temp_db(callback)
  local temp_dir = M.create_temp_dir()
  local temp_db_path = temp_dir .. "/test_weights.json"

  local ok, result = pcall(callback, temp_db_path)

  -- Always cleanup
  M.cleanup_temp_dir(temp_dir)

  if not ok then
    error(result)
  end

  return result
end

-- Get a fresh copy of default config from init.lua
-- Use for: Tests that need to inspect or verify default values
---@return NosConfig
function M.get_default_config()
  local init = require("neural-open")
  return vim.deepcopy(init.config)
end

-- Create full config with selective overrides
-- Use for: Tests that need to modify top-level config (algorithm, weights_path, debug.preview)
---@param overrides table? Partial config to merge
---@return NosConfig
function M.create_test_config(overrides)
  local config = M.get_default_config()
  return vim.tbl_deep_extend("force", config, overrides or {})
end

-- Create config with algorithm-specific overrides
-- Use for: Tests that only need to modify one algorithm's config (most common case)
---@param algorithm_name AlgorithmName "classic" | "naive" | "nn"
---@param overrides table? Partial algorithm config to merge
---@return NosConfig
function M.create_algorithm_config(algorithm_name, overrides)
  local config = M.get_default_config()
  if overrides then
    config.algorithm_config[algorithm_name] =
      vim.tbl_deep_extend("force", config.algorithm_config[algorithm_name], overrides)
  end
  return config
end

-- Create a simple weights module mock for testing
-- Returns a table with:
--   - mock: The mock module to assign to package.loaded
--   - get_saved: Function to retrieve saved weights for assertions
--
-- Usage:
--   local weights_mock = helpers.create_weights_mock()
--   package.loaded["neural-open.weights"] = weights_mock.mock
--
--   -- Later in test:
--   assert.is_not_nil(weights_mock.get_saved())
--
function M.create_weights_mock()
  local saved_weights = nil

  return {
    mock = {
      get_weights = function(algo_name)
        return saved_weights or {}
      end,
      save_weights = function(algo_name, weights)
        saved_weights = weights
      end,
    },
    get_saved = function()
      return saved_weights
    end,
  }
end

-- Run isolation verification immediately when this module is loaded
verify_isolation()

return M
