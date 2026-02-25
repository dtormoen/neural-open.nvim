--- Picker hot-path benchmark for neural-open.nvim
--- Measures per-keystroke scoring latency across realistic repository sizes.
---
--- Run via: just benchmark
--- Requires nlua (Neovim Lua interpreter) with test isolation.

-- Setup package paths for project modules
package.path = "lua/?.lua;lua/?/init.lua;./?/init.lua;" .. package.path

-- Verify test isolation
local helpers = require("tests.helpers")
helpers.setup()

--------------------------------------------------------------------------------
-- Configuration
--------------------------------------------------------------------------------

local WARMUP_ITERATIONS = 5
local MEASURED_ITERATIONS = 50
local REPO_SIZES = { 1000, 10000, 100000 }

local SEED = 42

--------------------------------------------------------------------------------
-- Module imports
--------------------------------------------------------------------------------

local scorer = require("neural-open.scorer")
local trigrams = require("neural-open.trigrams")
-- Mock the weights module before requiring nn so load_weights() uses bundled defaults
package.loaded["neural-open.weights"] = {
  get_weights = function()
    return {}
  end,
  save_weights = function() end,
}

local nn = require("neural-open.algorithms.nn")

-- Initialize nn with default config
local config = helpers.get_default_config()
local nn_config = config.algorithm_config.nn
nn.init(nn_config)
nn.load_weights()

--------------------------------------------------------------------------------
-- Synthetic data generation
--------------------------------------------------------------------------------

local DIRECTORIES = {
  "src",
  "src/components",
  "src/components/ui",
  "src/components/forms",
  "src/utils",
  "src/hooks",
  "src/pages",
  "src/pages/auth",
  "src/pages/dashboard",
  "src/layouts",
  "src/styles",
  "src/types",
  "lib",
  "lib/core",
  "lib/adapters",
  "tests",
  "tests/unit",
  "tests/integration",
  "tests/fixtures",
  "docs",
  "config",
  "scripts",
  "api",
  "api/routes",
  "api/middleware",
  "api/controllers",
  "models",
  "models/schemas",
  "views",
  "views/partials",
  "services",
  "services/auth",
  "services/cache",
  "helpers",
}

local EXTENSIONS = {
  ".lua",
  ".js",
  ".ts",
  ".tsx",
  ".py",
  ".go",
  ".rs",
  ".md",
  ".json",
  ".yaml",
  ".css",
  ".html",
}

local BASE_NAMES = {
  "app",
  "main",
  "config",
  "utils",
  "helpers",
  "types",
  "constants",
  "router",
  "store",
  "context",
  "provider",
  "handler",
  "middleware",
  "service",
  "controller",
  "model",
  "schema",
  "validator",
  "factory",
  "adapter",
  "client",
  "server",
  "logger",
  "parser",
  "formatter",
  "renderer",
  "loader",
  "builder",
  "manager",
  "registry",
  "dispatcher",
  "emitter",
  "listener",
  "observer",
  "subscriber",
  "publisher",
  "worker",
  "queue",
  "cache",
  "pool",
  "buffer",
  "stream",
  "transform",
  "pipeline",
  "filter",
  "mapper",
  "reducer",
  "selector",
  "resolver",
  "connector",
}

--- Generate deterministic file paths resembling a real project.
---@param count number Number of paths to generate
---@return string[]
local function generate_file_paths(count)
  math.randomseed(SEED)
  local paths = {}
  local used = {}

  -- Sprinkle in special files (index.js, init.lua) for virtual name testing
  local special_files = { "index.js", "index.ts", "index.tsx", "init.lua", "mod.rs" }

  for i = 1, count do
    local path
    repeat
      local dir = DIRECTORIES[math.random(#DIRECTORIES)]
      local name
      if i <= #DIRECTORIES * #special_files and math.random() < 0.08 then
        name = special_files[math.random(#special_files)]
      else
        local base = BASE_NAMES[math.random(#BASE_NAMES)]
        local ext = EXTENSIONS[math.random(#EXTENSIONS)]
        -- Add numeric suffix for uniqueness at large scales
        if count > #DIRECTORIES * #BASE_NAMES then
          base = base .. math.random(1, math.ceil(count / (#DIRECTORIES * #BASE_NAMES)))
        end
        name = base .. ext
      end
      path = "/project/" .. dir .. "/" .. name
    until not used[path]
    used[path] = true
    paths[i] = path
  end

  return paths
end

--- Build a session context with realistic recency, transition, and trigram data.
---@param paths string[]
---@return NosContext
local function create_context(paths)
  math.randomseed(SEED + 1)

  local current_file = "/project/src/components/App.tsx"
  local current_file_vname = scorer.get_virtual_name(current_file, config.special_files)
  local current_file_tris = trigrams.compute_trigrams(current_file_vname)
  local current_file_tris_size = trigrams.count_trigrams(current_file_tris)

  -- Build recency map from first 100 paths
  local recent_files = {}
  local recency_count = math.min(100, #paths)
  for i = 1, recency_count do
    recent_files[paths[i]] = { recent_rank = i }
  end

  -- Build transition scores for first 50 paths
  local transition_scores = {}
  local transition_count = math.min(50, #paths)
  for i = 1, transition_count do
    transition_scores[paths[i]] = math.random() * 0.9
  end

  -- Precompute directory info for proximity calculations (matches source.lua)
  local current_file_dir, current_file_depth = scorer.compute_dir_info(current_file)

  return {
    cwd = "/project",
    current_file = current_file,
    current_file_dir = current_file_dir,
    current_file_depth = current_file_depth,
    current_file_trigrams = current_file_tris,
    current_file_trigrams_size = current_file_tris_size,
    recent_files = recent_files,
    alternate_buf = paths[1],
    algorithm = nn,
    transition_scores = transition_scores,
  }
end

--- Build items with pre-computed static features, simulating the transform phase.
---@param paths string[]
---@param context NosContext
---@return NeuralOpenItem[]
local function create_items(paths, context)
  math.randomseed(SEED + 2)

  local items = {}
  local open_count = math.min(10, #paths)

  for i, path in ipairs(paths) do
    local is_open = i <= open_count
    local is_alt = (path == context.alternate_buf)
    local recent_entry = context.recent_files[path]
    local virtual_name = scorer.get_virtual_name(path, config.special_files)

    local recent_rank = recent_entry and recent_entry.recent_rank or nil

    local raw_features = scorer.compute_static_raw_features(path, context, is_open, is_alt, recent_rank, virtual_name)

    -- Set plausible dynamic feature values (normally set per-keystroke by matcher)
    raw_features.match = math.random(0, 300)
    raw_features.virtual_name = math.random(0, 200)
    raw_features.frecency = math.random() * 50

    -- Pre-allocate input_buf (mirrors source.lua transform phase)
    local recency_val = 0
    if recent_rank and recent_rank > 0 then
      recency_val = scorer.calculate_recency_score(recent_rank)
    end
    local input_buf = {
      0, -- [1] match (dynamic)
      0, -- [2] virtual_name (dynamic)
      0, -- [3] frecency (dynamic)
      is_open and 1 or 0, -- [4] open
      is_alt and 1 or 0, -- [5] alt
      raw_features.proximity, -- [6] proximity (already [0,1])
      raw_features.project, -- [7] project (already 0/1)
      recency_val, -- [8] recency (normalized)
      raw_features.trigram, -- [9] trigram (already [0,1])
      raw_features.transition, -- [10] transition (already [0,1])
    }

    items[i] = {
      file = path,
      text = path,
      nos = {
        raw_features = raw_features,
        neural_score = 0,
        normalized_path = path,
        is_open_buffer = is_open,
        is_alternate = is_alt,
        recent_rank = recent_rank,
        virtual_name = virtual_name,
        input_buf = input_buf,
        ctx = context,
      },
    }
  end

  return items
end

--------------------------------------------------------------------------------
-- Timing utilities
--------------------------------------------------------------------------------

local hrtime = vim.uv.hrtime

--- Run a function for warmup + measured iterations and return the median elapsed time in nanoseconds.
---@param fn fun() The function to benchmark
---@return number median_ns
---@return number mean_ns
local function benchmark(fn)
  -- Warmup
  for _ = 1, WARMUP_ITERATIONS do
    fn()
  end

  -- Collect measurements
  local times = {}
  for i = 1, MEASURED_ITERATIONS do
    local start = hrtime()
    fn()
    times[i] = hrtime() - start
  end

  -- Sort for median
  table.sort(times)
  local mid = math.floor(MEASURED_ITERATIONS / 2)
  local median
  if MEASURED_ITERATIONS % 2 == 0 then
    median = (times[mid] + times[mid + 1]) / 2
  else
    median = times[mid + 1]
  end

  -- Mean
  local sum = 0
  for _, t in ipairs(times) do
    sum = sum + t
  end
  local mean = sum / MEASURED_ITERATIONS

  return median, mean
end

--------------------------------------------------------------------------------
-- Formatting
--------------------------------------------------------------------------------

--- Format nanoseconds as a human-readable string with total and per-item cost.
---@param ns number Total nanoseconds
---@param item_count number Number of items processed
---@return string
local function format_timing(ns, item_count)
  local total_ms = ns / 1e6
  local per_item_us = ns / 1e3 / item_count
  return string.format("%8.2fms total | %6.3fus/item", total_ms, per_item_us)
end

--------------------------------------------------------------------------------
-- Benchmark runner
--------------------------------------------------------------------------------

local arch = table.concat(nn_config.architecture, " -> ")
print("=== Neural-Open Picker Benchmark ===")
print(string.format("Algorithm: nn (%s)", arch))
print(string.format("Date: %s", os.date("%Y-%m-%d")))
print(string.format("Iterations: %d", MEASURED_ITERATIONS))

for _, size in ipairs(REPO_SIZES) do
  -- Generate data for this repo size
  local paths = generate_file_paths(size)
  local context = create_context(paths)
  local items = create_items(paths, context)

  -- 1. Static feature computation (one-time per item)
  local static_median = benchmark(function()
    for _, item in ipairs(items) do
      scorer.compute_static_raw_features(
        item.nos.normalized_path,
        context,
        item.nos.is_open_buffer,
        item.nos.is_alternate,
        item.nos.recent_rank,
        item.nos.virtual_name
      )
    end
  end)

  -- 2. Per-keystroke: update dynamic features in input_buf + inference (zero table allocation)
  local keystroke_median = benchmark(function()
    for _, item in ipairs(items) do
      local rf = item.nos.raw_features
      local input_buf = item.nos.input_buf
      input_buf[1] = scorer.normalize_match_score(rf.match)
      input_buf[2] = scorer.normalize_match_score(rf.virtual_name)
      input_buf[3] = scorer.normalize_frecency(rf.frecency)
      nn.calculate_score(input_buf)
    end
  end)

  -- 3. Normalize alone
  local normalize_median = benchmark(function()
    for _, item in ipairs(items) do
      scorer.normalize_features(item.nos.raw_features)
    end
  end)

  -- 4. NN inference alone (pre-normalize once, then measure inference)
  local pre_normalized = {}
  for i, item in ipairs(items) do
    local input_buf = item.nos.input_buf
    pre_normalized[i] = { unpack(input_buf) }
  end

  local nn_median = benchmark(function()
    for _, buf in ipairs(pre_normalized) do
      nn.calculate_score(buf)
    end
  end)

  -- 5. Trigram computation (isolated from static features)
  local virtual_names = {}
  for i, item in ipairs(items) do
    virtual_names[i] = item.nos.virtual_name or item.nos.normalized_path:match("[^/]+$") or ""
  end

  local trigram_median = benchmark(function()
    for i = 1, #virtual_names do
      local target_tris = trigrams.compute_trigrams(virtual_names[i])
      trigrams.dice_coefficient(context.current_file_trigrams, target_tris)
    end
  end)

  -- 5b. Trigram computation (zero-allocation direct path, used in production)
  local trigram_direct_median = benchmark(function()
    for i = 1, #virtual_names do
      trigrams.dice_coefficient_direct(
        context.current_file_trigrams,
        context.current_file_trigrams_size,
        virtual_names[i]
      )
    end
  end)

  -- 6. Transform phase: full per-item processing during discovery
  -- Matches source.lua create_neural_transform() including nn_input alloc and nos table creation
  local transform_median = benchmark(function()
    local done = {}
    for _, item in ipairs(items) do
      local path = item.nos.normalized_path
      if not done[path] then
        done[path] = true
        local is_open_buffer = item.nos.is_open_buffer
        local is_alternate = item.nos.is_alternate
        local recent_rank = item.nos.recent_rank
        local virtual_name = scorer.get_virtual_name(path, config.special_files)
        local raw_features =
          scorer.compute_static_raw_features(path, context, is_open_buffer, is_alternate, recent_rank, virtual_name)
        -- NN input buffer allocation (mirrors source.lua)
        local recency_val = 0
        if recent_rank and recent_rank > 0 then
          recency_val = scorer.calculate_recency_score(recent_rank)
        end
        local _nn_input = {
          0,
          0,
          0,
          is_open_buffer and 1 or 0,
          is_alternate and 1 or 0,
          raw_features.proximity,
          raw_features.project,
          recency_val,
          raw_features.trigram,
          raw_features.transition,
        }
        -- nos table creation (mirrors source.lua)
        local _nos = { -- luacheck: ignore 211
          normalized_path = path,
          virtual_name = virtual_name,
          is_open_buffer = is_open_buffer,
          is_alternate = is_alternate,
          recent_rank = recent_rank,
          nn_input = _nn_input,
          ctx = context,
          raw_features = raw_features,
          normalized_features = {},
          neural_score = 0,
        }
      end
    end
  end)

  -- 7. Weight loading: load_weights triggers ensure_weights(true) + prepare_inference_cache
  local load_median = benchmark(function()
    nn.load_weights()
  end)

  -- Print results
  local formatted_size = tostring(size):reverse():gsub("(%d%d%d)", "%1,"):reverse():gsub("^,", "")
  print("")
  print(string.format("--- %s files ---", formatted_size))
  print(string.format("Static features:    %s", format_timing(static_median, size)))
  print(string.format("Per-keystroke:      %s", format_timing(keystroke_median, size)))
  print(string.format("  normalize:        %s", format_timing(normalize_median, size)))
  print(string.format("  nn_inference:     %s", format_timing(nn_median, size)))
  print(string.format("  trigrams (alloc): %s", format_timing(trigram_median, size)))
  print(string.format("  trigrams (direct):%s", format_timing(trigram_direct_median, size)))
  print(string.format("Transform phase:    %s", format_timing(transform_median, size)))
  print(string.format("Weight loading:     %8.2fms total (one-time)", load_median / 1e6))
end
