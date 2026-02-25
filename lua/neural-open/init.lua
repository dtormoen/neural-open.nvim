local M = {}

M.version = "0.0.1" -- x-release-please-version

---@class NosConfig
M.config = {
  algorithm = "nn", -- "naive" | "classic" | "nn"
  algorithm_config = {
    classic = {
      learning_rate = 0.6,
      default_weights = {
        match = 140, -- Snacks fuzzy matching
        virtual_name = 131, -- Virtual name matching
        open = 3, -- Open buffer bonus
        alt = 4, -- Alternate buffer bonus
        proximity = 13, -- Directory proximity
        project = 10, -- Project (cwd) bonus
        frecency = 17, -- Frecency score
        recency = 9, -- Recency score
        trigram = 10, -- Trigram similarity
        transition = 5, -- File transition tracking
      },
    },
    naive = {
      -- No configuration needed
    },
    nn = {
      architecture = { 10, 16, 16, 8, 1 }, -- Input → Hidden1 → Hidden2 → Hidden3 → Output
      optimizer = "adamw",
      learning_rate = 0.001,
      batch_size = 128,
      history_size = 2000,
      batches_per_update = 5,
      weight_decay = 0.0001, -- L2 regularization to prevent overfitting
      layer_decay_multipliers = nil, -- Optional per-layer decay rates
      dropout_rates = { 0, 0.25, 0.25 }, -- Optional dropout rates for hidden layers (not applied to output)
      warmup_steps = 10, -- Number of steps to warm up learning rate (recommended for AdamW)
      warmup_start_factor = 0.1, -- Start at 10% of learning rate
      adam_beta1 = 0.9, -- AdamW first moment decay
      adam_beta2 = 0.999, -- AdamW second moment decay
      adam_epsilon = 1e-8, -- AdamW numerical stability
      match_dropout = 0.25, -- Dropout rate for match/virtual_name features during training
      margin = 1.0, -- Margin for pairwise hinge loss
    },
  },
  weights_path = vim.fn.stdpath("data") .. "/neural-open/weights.json",
  special_files = {
    ["__init__.py"] = true,
    ["index.js"] = true,
    ["index.jsx"] = true,
    ["index.ts"] = true,
    ["index.tsx"] = true,
    ["init.lua"] = true,
    ["init.vim"] = true,
    ["mod.rs"] = true,
  },
  transition_history_size = 200, -- Ring buffer size for transition history
  recency_list_size = 100, -- Maximum number of files in persistent recency list
  -- Debug settings (all optional, for development/troubleshooting)
  debug = {
    preview = false, -- Show detailed score breakdown in preview
    latency = false, -- Log detailed latency metrics for performance debugging
    latency_file = nil, -- Optional file path for persistent latency logging
    latency_threshold_ms = 100, -- Only log operations exceeding this duration
    latency_auto_clipboard = false, -- Copy timing report to clipboard
    snacks_scores = false, -- Show Snacks.nvim debug scores in picker
  },
}

-- Flag to prevent concurrent weight updates
local pending_update = false

-- Lazy initialization flag
M._initialized = false

-- Confirm handler for file selection with learning
---@type snacks.picker.Action.fn
local function confirm_handler(picker, item)
  -- Create timing context for this selection (nil if disabled)
  local latency = require("neural-open.latency")
  local timing_ctx = latency.create_context()

  -- First do the default file opening behavior
  latency.start(timing_ctx, "confirm.file_open")
  local actions = require("snacks.picker.actions")
  actions.jump(picker, item, { action = function() end, cmd = "edit" })
  latency.finish(timing_ctx, "confirm.file_open")

  -- Then add our custom logic (asynchronously to avoid blocking UI)
  if item and item.file then
    -- Get the actual visible rank (position in filtered list)
    local items, items_ok = latency.measure(timing_ctx, "confirm.get_items", function()
      return picker:items()
    end)
    local visible_rank = nil

    if items_ok then
      -- Find the item's position in the filtered/sorted list
      visible_rank = latency.measure(timing_ctx, "confirm.find_rank", function()
        for i, list_item in ipairs(items) do
          if list_item.file == item.file then
            return i
          end
        end
        return nil
      end)
    end

    -- Schedule both transition recording and weight updates together to avoid race conditions
    -- Pass timing_ctx into async context (captured by closure - thread safe!)
    vim.schedule(function()
      -- Record transition for future scoring (source_file -> destination_file)
      if item.nos and item.nos.ctx then
        local source_file = item.nos.ctx.current_file
        local dest_file = item.nos.normalized_path

        -- Only record if we have a valid source and dest is different
        if source_file and source_file ~= "" and dest_file ~= source_file then
          latency.start(timing_ctx, "async.transition_record")
          local transitions = require("neural-open.transitions")
          transitions.record_transition(source_file, dest_file, timing_ctx)
          latency.finish(timing_ctx, "async.transition_record")
        end
      end

      -- Update weights if not already pending (check inside vim.schedule to prevent race)
      if visible_rank then
        -- Check and set pending_update atomically within the scheduled function
        if not pending_update then
          local nos_ctx = item.nos and item.nos.ctx
          if nos_ctx and nos_ctx.algorithm and nos_ctx.algorithm.update_weights then
            local algorithm = nos_ctx.algorithm

            -- Set neural_rank for weight learning
            item.neural_rank = visible_rank

            latency.start(timing_ctx, "async.weight_update")
            pending_update = true
            local ok, err = pcall(algorithm.update_weights, item, items, timing_ctx)
            if not ok then
              vim.notify("neural-open: Failed to update weights: " .. tostring(err), vim.log.levels.ERROR)
            end
            pending_update = false
            latency.finish(timing_ctx, "async.weight_update")
          end
        end
      end

      -- Log all timing data at the end
      latency.log_context(timing_ctx, item.file)
    end)
  end
end

-- Helper function to get neural-open source configuration
local function get_neural_source_config()
  return {
    finder = function(opts, ctx)
      -- Capture context early before any async operations
      local source_mod = require("neural-open.source")
      source_mod.capture_context(ctx)

      -- Use the multi finder with the captured context
      local Finder = require("snacks.picker.core.finder")
      local snacks = require("snacks")
      local multi_sources = { "buffers", "recent", "files", "git_files" }
      local finders = {}

      for _, source_name in ipairs(multi_sources) do
        local source_config = vim.deepcopy(snacks.picker.sources[source_name])
        local finder = require("snacks.picker.config").finder(source_config.finder)
        finders[#finders + 1] = finder
      end

      return Finder.multi(finders)(opts, ctx)
    end,
    format = "file",
    preview = function(ctx)
      if M.config.debug.preview then
        local debug = require("neural-open.debug")
        return debug.debug_preview(ctx)
      else
        return require("snacks.picker.preview").file(ctx)
      end
    end,
    transform = require("neural-open.source").create_neural_transform(M.config, require("neural-open.scorer"), {}),
    matcher = {
      sort_empty = true,
      frecency = true,
      cwd_bonus = false, -- Disable CWD bonus - we handle this in our scorer
      on_match = require("neural-open.scorer").on_match_handler,
    },
    sort = {
      fields = { "score:desc", "idx" },
    },
    confirm = confirm_handler,
    debug = {
      scores = M.config.debug.snacks_scores,
    },
  }
end

--- Ensures the plugin is initialized (registers Snacks source, sets up latency tracking)
--- Called automatically on first open() call
local function ensure_initialized()
  if M._initialized then
    return
  end
  M._initialized = true

  -- Register the source with Snacks
  local snacks = require("snacks")
  snacks.picker.sources = snacks.picker.sources or {}
  snacks.picker.sources.neural_open = get_neural_source_config()

  -- Enable latency tracking based on config
  local latency = require("neural-open.latency")
  latency.set_enabled(M.config.debug.latency)
end

function M.setup(opts)
  M.config = vim.tbl_deep_extend("force", M.config, opts or {})

  -- If already initialized, re-initialize to apply new config
  if M._initialized then
    M._initialized = false
    ensure_initialized()
  end
end

function M.open(opts)
  ensure_initialized()
  local snacks = require("snacks")
  snacks.picker.pick("neural_open", opts)
end

function M.reset_weights(algorithm_name)
  local weights = require("neural-open.weights")

  algorithm_name = algorithm_name or M.config.algorithm or "classic"

  local defaults = nil
  if algorithm_name == "classic" then
    defaults = M.config.algorithm_config.classic.default_weights
  end

  weights.reset_weights(algorithm_name, defaults)
  vim.notify(string.format("Reset weights for %s algorithm", algorithm_name), vim.log.levels.INFO)
end

-- Valid algorithm names
local valid_algorithms = { "classic", "naive", "nn" }

--- Sets the current algorithm or displays algorithm information
---@param algorithm_name string|nil Algorithm name or nil to show current
function M.set_algorithm(algorithm_name)
  if not algorithm_name then
    -- Show current algorithm
    local current = M.config.algorithm or "classic"
    vim.notify(string.format("Current algorithm: %s\nAvailable: classic, naive, nn", current), vim.log.levels.INFO)
  else
    -- Validate algorithm name
    if vim.tbl_contains(valid_algorithms, algorithm_name) then
      M.config.algorithm = algorithm_name
      vim.notify(string.format("Switched to %s algorithm", algorithm_name), vim.log.levels.INFO)
    else
      vim.notify(
        string.format("Algorithm '%s' not found. Available: classic, naive, nn", algorithm_name),
        vim.log.levels.ERROR
      )
    end
  end
end

--- Handles NeuralOpen command with subcommands
---@param args table Command arguments from nvim_create_user_command
function M.command(args)
  local subcommand = args.fargs[1]

  if not subcommand then
    M.open()
  elseif subcommand == "algorithm" then
    M.set_algorithm(args.fargs[2])
  elseif subcommand == "reset" then
    M.reset_weights(args.fargs[2])
  else
    vim.notify("Unknown subcommand: " .. subcommand, vim.log.levels.ERROR)
  end
end

--- Provides command completion for NeuralOpen
---@param arg_lead string Current argument being typed
---@param cmd_line string Full command line
---@param cursor_pos number Cursor position
---@return string[] Completion candidates
function M.complete(arg_lead, cmd_line, cursor_pos)
  ---@diagnostic disable-next-line: unused-local
  local _ = cursor_pos -- Unused but required by command completion API

  local args = vim.split(cmd_line, "%s+")
  -- args[1] is the command name itself (NeuralOpen)
  -- args[2] would be the subcommand, args[3] would be subcommand argument

  if #args <= 2 then
    -- Complete subcommands
    local subcommands = { "algorithm", "reset" }
    return vim.tbl_filter(function(s)
      return s:find(arg_lead, 1, true) == 1
    end, subcommands)
  elseif args[2] == "algorithm" or args[2] == "reset" then
    -- Complete algorithm names
    return vim.tbl_filter(function(s)
      return s:find(arg_lead, 1, true) == 1
    end, valid_algorithms)
  end
  return {}
end

return M
