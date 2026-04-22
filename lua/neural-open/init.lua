local M = {}

M.version = "0.1.5" -- x-release-please-version

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
        not_current = 5, -- Not-current-file bonus
      },
    },
    naive = {
      -- No configuration needed
    },
    nn = {
      architecture = { 11, 16, 16, 8, 1 }, -- Input → Hidden1 → Hidden2 → Hidden3 → Output
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
  item_algorithm_config = {
    classic = {
      learning_rate = 0.6,
      default_weights = {
        match = 140,
        frecency = 17,
        cwd_frecency = 15,
        recency = 9,
        cwd_recency = 8,
        text_length_inv = 3,
        not_last_selected = 2,
        transition = 5,
      },
    },
    naive = {},
    nn = {
      architecture = { 8, 16, 8, 1 }, -- 8 inputs for item features
      optimizer = "adamw",
      learning_rate = 0.001,
      batch_size = 128,
      history_size = 2000,
      batches_per_update = 5,
      weight_decay = 0.0001,
      dropout_rates = { 0, 0.25 },
      warmup_steps = 10,
      warmup_start_factor = 0.1,
      adam_beta1 = 0.9,
      adam_beta2 = 0.999,
      adam_epsilon = 1e-8,
      match_dropout = 0.25,
      margin = 1.0,
    },
  },
  weights_path = vim.fn.stdpath("data") .. "/neural-open/files.json",
  weights_dir = nil, -- Directory for all picker weight files (defaults to dirname of weights_path)
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
  recency_list_size = 100, -- Maximum number of files in persistent recency list
  file_sources = { "buffers", "neural_recent", "files", "git_files" },
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

-- Flag to prevent concurrent item weight updates
local pending_item_update = false

-- Registry of picker configurations
---@type table<string, NosPickerConfig>
local picker_registry = {}

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

        -- Only record if we have a valid source
        if source_file and source_file ~= "" then
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

-- Confirm handler for item selection with learning
---@param picker_name string Picker name for weight isolation
---@param user_confirm fun(picker: table, item: table)? User's original confirm function
---@return snacks.picker.Action.fn
local function create_item_confirm_handler(picker_name, user_confirm)
  return function(picker, item)
    -- Call user's confirm function first
    if user_confirm then
      user_confirm(picker, item)
    end

    -- Then add learning logic asynchronously
    if item and item.nos then
      local item_id = item.nos.item_id
      local nos_ctx = item.nos.ctx

      -- Get visible rank before picker closes
      local items = picker:items()
      local visible_rank = nil
      if items then
        for i, list_item in ipairs(items) do
          if list_item.nos and list_item.nos.item_id == item_id then
            visible_rank = i
            break
          end
        end
      end

      vim.schedule(function()
        -- Record selection for tracking
        if nos_ctx then
          local item_tracking = require("neural-open.item_tracking")
          item_tracking.record_selection(picker_name, item_id, nos_ctx.cwd)
        end

        -- Update weights if not already pending
        if visible_rank and not pending_item_update then
          if nos_ctx and nos_ctx.algorithm and nos_ctx.algorithm.update_weights then
            item.neural_rank = visible_rank
            pending_item_update = true
            local ok, err = pcall(nos_ctx.algorithm.update_weights, item, items)
            if not ok then
              vim.notify("neural-open: Failed to update item weights: " .. tostring(err), vim.log.levels.ERROR)
            end
            pending_item_update = false
          end
        end
      end)
    end
  end
end

-- Build an effective config that merges per-picker overrides over the global config.
---@param picker_config NosPickerConfig
---@return NosConfig
local function build_effective_config(picker_config)
  if not picker_config.algorithm and not picker_config.algorithm_config then
    return M.config
  end

  local effective = vim.deepcopy(M.config)
  if picker_config.algorithm then
    effective.algorithm = picker_config.algorithm
  end
  if picker_config.algorithm_config then
    effective.algorithm_config =
      vim.tbl_deep_extend("force", effective.algorithm_config, picker_config.algorithm_config)
    effective.item_algorithm_config =
      vim.tbl_deep_extend("force", effective.item_algorithm_config or {}, picker_config.algorithm_config)
  end
  return effective
end

-- Build a Snacks source config for a custom file picker
---@param picker_name string
---@param picker_config NosPickerConfig
---@return table Snacks source config
local function build_file_source_config(picker_name, picker_config)
  local source_mod = require("neural-open.source")
  local scorer = require("neural-open.scorer")
  local effective_config = build_effective_config(picker_config)

  return {
    finder = function(opts, ctx)
      source_mod.capture_context(ctx, picker_name, effective_config)
      if picker_config.finder then
        return picker_config.finder(opts, ctx)
      end
    end,
    format = picker_config.format or "file",
    preview = function(ctx)
      if M.config.debug.preview then
        ctx.meta = ctx.meta or {}
        ctx.meta.nos_user_preview = picker_config.preview
        local debug_mod = require("neural-open.debug")
        return debug_mod.debug_preview(ctx)
      elseif picker_config.preview then
        return picker_config.preview(ctx)
      else
        return require("snacks.picker.preview").file(ctx)
      end
    end,
    transform = source_mod.create_neural_transform(effective_config, scorer, {}),
    matcher = {
      sort_empty = true,
      frecency = true,
      cwd_bonus = false,
      on_match = scorer.on_match_handler,
    },
    sort = { fields = { "score:desc", "idx" } },
    confirm = picker_config.confirm or confirm_handler,
    title = picker_config.title,
    debug = { scores = M.config.debug.snacks_scores },
  }
end

-- Build a Snacks source config for an item picker
---@param picker_name string
---@param picker_config NosPickerConfig
---@return table Snacks source config
local function build_item_source_config(picker_name, picker_config)
  local item_source = require("neural-open.item_source")
  local item_scorer = require("neural-open.item_scorer")
  local effective_config = build_effective_config(picker_config)

  local finder
  if picker_config.finder then
    finder = function(opts, ctx)
      item_source.capture_context(picker_name, ctx, effective_config)
      return picker_config.finder(opts, ctx)
    end
  elseif picker_config.items then
    finder = function(opts, ctx)
      item_source.capture_context(picker_name, ctx, effective_config)
      return picker_config.items
    end
  end

  return {
    finder = finder,
    format = picker_config.format,
    preview = function(ctx)
      if M.config.debug.preview then
        ctx.meta = ctx.meta or {}
        ctx.meta.nos_user_preview = picker_config.preview
        local debug_mod = require("neural-open.debug")
        return debug_mod.debug_preview(ctx)
      elseif picker_config.preview then
        return picker_config.preview(ctx)
      end
    end,
    transform = item_source.create_item_transform(picker_name, effective_config, item_scorer),
    matcher = {
      sort_empty = true,
      on_match = item_scorer.on_match_handler,
    },
    sort = { fields = { "score:desc", "idx" } },
    confirm = create_item_confirm_handler(picker_name, picker_config.confirm),
    title = picker_config.title,
    actions = picker_config.actions,
    win = picker_config.win,
    debug = { scores = M.config.debug.snacks_scores },
  }
end

-- Wrap git_files finder so it lists the whole repo but does NOT call
-- ctx.picker:set_cwd(git_root), which would clobber the picker's cwd (breaking
-- scoring's project bonus and preview path resolution when nvim is launched
-- from a subdirectory of the repo).
--
-- We must propagate cwd through the ctx as well: git_files's inner proc call
-- resolves opts.cwd through ctx._opts -> picker.opts. If we only pin the opts
-- table, proc falls back to picker.opts.cwd (nil) and git ls-files runs from
-- uv.cwd() (the subdir), yielding paths relative to the subdir but stamped
-- with item.cwd=git_root, producing bogus absolute paths that skip our dedup.
---@param inner fun(opts: table, ctx: table): any The raw git_files finder
---@param git_root string Absolute path to git root
---@return fun(opts: table, ctx: table): any Wrapped finder
function M._pin_git_files(inner, git_root)
  return function(inner_opts, inner_ctx)
    local pinned = vim.tbl_extend("force", inner_opts or {}, { cwd = git_root })
    return inner(pinned, inner_ctx:clone(pinned))
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
      local multi_sources = M.config.file_sources
      -- Detect git repo and capture its root in a single call
      -- (handles worktrees, GIT_DIR, submodules; fails for bare repos, which is fine)
      local git_root = nil
      if vim.fn.executable("git") == 1 then
        local output = vim.fn.system({ "git", "rev-parse", "--show-toplevel" })
        if vim.v.shell_error == 0 then
          local trimmed = vim.trim(output)
          if trimmed ~= "" then
            git_root = trimmed
          end
        end
      end
      local finders = {}

      for _, source_name in ipairs(multi_sources) do
        if source_name ~= "git_files" or git_root then
          local source_config = vim.deepcopy(snacks.picker.sources[source_name])
          local finder = require("snacks.picker.config").finder(source_config.finder)
          if source_name == "git_files" then
            finder = M._pin_git_files(finder, assert(git_root))
          end
          finders[#finders + 1] = finder
        end
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
  snacks.picker.sources.neural_recent = {
    finder = require("neural-open.recent_finder").finder,
  }

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

--- Register a picker configuration for later use.
---@param name string Picker name (used for weight file namespacing)
---@param config NosPickerConfig Picker configuration
function M.register_picker(name, config)
  config.type = config.type or "item"
  picker_registry[name] = config
end

--- Open a picker with neural scoring.
--- If not previously registered, registers with the provided opts.
--- If previously registered, merges opts over the registered config.
---@param name string Picker name
---@param opts? NosPickerConfig Picker configuration (merged over registered config)
function M.pick(name, opts)
  ensure_initialized()
  opts = opts or {}

  -- Register or merge with existing registration
  if not picker_registry[name] then
    M.register_picker(name, opts)
  elseif next(opts) then
    picker_registry[name] = vim.tbl_deep_extend("force", picker_registry[name], opts)
  end

  local picker_config = picker_registry[name]
  local source_name = "neural_open_" .. name

  -- Build and register the Snacks source
  local snacks = require("snacks")
  snacks.picker.sources = snacks.picker.sources or {}

  if picker_config.type == "file" then
    snacks.picker.sources[source_name] = build_file_source_config(name, picker_config)
  else
    snacks.picker.sources[source_name] = build_item_source_config(name, picker_config)
  end

  -- Open the picker
  snacks.picker.pick(source_name, { title = picker_config.title })
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
  elseif subcommand == "pick" then
    local picker_name = args.fargs[2]
    if picker_name then
      M.pick(picker_name)
    else
      vim.notify("Usage: NeuralOpen pick <picker_name>", vim.log.levels.ERROR)
    end
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
    local subcommands = { "algorithm", "pick", "reset" }
    return vim.tbl_filter(function(s)
      return s:find(arg_lead, 1, true) == 1
    end, subcommands)
  elseif args[2] == "pick" then
    local names = vim.tbl_keys(picker_registry)
    table.sort(names)
    return vim.tbl_filter(function(s)
      return s:find(arg_lead, 1, true) == 1
    end, names)
  elseif args[2] == "algorithm" or args[2] == "reset" then
    -- Complete algorithm names
    return vim.tbl_filter(function(s)
      return s:find(arg_lead, 1, true) == 1
    end, valid_algorithms)
  end
  return {}
end

return M
