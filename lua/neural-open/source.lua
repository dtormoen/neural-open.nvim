local M = {}

-- Cached at module load: check once if vim.fs.normalize is available
local _has_fs_normalize = (function()
  local ok, result = pcall(function()
    return vim.fs ~= nil and vim.fs.normalize ~= nil
  end)
  return ok and result
end)()
local _normalize_opts = { expand_env = false }
local _is_windows = vim.fn.has("win32") == 1

--- Capture global context that is shared across all items in the session
--- This is called once at the beginning of a file picking session
---@param ctx table The Snacks picker context
function M.capture_context(ctx)
  -- Capture buffer context safely here (before async operations)
  local recent = require("neural-open.recent")
  local recent_files = recent.get_recency_map()
  local alternate_buf = vim.fn.bufnr("#")

  -- Capture current working directory
  local cwd = vim.fn.getcwd()

  -- Capture current file from current buffer
  local current_file = ""
  local current_buf = vim.api.nvim_get_current_buf()
  if current_buf and vim.api.nvim_buf_is_valid(current_buf) then
    local buf_name = vim.api.nvim_buf_get_name(current_buf)
    if buf_name and buf_name ~= "" then
      current_file = vim.fn.fnamemodify(buf_name, ":p")
    end
  end

  -- Precompute per-session current file data (trigrams, directory info)
  local current_file_trigrams = nil
  local current_file_virtual_name = ""
  local current_file_dir, current_file_depth = nil, 0
  local config = require("neural-open").config
  local scorer = require("neural-open.scorer")
  scorer.set_recency_list_size(config.recency_list_size)
  if current_file ~= "" then
    local trigrams_mod = require("neural-open.trigrams")
    current_file_virtual_name = scorer.get_virtual_name(current_file, config.special_files)
    current_file_trigrams = trigrams_mod.compute_trigrams(current_file_virtual_name)
    current_file_dir, current_file_depth = scorer.compute_dir_info(current_file)
  end

  local current_file_trigrams_size = 0
  if current_file_trigrams then
    current_file_trigrams_size = require("neural-open.trigrams").count_trigrams(current_file_trigrams)
  end

  -- Setup the algorithm once for the session
  local registry = require("neural-open.algorithms.registry")

  -- Get the algorithm from config (guaranteed to return a valid algorithm)
  local algorithm = registry.get_algorithm()

  -- Load the latest weights for this algorithm
  algorithm.load_weights()

  -- Precompute transition scores for all potential destinations
  local transition_scores = nil
  if current_file and current_file ~= "" then
    local transitions = require("neural-open.transitions")
    transition_scores = transitions.compute_scores_from(current_file)
  end

  -- Store all neural-open context in a single field
  ctx.meta.nos_ctx = {
    recent_files = recent_files,
    alternate_buf = alternate_buf,
    cwd = cwd,
    current_file = current_file,
    current_file_dir = current_file_dir,
    current_file_depth = current_file_depth,
    current_file_trigrams = current_file_trigrams,
    current_file_trigrams_size = current_file_trigrams_size,
    current_file_virtual_name = current_file_virtual_name,
    -- Store algorithm for this session
    algorithm = algorithm,
    transition_scores = transition_scores,
  }
end

--- Create a transform function that computes per-item data once
--- This is called once per item when it's first discovered
---@param config table Plugin configuration
---@param scorer table Scorer module
---@param opts table? Additional options
---@return function Transform function for Snacks picker
function M.create_neural_transform(config, scorer, opts)
  return function(item, ctx)
    if not item.file then
      return item
    end

    -- Normalize the path to ensure consistent deduplication
    local path = item.file

    -- Check if path is already absolute
    local is_absolute = vim.startswith(path, "/") or vim.startswith(path, "~") or (_is_windows and path:match("^%a:"))

    -- Only join with cwd if file is relative and cwd is provided
    if item.cwd and not is_absolute then
      path = item.cwd .. "/" .. path
    end

    -- Normalize the path (availability check and opts table cached at module load)
    local normalized_path
    if _has_fs_normalize then
      normalized_path = vim.fs.normalize(path, _normalize_opts)
    else
      normalized_path = vim.fn.fnamemodify(path, ":p")
    end

    -- Set item._path to our normalized absolute path.
    -- This is the cache field that Snacks.picker.util.path() checks first,
    -- so it will use our normalized path instead of concatenating item.cwd + item.file.
    -- We intentionally don't modify item.file or item.cwd to preserve the original
    -- source data for display formatting.
    item._path = normalized_path

    -- Apply unique filter to deduplicate files
    ctx.meta.done = ctx.meta.done or {} ---@type table<string, boolean>
    if ctx.meta.done[normalized_path] then
      return false
    end
    ctx.meta.done[normalized_path] = true

    -- Get safely captured context from finder (no vim API calls in async context)
    local nos_ctx = ctx.meta.nos_ctx or {}
    local recent_files = nos_ctx.recent_files or {}
    local alternate_buf = nos_ctx.alternate_buf

    local is_open_buffer = item.buf ~= nil
    local is_alternate = item.buf ~= nil and item.buf == alternate_buf

    -- Get recent file info
    local recent_rank = nil
    if recent_files[normalized_path] then
      recent_rank = recent_files[normalized_path].recent_rank
    end

    -- Compute virtual name for special files
    local virtual_name = scorer.get_virtual_name(normalized_path, config.special_files)

    -- Compute static raw features once during transform
    local raw_features = scorer.compute_static_raw_features(
      normalized_path,
      nos_ctx,
      is_open_buffer,
      is_alternate,
      recent_rank,
      virtual_name
    )

    -- Pre-allocate input_buf with normalized static features for all algorithms.
    -- Dynamic features (match, virtual_name, frecency) are filled per-keystroke in on_match_handler.
    local recency_val = 0
    if recent_rank and recent_rank > 0 then
      recency_val = scorer.calculate_recency_score(recent_rank)
    end
    local input_buf = {
      0, -- [1] match (dynamic)
      0, -- [2] virtual_name (dynamic)
      0, -- [3] frecency (dynamic)
      is_open_buffer and 1 or 0, -- [4] open
      is_alternate and 1 or 0, -- [5] alt
      raw_features.proximity, -- [6] proximity (already [0,1])
      raw_features.project, -- [7] project (already 0/1)
      recency_val, -- [8] recency (normalized)
      raw_features.trigram, -- [9] trigram (already [0,1])
      raw_features.transition, -- [10] transition (already [0,1])
    }

    -- Initialize the nos field structure with all per-item data
    item.nos = {
      -- Path data
      normalized_path = normalized_path,
      virtual_name = virtual_name,

      -- Buffer status
      is_open_buffer = is_open_buffer,
      is_alternate = is_alternate,

      -- Recent file data
      recent_rank = recent_rank,

      -- Pre-allocated flat input buffer for all algorithms
      input_buf = input_buf,

      -- Reference to shared context
      ctx = nos_ctx,

      -- Raw features (will be updated per-keystroke for dynamic features)
      raw_features = raw_features,
      neural_score = 0,
    }

    return item
  end
end

return M
