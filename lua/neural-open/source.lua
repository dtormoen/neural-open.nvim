local M = {}

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

  -- Compute trigrams for current file's virtual name if available
  local current_file_trigrams = nil
  local current_file_virtual_name = ""
  if current_file and current_file ~= "" then
    local scorer = require("neural-open.scorer")
    local trigrams = require("neural-open.trigrams")
    local config = require("neural-open").config
    current_file_virtual_name = scorer.get_virtual_name(current_file, config.special_files)
    current_file_trigrams = trigrams.compute_trigrams(current_file_virtual_name)
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
    current_file_trigrams = current_file_trigrams,
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
    local is_absolute = vim.startswith(path, "/")
      or vim.startswith(path, "~")
      or (vim.fn.has("win32") == 1 and path:match("^%a:"))

    -- Only join with cwd if file is relative and cwd is provided
    if item.cwd and not is_absolute then
      path = item.cwd .. "/" .. path
    end

    -- Normalize the path
    local normalized_path
    -- Use pcall to safely check for vim.fs.normalize
    local has_normalize = false
    local ok = pcall(function()
      has_normalize = vim.fs ~= nil and vim.fs.normalize ~= nil
    end)

    if ok and has_normalize then
      normalized_path = vim.fs.normalize(path, { expand_env = false })
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

      -- Reference to shared context
      ctx = nos_ctx,

      -- Initialize feature structures (will be populated later)
      raw_features = scorer.compute_static_raw_features(normalized_path, nos_ctx, {
        is_open_buffer = is_open_buffer,
        is_alternate = is_alternate,
        recent_rank = recent_rank,
        virtual_name = virtual_name,
      }),
      normalized_features = {},
      neural_score = 0,
    }

    return item
  end
end

return M
