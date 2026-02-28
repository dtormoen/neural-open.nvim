--- Item picker source module.
--- Handles context capture and per-item transform for non-file item pickers.
--- Parallel to source.lua but for generic items with 7 features instead of 11.
local M = {}

--- Capture context for an item picker session.
--- Called once at the beginning of a picker session before any async operations.
---@param picker_name string Name of the picker (for weight isolation)
---@param ctx table The Snacks picker context
---@param config NosConfig Plugin configuration
function M.capture_context(picker_name, ctx, config)
  local cwd = vim.fn.getcwd()

  -- Load tracking data for this picker
  local item_tracking = require("neural-open.item_tracking")
  item_tracking.init(picker_name)
  local tracking_data = item_tracking.get_tracking_data(picker_name, cwd)

  -- Setup the algorithm for this session with item-specific feature names
  local registry = require("neural-open.algorithms.registry")
  local item_scorer = require("neural-open.item_scorer")
  local algorithm = registry.get_algorithm_for_picker(
    config.algorithm,
    config.item_algorithm_config or config.algorithm_config,
    picker_name,
    { feature_names = item_scorer.ITEM_FEATURE_NAMES }
  )
  algorithm.load_weights()

  -- Store all context in a single field
  ctx.meta = ctx.meta or {}
  ctx.meta.nos_ctx = {
    cwd = cwd,
    algorithm = algorithm,
    tracking_data = tracking_data,
    picker_name = picker_name,
  }
end

--- Create a transform function that computes per-item data once.
--- Called once per item when it's first discovered.
---@param picker_name string Name of the picker
---@param config NosConfig Plugin configuration
---@param item_scorer table The item_scorer module
---@return function Transform function for Snacks picker
function M.create_item_transform(picker_name, config, item_scorer)
  local recency_list_size = config.recency_list_size or 100

  return function(item, ctx)
    if not item.text then
      return item
    end

    -- Determine item identity
    local item_id = item.value or item.text
    if not item_id or item_id == "" then
      return item
    end

    -- Deduplicate by item identity
    ctx.meta.done = ctx.meta.done or {} ---@type table<string, boolean>
    if ctx.meta.done[item_id] then
      return false
    end
    ctx.meta.done[item_id] = true

    -- Get safely captured context
    local nos_ctx = ctx.meta.nos_ctx or {}
    local tracking_data = nos_ctx.tracking_data or {}

    -- Compute static raw features
    local frecency = tracking_data.frecency and tracking_data.frecency[item_id] or 0
    local cwd_frecency = tracking_data.cwd_frecency and tracking_data.cwd_frecency[item_id] or 0
    local recency_rank = tracking_data.recency_rank and tracking_data.recency_rank[item_id]
    local cwd_recency_rank = tracking_data.cwd_recency_rank and tracking_data.cwd_recency_rank[item_id]
    local last_selected = tracking_data.last_selected
    local text_len = #item.text

    local raw_features = {
      match = 0, -- Dynamic, updated per-keystroke in on_match_handler
      frecency = frecency,
      cwd_frecency = cwd_frecency,
      recency = recency_rank or 0,
      cwd_recency = cwd_recency_rank or 0,
      text_length_inv = text_len,
      not_last_selected = (item_id == last_selected) and 0 or 1,
    }

    -- Pre-allocate input_buf with normalized static features.
    -- Dynamic feature (match) left as 0, filled per-keystroke by on_match_handler.
    local recency_val = item_scorer.calculate_recency_score(recency_rank, recency_list_size)
    local cwd_recency_val = item_scorer.calculate_recency_score(cwd_recency_rank, recency_list_size)

    local input_buf = {
      0, -- [1] match (dynamic)
      item_scorer.normalize_item_frecency(frecency), -- [2] frecency
      item_scorer.normalize_item_frecency(cwd_frecency), -- [3] cwd_frecency
      recency_val, -- [4] recency
      cwd_recency_val, -- [5] cwd_recency
      item_scorer.normalize_text_length(text_len), -- [6] text_length_inv
      raw_features.not_last_selected, -- [7] not_last_selected (binary)
    }

    -- Attach nos field to item
    item.nos = {
      raw_features = raw_features,
      neural_score = 0,
      item_id = item_id,
      input_buf = input_buf,
      ctx = nos_ctx,
    }

    return item
  end
end

return M
