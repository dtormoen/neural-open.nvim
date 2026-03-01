--- Item picker scoring module.
--- Defines the 8-feature scoring pipeline for non-file item pickers.
--- Parallel to scorer.lua but for generic items (commands, recipes, etc.).
local M = {}

local math_exp = math.exp
local scorer = require("neural-open.scorer")

--- Canonical feature names in input buffer order for item pickers (8 features)
M.ITEM_FEATURE_NAMES = {
  "match",
  "frecency",
  "cwd_frecency",
  "recency",
  "cwd_recency",
  "text_length_inv",
  "not_last_selected",
  "transition",
}

--- Convert a flat input buffer to a named features table
---@param input_buf number[] Flat array of normalized features in ITEM_FEATURE_NAMES order
---@return table<string, number>
function M.input_buf_to_features(input_buf)
  local features = {}
  for i, name in ipairs(M.ITEM_FEATURE_NAMES) do
    features[name] = input_buf[i]
  end
  return features
end

--- Normalize a fuzzy match score to [0,1] using sigmoid (same formula as file scorer)
---@param raw_score number Raw fuzzy match score (typically 0-200+)
---@return number Normalized value in [0,1]
function M.normalize_match_score(raw_score)
  return (raw_score and raw_score > 0) and (1 / (1 + math_exp(-0.02 * raw_score + 2))) or 0
end

--- Normalize an item frecency score to [0,1].
--- Uses same 1-1/(1+x/8) formula as file frecency for consistent scaling.
---@param raw_frecency number Raw frecency score from item_tracking (0-∞)
---@return number Normalized value in [0,1]
function M.normalize_item_frecency(raw_frecency)
  return (raw_frecency and raw_frecency > 0) and (1 - 1 / (1 + raw_frecency / 8)) or 0
end

--- Calculate recency score with linear decay (delegates to scorer)
---@param recent_rank number? 1-based position in recency list
---@param max_items number? Maximum list size
---@return number Normalized value in [0,1]
function M.calculate_recency_score(recent_rank, max_items)
  return scorer.calculate_recency_score(recent_rank, max_items)
end

--- Normalize text length to [0,1] favoring shorter text.
--- Formula: 1/(1 + len*0.1) gives good spread for typical item text lengths.
---@param text_len number Length of item text
---@return number Normalized value in [0,1]
function M.normalize_text_length(text_len)
  return 1 / (1 + (text_len or 0) * 0.1)
end

-- Reusable temp item for matcher calls (avoids allocation per keystroke)
local _temp_item = { text = "", idx = 1, score = 0 }

--- Handle match scoring for an item during search.
--- This is called each time the search query changes (per-keystroke hot path).
--- Zero allocations: reuses _temp_item and updates input_buf in-place.
---@param matcher table The Snacks matcher instance
---@param item table The item to score (must have item.nos with input_buf and ctx)
function M.on_match_handler(matcher, item)
  if not item or not item.nos then
    return
  end

  local nos_ctx = item.nos.ctx
  if not nos_ctx or not nos_ctx.algorithm then
    return
  end

  local algorithm = nos_ctx.algorithm

  -- Get current query from matcher
  local current_query = ""
  if matcher.filter and matcher.filter.search then
    current_query = matcher.filter.search
  elseif matcher.pattern then
    current_query = matcher.pattern
  elseif matcher.query then
    current_query = matcher.query
  end

  -- Calculate fuzzy match score against item text
  local raw_match_score = 0
  if current_query ~= "" then
    _temp_item.text = item.text or ""
    _temp_item.score = 0
    raw_match_score = matcher:match(_temp_item) or 0
  end

  -- Update dynamic raw feature
  item.nos.raw_features.match = raw_match_score

  -- Update pre-allocated input_buf with dynamic match feature and score
  local input_buf = item.nos.input_buf
  input_buf[1] = M.normalize_match_score(raw_match_score)
  local total_weighted_score = algorithm.calculate_score(input_buf)
  item.nos.neural_score = total_weighted_score
  item.score = total_weighted_score
end

return M
