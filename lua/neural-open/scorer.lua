local M = {}

---@param path string
---@return string
local function normalize_path(path)
  path = path:gsub("\\", "/")
  path = path:gsub("/+", "/")
  return path
end

---@param path string
---@return string
local function get_directory(path)
  local dir = path:match("(.*/)")
  return dir or ""
end

---@param current_path string?
---@param target_path string
---@return number
local function calculate_proximity(current_path, target_path)
  if not current_path or current_path == "" then
    return 0
  end

  local current_dir = get_directory(normalize_path(current_path))
  local target_dir = get_directory(normalize_path(target_path))

  if current_dir == target_dir then
    return 1.0
  end

  local current_parts = vim.split(current_dir, "/", { trimempty = true })
  local target_parts = vim.split(target_dir, "/", { trimempty = true })

  local common_depth = 0
  local min_depth = math.min(#current_parts, #target_parts)

  for i = 1, min_depth do
    if current_parts[i] == target_parts[i] then
      common_depth = i
    else
      break
    end
  end

  if common_depth == 0 then
    return 0
  end

  local total_depth = math.max(#current_parts, #target_parts)
  return common_depth / total_depth
end

--- Get virtual name for a file, handling special files like index.js
---@param path string The file path
---@param special_files table<string, boolean>? Table of special filenames
---@return string Virtual name for display/matching
function M.get_virtual_name(path, special_files)
  local filename = vim.fn.fnamemodify(path, ":t")
  local parent_dir = vim.fn.fnamemodify(path, ":h:t")

  if special_files and special_files[filename] and parent_dir and parent_dir ~= "" then
    return parent_dir .. "/" .. filename
  end

  return filename
end

---@param recent_rank number?
---@param max_items number?
---@return number
function M.calculate_recency_score(recent_rank, max_items)
  if not recent_rank or recent_rank <= 0 then
    return 0
  end
  max_items = max_items or require("neural-open").config.recency_list_size or 100
  if recent_rank > max_items then
    return 0
  end
  return (max_items - recent_rank + 1) / max_items
end

--- Compute static raw features that don't depend on the search query
--- These are computed once per item during the transform phase
---@param normalized_path string The normalized absolute path
---@param context NosContext The shared session context
---@param item_data table Additional item-specific data
---@return NosRawFeatures
function M.compute_static_raw_features(normalized_path, context, item_data)
  local raw_features = {
    match = 0, -- Will be set in on_match_handler
    virtual_name = 0, -- Will be set in on_match_handler
    frecency = 0, -- Will be set in on_match_handler from Snacks
    open = item_data.is_open_buffer and 1 or 0,
    alt = item_data.is_alternate and 1 or 0,
    proximity = 0,
    project = 0,
    recency = item_data.recent_rank or 0,
    trigram = 0,
    transition = 0,
  }

  -- Calculate proximity
  if context.current_file and context.current_file ~= "" then
    raw_features.proximity = calculate_proximity(context.current_file, normalized_path)
  end

  -- Check if in project
  if context.cwd then
    local normalized_file = normalize_path(normalized_path)
    local normalized_cwd = normalize_path(context.cwd)
    if normalized_file:sub(1, #normalized_cwd) == normalized_cwd then
      raw_features.project = 1
    end
  end

  -- Calculate trigram similarity if current file trigrams are available
  if context.current_file_trigrams and item_data.virtual_name then
    local trigrams = require("neural-open.trigrams")

    -- Compute target file's trigrams
    local target_trigrams = trigrams.compute_trigrams(item_data.virtual_name)

    -- Calculate Dice coefficient
    raw_features.trigram = trigrams.dice_coefficient(context.current_file_trigrams, target_trigrams)
  end

  -- Lookup precomputed transition score
  if context.transition_scores then
    raw_features.transition = context.transition_scores[normalized_path] or 0
  end

  return raw_features
end

--- Normalize all raw features to [0,1] range
---@param raw_features NosRawFeatures
---@return NosNormalizedFeatures
function M.normalize_features(raw_features)
  local normalized = {}

  -- Match score normalization using sigmoid
  normalized.match = M.normalize_match_score(raw_features.match)

  -- Virtual name normalization using sigmoid
  normalized.virtual_name = M.normalize_virtual_name_score(raw_features.virtual_name)

  -- Frecency normalization: 1 - 1/(1+x)
  normalized.frecency = raw_features.frecency > 0 and (1 - 1 / (1 + raw_features.frecency / 8)) or 0

  -- Binary features are already normalized
  normalized.open = raw_features.open or 0
  normalized.alt = raw_features.alt or 0

  -- Proximity is already 0-1 from calculation
  normalized.proximity = raw_features.proximity or 0

  -- Project is binary
  normalized.project = raw_features.project or 0

  -- Recency normalization: linear decay (max - rank + 1) / max
  if raw_features.recency and raw_features.recency > 0 then
    normalized.recency = M.calculate_recency_score(raw_features.recency)
  else
    normalized.recency = 0
  end

  -- Trigram is already normalized (Dice coefficient is 0-1)
  normalized.trigram = raw_features.trigram or 0

  -- Transition is already normalized (1-1/(1+count) formula produces [0,1])
  normalized.transition = raw_features.transition or 0

  return normalized
end

--- Normalize a match score to [0,1] using sigmoid
---@param score number?
---@return number
function M.normalize_match_score(score)
  if not score or score <= 0 then
    return 0
  end
  -- Snacks match scores typically range from ~10-200 for good matches
  -- We use a sigmoid similar to smart-open.nvim but adapted for positive scores
  -- Scale factor 0.02 works well for the 10-200 range
  return 1 / (1 + math.exp(-0.02 * score + 2))
end

--- Normalize a virtual name score to [0,1] using sigmoid
---@param score number?
---@return number
function M.normalize_virtual_name_score(score)
  return M.normalize_match_score(score)
end

--- Handle match scoring for an item during search
--- This is called each time the search query changes
---@param matcher table The Snacks matcher instance
---@param item NeuralOpenItem The item to score
function M.on_match_handler(matcher, item)
  if not item or not item.file or not item.nos then
    return
  end

  -- Ensure raw_features exists (should be initialized in transform)
  if not item.nos.raw_features then
    return
  end

  -- Get algorithm from context (already loaded in capture_context)
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

  -- Calculate virtual name score now that we have the query
  local raw_virtual_name_score = 0
  if current_query ~= "" and item.nos.virtual_name then
    -- Reuse the existing matcher that was passed in
    local mock_item = { text = item.nos.virtual_name, idx = 1, score = 0 }
    raw_virtual_name_score = matcher:match(mock_item) or 0
  end

  -- Calculate base match score
  local raw_match_score = 0
  if current_query ~= "" then
    -- Get the pure match score by calling the matcher directly
    -- This gives us the base score without any bonuses (frecency, cwd, etc.)
    -- We create a temporary item to avoid modifying the original
    local temp_item = { text = item.text or item.file or "", idx = 1, score = 0 }
    raw_match_score = matcher:match(temp_item) or 0
  end

  -- Update dynamic raw features
  item.nos.raw_features.match = raw_match_score
  item.nos.raw_features.virtual_name = raw_virtual_name_score

  -- Capture frecency from Snacks.nvim (it sets this during matching)
  local frecency_value = item.frecency or 0
  item.nos.raw_features.frecency = frecency_value

  -- Normalize all features at once
  item.nos.normalized_features = M.normalize_features(item.nos.raw_features)

  -- Calculate total weighted score using algorithm
  local total_weighted_score = algorithm.calculate_score(item.nos.normalized_features)

  -- Update neural score and final score
  item.nos.neural_score = total_weighted_score
  item.score = total_weighted_score
end

return M
