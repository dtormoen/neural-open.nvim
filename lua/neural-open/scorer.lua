local M = {}

local math_exp = math.exp
local trigrams = require("neural-open.trigrams")

--- Canonical feature names in input buffer order (shared by all algorithms)
M.FEATURE_NAMES =
  { "match", "virtual_name", "frecency", "open", "alt", "proximity", "project", "recency", "trigram", "transition" }

--- Convert a flat input buffer to a named features table
---@param input_buf number[] Flat array of normalized features in FEATURE_NAMES order
---@return table<string, number>
function M.input_buf_to_features(input_buf)
  local features = {}
  for i, name in ipairs(M.FEATURE_NAMES) do
    features[name] = input_buf[i]
  end
  return features
end

-- Reusable temp items for matcher calls in on_match_handler (avoids 2 allocations per item per keystroke)
local _mock_item = { text = "", idx = 1, score = 0 }
local _temp_item = { text = "", idx = 1, score = 0 }

--- Compute directory path and depth from a file path.
--- Returns the directory (up to and including last /) and the segment count.
---@param file_path string
---@return string? dir Directory portion, nil if no directory
---@return number depth Number of directory segments (slash count excluding leading /)
function M.compute_dir_info(file_path)
  if not file_path or file_path == "" then
    return nil, 0
  end
  local dir = file_path:match("(.*/)")
  if not dir then
    return nil, 0
  end
  local depth = 0
  for i = 2, #dir do
    if dir:byte(i) == 0x2F then
      depth = depth + 1
    end
  end
  return dir, depth
end

--- Calculate directory proximity between current file and a target path.
--- Uses zero-allocation character scanning instead of vim.split.
---@param current_dir string Precomputed directory of current file (up to and including last /)
---@param current_depth number Precomputed depth of current directory (slash count excluding leading /)
---@param target_path string The target file path
---@return number Proximity score in [0, 1]
local function calculate_proximity(current_dir, current_depth, target_path)
  if not current_dir then
    return 0
  end

  -- Find end of target directory (last slash position)
  local last_slash = 0
  for i = #target_path, 1, -1 do
    if target_path:byte(i) == 0x2F then
      last_slash = i
      break
    end
  end
  if last_slash == 0 then
    return 0
  end

  -- Quick exact-match check: compare directory portions directly
  local current_dir_len = #current_dir
  if current_dir_len == last_slash and target_path:sub(1, last_slash) == current_dir then
    return 1.0
  end

  -- Root-only paths have no segments to compare beyond the exact match above
  if last_slash <= 1 then
    return 0
  end

  -- Scan for common prefix, counting complete matching directory segments.
  -- Start at position 2 to skip the leading '/' (root prefix, not a segment).
  local common_depth = 0
  local scan_len = math.min(current_dir_len, last_slash)
  for i = 2, scan_len do
    if current_dir:byte(i) ~= target_path:byte(i) then
      break
    end
    if current_dir:byte(i) == 0x2F then
      common_depth = common_depth + 1
    end
  end

  if common_depth == 0 then
    return 0
  end

  -- Count target directory depth (number of slashes excluding leading)
  local target_depth = 0
  for i = 2, last_slash do
    if target_path:byte(i) == 0x2F then
      target_depth = target_depth + 1
    end
  end

  return common_depth / math.max(current_depth, target_depth)
end

--- Get virtual name for a file, handling special files like index.js
---@param path string The file path
---@param special_files table<string, boolean>? Table of special filenames
---@return string Virtual name for display/matching
function M.get_virtual_name(path, special_files)
  local filename = path:match("[^/]+$") or path
  if not special_files or not special_files[filename] then
    return filename
  end
  local parent = path:match("([^/]+)/[^/]+$")
  if parent and parent ~= "" then
    return parent .. "/" .. filename
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

  -- Calculate proximity using precomputed directory context (fast path)
  -- Falls back to deriving from current_file when precomputed values aren't available
  local current_dir = context.current_file_dir
  local current_depth = context.current_file_depth
  if not current_dir and context.current_file and context.current_file ~= "" then
    current_dir, current_depth = M.compute_dir_info(context.current_file)
  end
  if current_dir then
    raw_features.proximity = calculate_proximity(current_dir, current_depth, normalized_path)
  end

  -- Check if in project
  if context.cwd and normalized_path:find(context.cwd, 1, true) == 1 then
    raw_features.project = 1
  end

  -- Calculate trigram similarity if current file trigrams are available
  if context.current_file_trigrams and item_data.virtual_name then
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

--- Normalize a match or virtual_name score to [0,1] using sigmoid
---@param raw_score number Raw fuzzy match score (typically 0-200+)
---@return number Normalized value in [0,1]
function M.normalize_match_score(raw_score)
  return (raw_score and raw_score > 0) and (1 / (1 + math_exp(-0.02 * raw_score + 2))) or 0
end

--- Normalize a frecency value to [0,1]
---@param raw_frecency number Raw frecency value (0-∞)
---@return number Normalized value in [0,1]
function M.normalize_frecency(raw_frecency)
  return (raw_frecency and raw_frecency > 0) and (1 - 1 / (1 + raw_frecency / 8)) or 0
end

--- Normalize all raw features to [0,1] range
---@param raw_features NosRawFeatures
---@return NosNormalizedFeatures
function M.normalize_features(raw_features)
  local recency_val = 0
  if raw_features.recency and raw_features.recency > 0 then
    recency_val = M.calculate_recency_score(raw_features.recency)
  end

  return {
    match = M.normalize_match_score(raw_features.match),
    virtual_name = M.normalize_match_score(raw_features.virtual_name),
    frecency = M.normalize_frecency(raw_features.frecency),
    open = raw_features.open or 0,
    alt = raw_features.alt or 0,
    proximity = raw_features.proximity or 0,
    project = raw_features.project or 0,
    recency = recency_val,
    trigram = raw_features.trigram or 0,
    transition = raw_features.transition or 0,
  }
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
    _mock_item.text = item.nos.virtual_name
    _mock_item.score = 0
    raw_virtual_name_score = matcher:match(_mock_item) or 0
  end

  -- Calculate base match score
  local raw_match_score = 0
  if current_query ~= "" then
    _temp_item.text = item.text or item.file or ""
    _temp_item.score = 0
    raw_match_score = matcher:match(_temp_item) or 0
  end

  -- Update dynamic raw features
  item.nos.raw_features.match = raw_match_score
  item.nos.raw_features.virtual_name = raw_virtual_name_score

  -- Capture frecency from Snacks.nvim (it sets this during matching)
  local frecency_value = item.frecency or 0
  item.nos.raw_features.frecency = frecency_value

  -- Update pre-allocated input_buf with dynamic features and score
  -- (zero table allocation per keystroke for all algorithms)
  local input_buf = item.nos.input_buf
  input_buf[1] = M.normalize_match_score(raw_match_score)
  input_buf[2] = M.normalize_match_score(raw_virtual_name_score)
  input_buf[3] = M.normalize_frecency(frecency_value)
  local total_weighted_score = algorithm.calculate_score(input_buf)
  item.nos.neural_score = total_weighted_score
  item.score = total_weighted_score
end

return M
