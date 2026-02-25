--- Naive scoring algorithm - simple sum of normalized features
local M = {}

local scorer = require("neural-open.scorer")

--- Calculate score by summing all elements in the input buffer
---@param input_buf number[] Flat array of normalized features in FEATURE_NAMES order
---@return number
function M.calculate_score(input_buf)
  local score = 0
  for i = 1, #input_buf do
    local v = input_buf[i]
    if v > 0 then
      score = score + v
    end
  end
  return score
end

--- Update weights (no-op for naive algorithm, accepts latency_ctx for API consistency)
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@param latency_ctx? table Optional latency context (unused)
function M.update_weights(selected_item, ranked_items, latency_ctx)
  -- Naive algorithm doesn't learn from selections
end

local fmt = require("neural-open.debug_fmt")

--- Generate debug view for naive algorithm
---@param item NeuralOpenItem
---@param all_items NeuralOpenItem[]?
---@return string[], table[]?
function M.debug_view(item, all_items)
  local lines = {}
  local hl = {}

  fmt.add_title(lines, hl, "Naive Algorithm")
  table.insert(lines, "")
  fmt.add_label(lines, hl, "Algorithm", "Simple sum of all normalized features")
  fmt.add_label(lines, hl, "Learning", "None (static algorithm)")
  table.insert(lines, "")

  if item.nos and item.nos.neural_score then
    fmt.add_label(lines, hl, "Total Score", string.format("%.4f", item.nos.neural_score))
  end

  if item.nos and item.nos.input_buf then
    local normalized_features = scorer.input_buf_to_features(item.nos.input_buf)
    fmt.append_feature_value_table(lines, hl, normalized_features)
  end

  return lines, hl
end

--- Get algorithm name
---@return AlgorithmName
function M.get_name()
  return "naive"
end

--- Initialize algorithm (no-op for naive)
---@param config NosNaiveConfig
function M.init(config)
  -- Naive algorithm has no configuration
end

--- Load the latest weights from the weights module
function M.load_weights()
  -- Naive algorithm doesn't use weights
end

return M
