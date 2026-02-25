--- Naive scoring algorithm - simple sum of normalized features
local M = {}

--- Calculate score by summing all normalized features
---@param normalized_features table<string, number>
---@return number
function M.calculate_score(normalized_features)
  local score = 0
  for _, value in pairs(normalized_features) do
    if value and value > 0 then
      score = score + value
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

--- Generate debug view for naive algorithm
---@param item NeuralOpenItem
---@param all_items NeuralOpenItem[]?
---@return string[]
function M.debug_view(item, all_items)
  local lines = {}

  table.insert(lines, "🎯 Naive Algorithm")
  table.insert(lines, "")
  table.insert(lines, "Algorithm: Simple sum of all normalized features")
  table.insert(lines, "Learning: None (static algorithm)")
  table.insert(lines, "")

  if item.nos and item.nos.neural_score then
    table.insert(lines, string.format("Total Score: %.4f", item.nos.neural_score))
    table.insert(lines, "")
  end

  if item.nos and item.nos.normalized_features then
    table.insert(lines, "Normalized Features (all weighted equally):")
    table.insert(lines, "")

    -- Sort features by value for better readability
    local sorted_features = {}
    for name, value in pairs(item.nos.normalized_features) do
      table.insert(sorted_features, { name = name, value = value })
    end
    table.sort(sorted_features, function(a, b)
      return a.value > b.value
    end)

    for _, feature in ipairs(sorted_features) do
      local formatted_name = feature.name:gsub("_", " "):gsub("(%l)(%u)", "%1 %2")
      formatted_name = formatted_name:sub(1, 1):upper() .. formatted_name:sub(2)
      table.insert(lines, string.format("  %-15s: %.4f", formatted_name, feature.value))
    end
  end

  return lines
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
