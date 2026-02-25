--- Classic scoring algorithm with weighted features and self-learning
local M = {}

local config = {}
local current_weights = nil

--- Get default weights for display purposes
---@return table
local function get_default_weights()
  return require("neural-open.weights").get_default_weights("classic")
end

--- Ensure weights are loaded and available
---@return table
---@param force_reload boolean? Force reload weights even if already loaded
local function ensure_weights(force_reload)
  if not current_weights or force_reload then
    current_weights = require("neural-open.weights").get_weights("classic")
  end
  return current_weights
end

--- Initialize the algorithm with configuration
---@param algorithm_config NosClassicConfig
function M.init(algorithm_config)
  config = algorithm_config
end

--- Calculate weighted score from normalized features
---@param normalized_features table<string, number>
---@return number
function M.calculate_score(normalized_features)
  local weights = ensure_weights()

  local score = 0
  for feature_name, normalized_value in pairs(normalized_features) do
    if weights[feature_name] then
      score = score + (normalized_value * weights[feature_name])
    end
  end

  return score
end

--- Calculate weighted component scores from normalized features
---@param normalized_features table<string, number>
---@return table<string, number>
local function calculate_components(normalized_features)
  local weights = ensure_weights()

  local components = {}
  for feature_name, normalized_value in pairs(normalized_features) do
    if weights[feature_name] then
      components[feature_name] = normalized_value * weights[feature_name]
    end
  end

  return components
end

--- Calculate weight adjustments based on component differences
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@return table adjustments, number num_higher_items
local function calculate_adjustments(selected_item, ranked_items)
  local selected_rank = selected_item.neural_rank

  -- No adjustment needed if item is rank 1 or has no rank
  if not selected_rank or selected_rank == 1 then
    return {}, 0
  end

  local num_higher_items = selected_rank - 1
  if num_higher_items == 0 then
    return {}, 0
  end

  -- Get current weights to initialize adjustments
  local weights = ensure_weights()

  -- Initialize adjustments
  local adjustments = {}
  for key, _ in pairs(weights) do
    adjustments[key] = 0
  end

  -- Calculate components from normalized features
  local selected_components = {}
  if selected_item.nos and selected_item.nos.normalized_features then
    selected_components = calculate_components(selected_item.nos.normalized_features)
  end

  -- Compare with all higher-ranked items
  for i = 1, selected_rank - 1 do
    local higher_item = ranked_items[i]
    if higher_item then
      -- Calculate components from normalized features
      local higher_components = {}
      if higher_item.nos and higher_item.nos.normalized_features then
        higher_components = calculate_components(higher_item.nos.normalized_features)
      end

      -- Check where selected item scored better
      for key, value in pairs(selected_components) do
        if value and value > 0 then
          local higher_value = higher_components[key] or 0
          if value > higher_value then
            if adjustments[key] ~= nil then
              adjustments[key] = adjustments[key] + 1
            end
          end
        end
      end

      -- Check where higher items scored better
      for key, value in pairs(higher_components) do
        if value and value > 0 then
          local selected_value = selected_components[key] or 0
          if value > selected_value then
            if adjustments[key] ~= nil then
              adjustments[key] = adjustments[key] - 1
            end
          end
        end
      end
    end
  end

  -- Apply normalization with learning rate
  local learning_rate = config.learning_rate or 0.6
  for key, adj in pairs(adjustments) do
    if adj ~= 0 then
      adjustments[key] = (adj / num_higher_items) * learning_rate
    end
  end

  return adjustments, num_higher_items
end

--- Apply adjustments to weights and calculate changes
---@param adjustments table
---@param apply boolean Whether to actually apply the changes
---@return table new_weights, table changes, boolean has_changes
local function apply_adjustments(adjustments, apply)
  local weights = ensure_weights()

  local new_weights = {}
  local changes = {}
  local has_changes = false

  -- Copy all existing weights first
  for key, value in pairs(weights) do
    new_weights[key] = value
  end

  -- Apply adjustments
  for key, adj in pairs(adjustments) do
    if weights[key] then
      local old_weight = weights[key]
      local new_weight = math.max(1, math.min(200, old_weight + adj))
      new_weights[key] = new_weight

      if math.abs(new_weight - old_weight) > 0.01 then
        has_changes = true
        changes[key] = {
          old = old_weight,
          new = new_weight,
          delta = new_weight - old_weight,
        }
      end
    end
  end

  if has_changes and apply then
    -- Update internal state
    current_weights = new_weights

    -- Format changes for notification
    local formatted_changes = {}
    local default_weights = get_default_weights()
    for key, change in pairs(changes) do
      local default_val = default_weights[key] or 0
      formatted_changes[key] = string.format("%.2f → %.2f (default: %.2f)", change.old, change.new, default_val)
    end

    if not vim.tbl_isempty(formatted_changes) then
      vim.notify("Neural-open weights updated: " .. vim.inspect(formatted_changes), vim.log.levels.DEBUG)
    end
  end

  return new_weights, changes, has_changes
end

--- Update weights based on user selection (with optional latency tracking)
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@param latency_ctx? table Optional latency context (for consistency with other algorithms)
function M.update_weights(selected_item, ranked_items, latency_ctx)
  local adjustments, _ = calculate_adjustments(selected_item, ranked_items)
  local new_weights, _, has_changes = apply_adjustments(adjustments, true)

  -- Save updated weights if changed
  if has_changes then
    local weights_module = require("neural-open.weights")
    weights_module.save_weights("classic", new_weights, latency_ctx)
  end
end

--- Simulate weight adjustments without applying them
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@return table?
function M.simulate_weight_adjustments(selected_item, ranked_items)
  local adjustments, num_higher_items = calculate_adjustments(selected_item, ranked_items)

  if num_higher_items == 0 then
    return nil
  end

  local new_weights, changes, has_changes = apply_adjustments(adjustments, false) -- Don't apply

  if not has_changes then
    return nil
  end

  return {
    changes = changes,
    new_weights = new_weights,
    adjustments = adjustments,
    compared_with = num_higher_items,
  }
end

--- Generate debug view for classic algorithm
---@param item NeuralOpenItem
---@param all_items NeuralOpenItem[]?
---@return string[]
function M.debug_view(item, all_items)
  local lines = {}
  local weights = ensure_weights()

  table.insert(lines, "⚖️ Classic Algorithm")
  table.insert(lines, "")
  table.insert(lines, "Algorithm: Weighted sum with self-learning")
  table.insert(lines, string.format("Learning Rate: %.2f", config.learning_rate or 0.6))
  table.insert(lines, "")

  if item.nos then
    -- Score totals
    table.insert(lines, string.format("🎯 Total Neural Score: %.2f", item.nos.neural_score or 0))
    if item.score then
      table.insert(lines, string.format("📋 Final Snacks Score: %.2f", item.score))
    end
    table.insert(lines, "")

    -- Combined Raw and Normalized Features
    table.insert(lines, "🔢 Features (Raw → Normalized):")
    table.insert(lines, "")

    local all_features =
      { "match", "virtual_name", "open", "alt", "proximity", "project", "frecency", "recency", "trigram", "transition" }

    for _, name in ipairs(all_features) do
      local raw_value = (item.nos.raw_features and item.nos.raw_features[name]) or 0
      local normalized_value = (item.nos.normalized_features and item.nos.normalized_features[name]) or 0

      local formatted_name = name:gsub("_", " "):gsub("(%l)(%u)", "%1 %2")
      formatted_name = formatted_name:sub(1, 1):upper() .. formatted_name:sub(2)

      table.insert(lines, string.format("  %-15s %8.2f → %6.4f", formatted_name .. ":", raw_value, normalized_value))
    end
    table.insert(lines, "")

    -- Weighted Components
    table.insert(lines, "⚖️ Weighted Components (Normalized × Weight = Score):")
    table.insert(lines, "")

    -- Calculate components on-the-fly
    local components = {}
    if item.nos.normalized_features then
      components = calculate_components(item.nos.normalized_features)
    end

    -- Create sorted list
    local sorted_components = {}
    for _, name in ipairs(all_features) do
      local value = components[name] or 0
      table.insert(sorted_components, { name = name, value = value })
    end

    table.sort(sorted_components, function(a, b)
      return a.value > b.value
    end)

    for _, comp in ipairs(sorted_components) do
      local name = comp.name
      local value = comp.value
      local formatted_name = name:gsub("_", " "):gsub("(%l)(%u)", "%1 %2")
      formatted_name = formatted_name:sub(1, 1):upper() .. formatted_name:sub(2)

      local normalized = item.nos.normalized_features and item.nos.normalized_features[name] or 0
      local weight = weights[name] or 0
      local default_weights = get_default_weights()
      local default_weight = default_weights[name] or 0

      table.insert(
        lines,
        string.format(
          "  %-15s %6.4f × %6.1f (%6.1f) = %8.2f",
          formatted_name .. ":",
          normalized,
          weight,
          default_weight,
          value
        )
      )
    end
    table.insert(lines, "")

    -- Weight adjustment preview
    if all_items then
      table.insert(lines, "📈 Potential Weight Adjustments (if selected):")
      table.insert(lines, "")

      -- Find current item's rank
      local current_rank = nil
      for i, ranked_item in ipairs(all_items) do
        if ranked_item.file == item.file then
          current_rank = i
          item.neural_rank = i -- Ensure item has the rank
          break
        end
      end

      if current_rank and current_rank > 1 then
        -- Simulate weight adjustments
        local simulation = M.simulate_weight_adjustments(item, all_items)

        if simulation and simulation.changes then
          table.insert(
            lines,
            string.format("  Rank: #%d (comparing with %d higher items)", current_rank, simulation.compared_with)
          )
          table.insert(lines, "")

          -- Sort changes by delta magnitude
          local sorted_changes = {}
          for key, change in pairs(simulation.changes) do
            table.insert(sorted_changes, { key = key, change = change })
          end
          table.sort(sorted_changes, function(a, b)
            return math.abs(a.change.delta) > math.abs(b.change.delta)
          end)

          for _, entry in ipairs(sorted_changes) do
            local key = entry.key
            local change = entry.change
            local formatted_key = key:gsub("_", " "):gsub("(%l)(%u)", "%1 %2")
            formatted_key = formatted_key:sub(1, 1):upper() .. formatted_key:sub(2)

            local arrow = change.delta > 0 and "↑" or "↓"
            local color_indicator = change.delta > 0 and "+" or ""

            table.insert(
              lines,
              string.format(
                "  %-15s %s %6.2f → %6.2f (%s%.2f)",
                formatted_key .. ":",
                arrow,
                change.old,
                change.new,
                color_indicator,
                change.delta
              )
            )
          end
        else
          table.insert(lines, "  No adjustments needed (already rank #1)")
        end
      else
        table.insert(lines, "  No adjustments needed (rank #1 or unranked)")
      end
    end
  end

  return lines
end

--- Get algorithm name
---@return AlgorithmName
function M.get_name()
  return "classic"
end

--- Load the latest weights from the weights module
function M.load_weights()
  ensure_weights(true)
end

return M
