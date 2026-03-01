--- Classic scoring algorithm with weighted features and self-learning
local M = {}

local scorer = require("neural-open.scorer")

local states = {} -- picker_name -> state table

---@class ClassicState
---@field config NosClassicConfig
---@field current_weights table<string, number>?
---@field weight_buf number[]?

---@return ClassicState
local function new_state()
  return { config = nil, current_weights = nil, weight_buf = nil }
end

---@param cs ClassicState
---@param algorithm_config NosClassicConfig?
local function init_state(cs, algorithm_config)
  cs.config = vim.deepcopy(algorithm_config or {})
end

--- Get default weights for display purposes
---@param cs ClassicState
---@return table?
local function get_default_weights(cs)
  return require("neural-open.weights").get_default_weights("classic", cs.config.picker_name)
end

--- Rebuild positional weight_buf from named weights in feature name order.
--- Uses config.feature_names if set (for item pickers), else scorer.FEATURE_NAMES (file picker).
--- Must only be called after current_weights has been assigned.
---@param cs ClassicState
local function rebuild_weight_buf(cs)
  local weights = cs.current_weights --[[@as table]]
  local feature_names = cs.config.feature_names or scorer.FEATURE_NAMES
  cs.weight_buf = {}
  for i, name in ipairs(feature_names) do
    cs.weight_buf[i] = weights[name] or 0
  end
end

--- Ensure weights are loaded and available
---@param cs ClassicState
---@param force_reload boolean? Force reload weights even if already loaded
---@return table
local function ensure_weights(cs, force_reload)
  if not cs.current_weights or force_reload then
    cs.current_weights = require("neural-open.weights").get_weights("classic", cs.config.picker_name)
    -- Backfill any new features from defaults that are missing in saved weights
    if cs.config.default_weights then
      for key, value in pairs(cs.config.default_weights) do
        if cs.current_weights[key] == nil then
          cs.current_weights[key] = value
        end
      end
    end
    rebuild_weight_buf(cs)
  end
  return cs.current_weights
end

--- Calculate weighted component scores from a flat input buffer
---@param cs ClassicState
---@param input_buf number[] Flat array of normalized features in FEATURE_NAMES order
---@return table<string, number>
local function calculate_components(cs, input_buf)
  local weights = ensure_weights(cs)
  local feature_names = cs.config.feature_names or scorer.FEATURE_NAMES
  local components = {}
  for i, name in ipairs(feature_names) do
    if weights[name] then
      components[name] = input_buf[i] * weights[name]
    end
  end
  return components
end

--- Calculate weight adjustments based on component differences
---@param cs ClassicState
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@return table adjustments, number num_higher_items
local function calculate_adjustments(cs, selected_item, ranked_items)
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
  local weights = ensure_weights(cs)

  -- Initialize adjustments
  local adjustments = {}
  for key, _ in pairs(weights) do
    adjustments[key] = 0
  end

  -- Calculate components from input buffer
  local selected_components = {}
  if selected_item.nos and selected_item.nos.input_buf then
    selected_components = calculate_components(cs, selected_item.nos.input_buf)
  end

  -- Compare with all higher-ranked items
  for i = 1, selected_rank - 1 do
    local higher_item = ranked_items[i]
    if higher_item then
      -- Calculate components from input buffer
      local higher_components = {}
      if higher_item.nos and higher_item.nos.input_buf then
        higher_components = calculate_components(cs, higher_item.nos.input_buf)
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
  local learning_rate = cs.config.learning_rate or 0.6
  for key, adj in pairs(adjustments) do
    if adj ~= 0 then
      adjustments[key] = (adj / num_higher_items) * learning_rate
    end
  end

  return adjustments, num_higher_items
end

--- Apply adjustments to weights and calculate changes
---@param cs ClassicState
---@param adjustments table
---@param apply boolean Whether to actually apply the changes
---@return table new_weights, table changes, boolean has_changes
local function apply_adjustments(cs, adjustments, apply)
  local weights = ensure_weights(cs)

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
    -- Update internal state and rebuild positional weight buffer
    cs.current_weights = new_weights
    rebuild_weight_buf(cs)

    -- Format changes for notification
    local formatted_changes = {}
    local default_weights = get_default_weights(cs) or {}
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

--- Simulate weight adjustments without applying them
---@param cs ClassicState
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@return table?
local function simulate_weight_adjustments(cs, selected_item, ranked_items)
  local adjustments, num_higher_items = calculate_adjustments(cs, selected_item, ranked_items)

  if num_higher_items == 0 then
    return nil
  end

  local new_weights, changes, has_changes = apply_adjustments(cs, adjustments, false) -- Don't apply

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

local fmt = require("neural-open.debug_fmt")

--- Generate debug view for classic algorithm
---@param cs ClassicState
---@param item NeuralOpenItem
---@param all_items NeuralOpenItem[]?
---@return string[], table[]?
local function debug_view_impl(cs, item, all_items)
  local lines = {}
  local hl = {}
  local weights = ensure_weights(cs)

  fmt.add_title(lines, hl, "Classic Algorithm")
  table.insert(lines, "")
  fmt.add_label(lines, hl, "Algorithm", "Weighted sum with self-learning")
  fmt.add_label(lines, hl, "Learning Rate", string.format("%.2f", cs.config.learning_rate or 0.6))
  table.insert(lines, "")

  if item.nos then
    fmt.add_label(lines, hl, "Total Neural Score", string.format("%.2f", item.nos.neural_score or 0))
    if item.score then
      fmt.add_label(lines, hl, "Final Snacks Score", string.format("%.2f", item.score))
    end
    table.insert(lines, "")

    -- Features table (raw + normalized)
    local all_features = cs.config.feature_names or scorer.FEATURE_NAMES
    local normalized_features = {}
    if item.nos.input_buf then
      for i, name in ipairs(all_features) do
        normalized_features[name] = item.nos.input_buf[i]
      end
    end

    local features_rows = { { "Features:", "Raw", "Normalized" } }
    for _, name in ipairs(all_features) do
      local raw_value = (item.nos.raw_features and item.nos.raw_features[name]) or 0
      local normalized_value = normalized_features[name] or 0
      table.insert(features_rows, {
        fmt.format_feature_name(name),
        string.format("%.2f", raw_value),
        string.format("%.4f", normalized_value),
      })
    end
    fmt.format_table(lines, hl, features_rows)
    table.insert(lines, "")

    -- Weighted components table
    local components = {}
    if item.nos.input_buf then
      components = calculate_components(cs, item.nos.input_buf)
    end

    local sorted_components = {}
    for _, name in ipairs(all_features) do
      local value = components[name] or 0
      table.insert(sorted_components, { name = name, value = value })
    end
    table.sort(sorted_components, function(a, b)
      return a.value > b.value
    end)

    local weighted_rows = { { "Weighted:", "Norm", "Weight", "(Default)", "Score" } }
    for _, comp in ipairs(sorted_components) do
      local name = comp.name
      local normalized = normalized_features[name] or 0
      local weight = weights[name] or 0
      local default_weights = get_default_weights(cs) or {}
      local default_weight = default_weights[name] or 0
      table.insert(weighted_rows, {
        fmt.format_feature_name(name),
        string.format("%.4f", normalized),
        string.format("%.1f", weight),
        string.format("(%.1f)", default_weight),
        string.format("%.2f", comp.value),
      })
    end
    fmt.format_table(lines, hl, weighted_rows)
    table.insert(lines, "")

    -- Weight adjustment preview
    if all_items then
      fmt.add_title(lines, hl, "Potential Weight Adjustments (if selected)")
      table.insert(lines, "")

      -- Find current item's rank
      local current_rank = nil
      local id = scorer.get_item_identity(item)
      for i, ranked_item in ipairs(all_items) do
        local ranked_id = scorer.get_item_identity(ranked_item)
        if id and ranked_id and id == ranked_id then
          current_rank = i
          item.neural_rank = i
          break
        end
      end

      if current_rank and current_rank > 1 then
        local simulation = simulate_weight_adjustments(cs, item, all_items)

        if simulation and simulation.changes then
          fmt.add_label(
            lines,
            hl,
            "Rank",
            string.format("#%d (comparing with %d higher items)", current_rank, simulation.compared_with)
          )
          table.insert(lines, "")

          local sorted_changes = {}
          for key, change in pairs(simulation.changes) do
            table.insert(sorted_changes, { key = key, change = change })
          end
          table.sort(sorted_changes, function(a, b)
            return math.abs(a.change.delta) > math.abs(b.change.delta)
          end)

          for _, entry in ipairs(sorted_changes) do
            local change = entry.change
            local sign = change.delta > 0 and "+" or ""
            fmt.add_label(
              lines,
              hl,
              fmt.format_feature_name(entry.key),
              string.format(
                "%.2f → %.2f (%s%.2f) %s",
                change.old,
                change.new,
                sign,
                change.delta,
                change.delta > 0 and "↑" or "↓"
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

  return lines, hl
end

--- Create a per-picker instance of the classic algorithm
---@param config NosClassicConfig
---@return table instance Algorithm instance with closures over per-picker state
function M.create_instance(config)
  local picker_name = config.picker_name or "__default__"
  local cs = states[picker_name]
  if not cs then
    cs = new_state()
    states[picker_name] = cs
  end
  init_state(cs, config)

  local instance = {
    calculate_score = function(input_buf)
      if not cs.weight_buf then
        ensure_weights(cs)
      end
      local wb = cs.weight_buf --[[@as number[] ]]
      local score = 0
      for i = 1, #input_buf do
        score = score + input_buf[i] * wb[i]
      end
      return score
    end,
    update_weights = function(selected_item, ranked_items, latency_ctx)
      ensure_weights(cs, true)
      local adjustments, _ = calculate_adjustments(cs, selected_item, ranked_items)
      local new_weights, _, has_changes = apply_adjustments(cs, adjustments, true)
      if has_changes then
        local weights_module = require("neural-open.weights")
        weights_module.save_weights("classic", new_weights, latency_ctx, cs.config.picker_name)
      end
    end,
    load_weights = function()
      ensure_weights(cs, true)
    end,
    debug_view = function(item, all_items)
      return debug_view_impl(cs, item, all_items)
    end,
    get_name = function()
      return "classic"
    end,
    init = function() end, -- no-op, config already set
    simulate_weight_adjustments = function(selected_item, ranked_items)
      return simulate_weight_adjustments(cs, selected_item, ranked_items)
    end,
  }

  return instance
end

---@diagnostic disable-next-line: undefined-field
if _G._TEST then
  function M._reset_states()
    states = {}
  end
end

return M
