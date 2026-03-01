--- Training pipeline functions for the Neural Network algorithm.
--- Extracted from nn.lua to separate training-time code from inference and state management.
local nn_core = require("neural-open.algorithms.nn_core")

local M = {}

--- Convert a flat input buffer to matrix format with optional match dropout
---@param input_buf number[] Flat array of normalized features in canonical order
---@param drop_match_features? boolean Whether to zero out match/virtual_name features
---@param match_idx? number Index of match feature in input_buf
---@param virtual_name_idx? number Index of virtual_name feature (nil for item pickers)
---@return table Input matrix
function M.features_to_input(input_buf, drop_match_features, match_idx, virtual_name_idx)
  -- Copy is required: input_buf is a shared mutable buffer; dropout would corrupt it
  local input = {}
  for i = 1, #input_buf do
    input[i] = input_buf[i]
  end
  if drop_match_features then
    if match_idx then
      input[match_idx] = 0
    end
    if virtual_name_idx then
      input[virtual_name_idx] = 0
    end
  end
  return { input } -- nn_core matrix format: { {v1, v2, ...} }
end

--- Convert input_buf flat array to named features table (for debug/display only)
---@param input_buf number[] Flat array of normalized features in canonical order
---@param feature_names? string[] Feature names to use (defaults to scorer.FEATURE_NAMES)
---@return table<string, number>
function M.input_buf_to_features(input_buf, feature_names)
  local names = feature_names or require("neural-open.scorer").FEATURE_NAMES
  local features = {}
  for i, name in ipairs(names) do
    features[name] = input_buf[i]
  end
  return features
end

--- Average gradients by batch size
---@param gradients table Gradients to average (modified in place)
---@param batch_size number Size of the batch
local function average_gradients(gradients, batch_size)
  for i = 1, #gradients do
    if gradients[i] then
      gradients[i] = nn_core.scalar_mul(gradients[i], 1.0 / batch_size)
    end
  end
end

--- Construct multiple batches from pairs and history as pair batches
---@param current_pairs table Current training pairs from user selection
---@param history table Training history (array of pairs)
---@param batch_size number Target number of PAIRS per batch
---@param num_batches number Number of batches to create
---@return table Array of pair batches
function M.construct_batches(current_pairs, history, batch_size, num_batches)
  local batch_data = {}
  local used_indices = {}

  -- Minimum batch size: 50% of target batch size
  local min_batch_size = math.ceil(batch_size * 0.5)

  -- First batch includes current pairs (up to batch_size)
  local first_batch_pairs = {}
  for _, pair in ipairs(current_pairs) do
    if #first_batch_pairs >= batch_size then
      break
    end
    table.insert(first_batch_pairs, pair)
  end

  -- Fill rest of first batch from history (recent pairs first)
  local remaining_first = batch_size - #first_batch_pairs
  if remaining_first > 0 and #history > 0 then
    -- Start from end of history (most recent)
    for i = #history, math.max(1, #history - remaining_first + 1), -1 do
      if #first_batch_pairs >= batch_size then
        break
      end
      table.insert(first_batch_pairs, history[i])
      used_indices[i] = true
    end
  end

  -- Only add first batch if it meets minimum size requirement
  if #first_batch_pairs >= min_batch_size then
    table.insert(batch_data, { pairs = first_batch_pairs })
  end

  -- Create additional batches from remaining history
  for _ = 2, num_batches do
    local batch_pairs = {}
    local available = {}

    -- Collect unused history indices
    for i = 1, #history do
      if not used_indices[i] then
        table.insert(available, i)
      end
    end

    -- Stop if no more unique pairs available
    if #available == 0 then
      break
    end

    -- Randomly sample from available history
    local pairs_to_take = math.min(batch_size, #available)
    for _ = 1, pairs_to_take do
      if #available == 0 then
        break
      end
      local rand_idx = math.random(#available)
      local history_idx = available[rand_idx]
      table.insert(batch_pairs, history[history_idx])
      used_indices[history_idx] = true
      table.remove(available, rand_idx)
    end

    -- Only add batch if it meets minimum size requirement
    if #batch_pairs >= min_batch_size then
      table.insert(batch_data, { pairs = batch_pairs })
    end
  end

  return batch_data
end

--- Train the network on multiple batches with pairwise hinge loss
---@param st table State table
---@param batches table Array of pair batches {pairs}
---@param latency_ctx table|nil Optional latency context for performance tracking
---@param nn table The nn module, providing forward_pass, backward_pass_pairwise, and update_parameters
---@return number Average loss across all batches
function M.train_on_batches(st, batches, latency_ctx, nn)
  if not st.weights or #batches == 0 then
    return 0
  end

  -- Track memory usage before training
  local mem_before_kb = collectgarbage("count")

  -- Set latency context for nn_core operations
  nn_core.set_latency_context(latency_ctx)
  nn_core.reset_call_stats()

  local total_loss = 0
  local total_pairs = 0

  -- Initialize loss history if needed
  st.stats.loss_history = st.stats.loss_history or {}

  -- Initialize ranking accuracy history if needed
  st.stats.ranking_accuracy_history = st.stats.ranking_accuracy_history or {}

  -- Get margin from config
  local margin = st.config.margin or 1.0

  for _, batch in ipairs(batches) do
    if batch.pairs and #batch.pairs > 0 then
      local batch_size = #batch.pairs -- Number of pairs in this batch
      local batch_timing = {}

      local batch_loss = 0
      local batch_correct = 0
      local batch_margin_correct = 0

      -- BATCHED FORWARD PASS: Process all items together for proper batch normalization
      -- Combine positive and negative items into single batch so batch norm sees full feature distribution
      local fwd_start = vim.loop.hrtime()

      -- Interleave positive and negative inputs: [pos1, neg1, pos2, neg2, ...]
      -- This ensures batch norm statistics reflect both positive and negative features together
      local combined_inputs = {}
      for i, pair in ipairs(batch.pairs) do
        combined_inputs[2 * i - 1] = pair.positive_input[1] -- Odd indices: positive items
        combined_inputs[2 * i] = pair.negative_input[1] -- Even indices: negative items
      end

      -- Single forward pass for all items (2 * batch_size × features)
      local combined_activations, combined_pre_activations, combined_bn_cache, combined_dropout_masks = nn.forward_pass(
        combined_inputs,
        st.weights,
        st.biases,
        st.gammas,
        st.betas,
        st.running_means,
        st.running_vars,
        true, -- training mode
        st.config.dropout_rates,
        true, -- return logits for training
        st.stats
      )

      local forward_time = vim.loop.hrtime() - fwd_start

      -- Extract logits: odd indices are positive, even indices are negative
      local combined_logits = combined_activations[#combined_activations] -- [2*batch_size × 1]

      -- Compute losses and accuracies by extracting positive/negative logits from interleaved batch
      local pair_losses = {}
      for i = 1, batch_size do
        local logit_pos = combined_logits[2 * i - 1][1] -- Odd index: positive item
        local logit_neg = combined_logits[2 * i][1] -- Even index: negative item

        -- Track ranking accuracy
        local is_correct = logit_pos > logit_neg
        local is_margin_correct = (logit_pos - logit_neg) >= margin

        if is_correct then
          batch_correct = batch_correct + 1
        end
        if is_margin_correct then
          batch_margin_correct = batch_margin_correct + 1
        end

        -- Compute pairwise hinge loss
        local pair_loss = nn_core.pairwise_hinge_loss(logit_pos, logit_neg, margin)
        pair_losses[i] = pair_loss
        batch_loss = batch_loss + pair_loss
      end

      -- BATCHED BACKWARD PASS: Compute gradients for all items at once
      local bwd_start = vim.loop.hrtime()

      -- Create gradient matrix [2*batch_size × 1] for combined batch
      -- Interleaved: [grad_pos1, grad_neg1, grad_pos2, grad_neg2, ...]
      local combined_grads = nn_core.zeros(2 * batch_size, 1)

      for i = 1, batch_size do
        if pair_losses[i] > 0 then
          combined_grads[2 * i - 1][1] = -1.0 -- Positive item: increase score
          combined_grads[2 * i][1] = 1.0 -- Negative item: decrease score
        end
        -- If loss == 0, gradients remain 0 (margin already satisfied)
      end

      -- Single backward pass for all items
      local weight_grads_accumulated, bias_grads_accumulated, gamma_grads_accumulated, beta_grads_accumulated =
        nn.backward_pass_pairwise(
          combined_activations,
          combined_pre_activations,
          combined_grads, -- Pass batched gradients for all items
          st.weights,
          st.gammas,
          combined_bn_cache,
          combined_dropout_masks,
          st.config.dropout_rates
        )

      local backward_time = vim.loop.hrtime() - bwd_start

      batch_timing.forward_ms = forward_time / 1e6
      batch_timing.backward_ms = backward_time / 1e6

      -- AVERAGE GRADIENTS by number of pairs
      if weight_grads_accumulated and bias_grads_accumulated then
        average_gradients(weight_grads_accumulated, batch_size)
        average_gradients(bias_grads_accumulated, batch_size)
        if gamma_grads_accumulated and beta_grads_accumulated then
          average_gradients(gamma_grads_accumulated, batch_size)
          average_gradients(beta_grads_accumulated, batch_size)
        end

        -- CLIP GRADIENTS
        local max_grad_norm = 5.0
        weight_grads_accumulated = nn_core.clip_gradients(weight_grads_accumulated, max_grad_norm)
        bias_grads_accumulated = nn_core.clip_gradients(bias_grads_accumulated, max_grad_norm)
        if gamma_grads_accumulated and beta_grads_accumulated then
          gamma_grads_accumulated = nn_core.clip_gradients(gamma_grads_accumulated, max_grad_norm)
          beta_grads_accumulated = nn_core.clip_gradients(beta_grads_accumulated, max_grad_norm)
        end

        -- UPDATE PARAMETERS
        local update_start = vim.loop.hrtime()
        nn.update_parameters(
          st,
          st.weights,
          st.biases,
          weight_grads_accumulated,
          bias_grads_accumulated,
          st.gammas,
          st.betas,
          gamma_grads_accumulated,
          beta_grads_accumulated,
          st.config.learning_rate,
          st.config
        )
        batch_timing.update_ms = (vim.loop.hrtime() - update_start) / 1e6
      else
        -- No gradients (all pairs had loss = 0)
        batch_timing.update_ms = 0
      end

      -- STORE TIMING in circular buffer
      table.insert(st.stats.batch_timings, batch_timing)
      if #st.stats.batch_timings > 10 then
        table.remove(st.stats.batch_timings, 1)
      end

      -- Calculate average timing
      if #st.stats.batch_timings > 0 then
        local avg_forward, avg_backward, avg_update = 0, 0, 0
        for _, timing in ipairs(st.stats.batch_timings) do
          avg_forward = avg_forward + timing.forward_ms
          avg_backward = avg_backward + timing.backward_ms
          avg_update = avg_update + timing.update_ms
        end
        local n = #st.stats.batch_timings
        st.stats.avg_batch_timing = {
          forward_ms = avg_forward / n,
          backward_ms = avg_backward / n,
          update_ms = avg_update / n,
          total_ms = (avg_forward + avg_backward + avg_update) / n,
        }
      end

      -- Track statistics
      batch_loss = batch_loss / batch_size
      total_loss = total_loss + batch_loss
      total_pairs = total_pairs + batch_size
      st.stats.batches_trained = st.stats.batches_trained + 1

      -- Add individual batch loss to history
      table.insert(st.stats.loss_history, batch_loss)
      if #st.stats.loss_history > 1000 then
        table.remove(st.stats.loss_history, 1)
      end

      -- Track ranking accuracy history
      table.insert(st.stats.ranking_accuracy_history, {
        correct = batch_correct,
        margin_correct = batch_margin_correct,
        total = batch_size,
      })
      if #st.stats.ranking_accuracy_history > 1000 then
        table.remove(st.stats.ranking_accuracy_history, 1)
      end
    end
  end

  -- Update statistics
  if #batches > 0 then
    st.stats.last_loss = total_loss / #batches
    st.stats.samples_per_batch = total_pairs / #batches
  else
    st.stats.last_loss = 0
    st.stats.samples_per_batch = 0
  end

  -- Track memory usage after training
  local mem_after_kb = collectgarbage("count")
  local mem_delta_kb = mem_after_kb - mem_before_kb

  -- Retrieve call stats and clear context
  local call_stats = nn_core.get_call_stats()
  nn_core.set_latency_context(nil)

  -- Add call stats and memory delta to latency metadata
  if latency_ctx then
    local latency = require("neural-open.latency")
    latency.add_metadata(latency_ctx, "nn.training", {
      core_matmul_count = call_stats.matmul_count,
      core_matmul_ops = call_stats.matmul_ops,
      core_element_wise_count = call_stats.element_wise_count,
      core_element_wise_ops = call_stats.element_wise_ops,
      core_element_wise2_count = call_stats.element_wise2_count,
      core_element_wise2_ops = call_stats.element_wise2_ops,
      core_add_count = call_stats.add_count,
      core_subtract_count = call_stats.subtract_count,
      core_scalar_mul_count = call_stats.scalar_mul_count,
      core_batch_norm_count = call_stats.batch_norm_count,
      memory_before_kb = mem_before_kb,
      memory_after_kb = mem_after_kb,
      memory_delta_kb = mem_delta_kb,
    })
  end

  return st.stats.last_loss
end

return M
