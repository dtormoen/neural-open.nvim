--- Neural Network scoring algorithm
local nn_core = require("neural-open.algorithms.nn_core")
local math_exp = math.exp
local M = {}

local states = {} -- picker_name -> state table

--- Create a fresh state table with default values
local function new_state()
  return {
    config = nil,
    weights = nil,
    biases = nil,
    gammas = nil, -- Batch norm scale parameters
    betas = nil, -- Batch norm shift parameters
    running_means = nil, -- Batch norm running means for inference
    running_vars = nil, -- Batch norm running variances for inference
    training_history = nil,
    dropout_masks = nil, -- Store dropout masks for backward pass
    optimizer_type = "sgd", -- Current optimizer type
    optimizer_state = nil, -- Optimizer-specific state (lazy initialized)
    inference_cache = nil, -- Fused weights/biases for fast inference
    stats = {
      samples_processed = 0,
      batches_trained = 0,
      last_loss = 0,
      loss_history = {},
      ranking_accuracy_history = {}, -- Circular buffer of {correct, margin_correct, total} per batch
      samples_per_batch = 0,
      weight_norms = {}, -- L2 norms of weights per layer
      avg_weight_magnitudes = {}, -- Average weight magnitude per layer
      dropout_active_rates = {}, -- Percentage of active neurons per layer during training
      batch_timings = {}, -- Circular buffer of last 10 batch timings {forward_ms, backward_ms, update_ms}
      avg_batch_timing = nil, -- Average timing of last 10 batches
    },
  }
end

--- Default feature names in input order (file picker, 11 features)
local scorer = require("neural-open.scorer")
local DEFAULT_FEATURE_NAMES = scorer.FEATURE_NAMES

--- Default backfill values for new features during input-size migration.
--- Each entry is either a static number or a function(row, feature_indices) -> number.
--- Functions receive the existing row data and a name->index map of the OLD features,
--- allowing heuristics to look up values by name rather than hardcoded positions.
--- Features not listed here default to 0.0.
local MIGRATION_DEFAULTS = {
  not_current = function(row, idx)
    -- Heuristic: high trigram similarity + max proximity suggests the current file
    local trigram = idx.trigram and row[idx.trigram]
    local proximity = idx.proximity and row[idx.proximity]
    if trigram and proximity then
      local is_current = trigram >= 0.99 and proximity == 1.0
      return is_current and 0.0 or 1.0
    end
    return 1.0 -- safe default when features unavailable (e.g., item pickers)
  end,
  not_last_selected = 1.0,
}

--- Get feature names for a state table (respects per-picker config override)
---@param st table State table
---@return string[]
local function get_feature_names(st)
  return st.config.feature_names or DEFAULT_FEATURE_NAMES
end

--- Initialize a state table with configuration (validation + setup)
local function init_state(st, config)
  st.config = vim.deepcopy(config or {})

  if not st.config.architecture then
    error("NN algorithm: config.architecture is required")
  end
  if not st.config.optimizer then
    error("NN algorithm: config.optimizer is required")
  end

  st.optimizer_type = st.config.optimizer

  if st.optimizer_type ~= "sgd" and st.optimizer_type ~= "adamw" then
    error(string.format("Invalid optimizer type: %s. Must be 'sgd' or 'adamw'", st.optimizer_type))
  end

  if st.config.dropout_rates then
    local hidden_layer_count = #st.config.architecture - 2
    if #st.config.dropout_rates ~= hidden_layer_count then
      error(
        string.format(
          "Dropout rates array length (%d) must match number of hidden layers (%d)",
          #st.config.dropout_rates,
          hidden_layer_count
        )
      )
    end

    for i, rate in ipairs(st.config.dropout_rates) do
      if rate < 0 or rate >= 1 then
        error(string.format("Dropout rate for layer %d must be in [0, 1) range, got %.2f", i, rate))
      end
    end
  end

  -- Precompute feature indices for match dropout
  local feature_names = get_feature_names(st)
  st.match_idx = nil
  st.virtual_name_idx = nil
  for i, name in ipairs(feature_names) do
    if name == "match" then
      st.match_idx = i
    elseif name == "virtual_name" then
      st.virtual_name_idx = i
    end
  end

  math.randomseed(os.time())
end

--- Forward propagation through the network with batch normalization and dropout
--- Uses Leaky ReLU activation (alpha=0.01) for all hidden layers to prevent dying neurons
--- and improve gradient flow, especially beneficial for binary features
--- For the output layer, returns logits (pre-sigmoid) to enable proper loss computation
---@param input table Input batch matrix (batch_size × features)
---@param weights table Network weights
---@param biases table Network biases
---@param gammas table? Batch norm scale parameters
---@param betas table? Batch norm shift parameters
---@param running_means table? Batch norm running means (updated during training, used during inference)
---@param running_vars table? Batch norm running variances (updated during training, used during inference)
---@param training boolean Whether in training mode (compute batch stats) or inference
---@param dropout_rates table? Dropout rates for each hidden layer
---@param return_logits boolean? If true, output layer returns logits instead of sigmoid (for training)
---@param stats table? Stats table for tracking dropout statistics
---@return table activations, table pre_activations, table bn_cache, table dropout_masks
local function forward_pass(
  input,
  weights,
  biases,
  gammas,
  betas,
  running_means,
  running_vars,
  training,
  dropout_rates,
  return_logits,
  stats
)
  local activations = { input }
  local pre_activations = {}
  local bn_cache = {} -- Store batch norm statistics for backward pass
  local dropout_masks = {} -- Store dropout masks for backward pass

  training = training == nil and true or training -- Default to training mode
  return_logits = return_logits == nil and false or return_logits -- Default to sigmoid output

  for i = 1, #weights do
    -- Linear transformation: z = X @ W + b
    local z = nn_core.matmul(activations[i], weights[i])
    z = nn_core.add_bias(z, biases[i])

    -- Apply batch normalization to hidden layers (not output)
    if i < #weights and gammas and betas and gammas[i] and betas[i] then
      local z_norm, batch_mean, batch_var = nn_core.batch_normalize(
        z,
        gammas[i],
        betas[i],
        1e-8, -- epsilon
        training,
        running_means and running_means[i],
        running_vars and running_vars[i],
        0.1 -- momentum
      )
      -- Cache statistics for backward pass (only needed in training)
      if training then
        bn_cache[i] = { input = z, mean = batch_mean, var = batch_var }
      end
      z = z_norm
    end

    pre_activations[i] = z

    -- Apply activation function
    local activation
    if i < #weights then
      -- All hidden layers use Leaky ReLU for better gradient flow and stability
      activation = nn_core.element_wise(z, function(x)
        return nn_core.leaky_relu(x, 0.01)
      end)

      -- Apply dropout to hidden layers during training
      if training and dropout_rates and dropout_rates[i] and dropout_rates[i] > 0 then
        activation, dropout_masks[i] = nn_core.dropout(activation, dropout_rates[i], true)

        -- Track dropout statistics
        if stats then
          local active_count = 0
          local total_count = 0
          for row = 1, #dropout_masks[i] do
            for col = 1, #dropout_masks[i][row] do
              total_count = total_count + 1
              if dropout_masks[i][row][col] > 0 then
                active_count = active_count + 1
              end
            end
          end
          -- Ensure the array is properly initialized
          if not stats.dropout_active_rates then
            stats.dropout_active_rates = {}
          end
          stats.dropout_active_rates[i] = (active_count / total_count) * 100
        end
      elseif stats and training then
        -- Initialize with 0 for layers without dropout to prevent nils
        if not stats.dropout_active_rates then
          stats.dropout_active_rates = {}
        end
        stats.dropout_active_rates[i] = 0
      end
    else
      -- Output layer: return logits for training, sigmoid for inference
      if return_logits then
        activation = z -- Return logits (pre-sigmoid) for proper loss computation
      else
        activation = nn_core.element_wise(z, nn_core.sigmoid)
      end
    end
    activations[i + 1] = activation
  end

  return activations, pre_activations, bn_cache, dropout_masks
end

--- Backward propagation for pairwise loss with batched output gradients
--- Assumes output layer returns logits (no sigmoid activation applied)
---@param activations table Forward pass activations
---@param pre_activations table Forward pass pre-activations
---@param output_grad table|number Gradient for output: [batch_size × 1] matrix or scalar for single sample
---@param weights table Network weights
---@param gammas table? Batch norm scale parameters
---@param bn_cache table? Batch norm cache from forward pass
---@param dropout_masks table? Dropout masks from forward pass
---@param config_dropout_rates table? Dropout rates from config
---@return table weight_gradients, table bias_gradients, table gamma_gradients, table beta_gradients
local function backward_pass_pairwise(
  activations,
  pre_activations,
  output_grad,
  weights,
  gammas,
  bn_cache,
  dropout_masks,
  config_dropout_rates
)
  local weight_gradients = {}
  local bias_gradients = {}
  local gamma_gradients = {}
  local beta_gradients = {}

  -- Start with output layer gradient
  -- For batched processing: output_grad is already [batch_size × 1] matrix
  -- For single sample: output_grad is scalar, convert to [1 × 1] matrix
  -- No sigmoid derivative needed since we're computing gradient w.r.t. logits
  local delta
  if type(output_grad) == "number" then
    delta = nn_core.vector_to_matrix({ output_grad })
  else
    delta = output_grad -- Already a [batch_size × 1] matrix
  end

  local deltas = { delta }

  -- Backpropagate through layers
  for i = #weights, 1, -1 do
    local current_delta = deltas[1]

    -- For hidden layers, backprop through activation function and dropout
    -- For output layer (i == #weights), gradient flows directly through (no activation)
    if i < #weights then
      -- Apply dropout mask if present
      if dropout_masks and dropout_masks[i] then
        local dropout_rate = config_dropout_rates and config_dropout_rates[i] or 0
        if dropout_rate > 0 then
          local scale = 1.0 / (1.0 - dropout_rate)
          current_delta = nn_core.hadamard(current_delta, dropout_masks[i])
          current_delta = nn_core.scalar_mul(current_delta, scale)
        end
      end

      -- Apply activation derivative (Leaky ReLU)
      local z = pre_activations[i]
      local activation_derivative = nn_core.element_wise(z, function(x)
        return nn_core.leaky_relu_derivative(x, 0.01)
      end)
      current_delta = nn_core.hadamard(current_delta, activation_derivative)
    end

    -- Backprop through batch norm if present
    if i < #weights and gammas and gammas[i] and bn_cache and bn_cache[i] then
      local grad_input, grad_gamma, grad_beta = nn_core.batch_normalize_backward(
        current_delta,
        bn_cache[i].input,
        gammas[i],
        bn_cache[i].mean,
        bn_cache[i].var,
        1e-8
      )
      gamma_gradients[i] = grad_gamma
      beta_gradients[i] = grad_beta
      current_delta = grad_input
    end

    -- Compute weight gradient: ∇W = a^T @ δ
    local activation_t = nn_core.transpose(activations[i])
    weight_gradients[i] = nn_core.matmul(activation_t, current_delta)

    -- Compute bias gradient: ∇b = sum(δ, axis=0)
    -- Sum over batch dimension to get [1 × output_size]
    local batch_size = #current_delta
    local output_size = #current_delta[1]
    local bias_grad = nn_core.zeros(1, output_size)
    for b = 1, batch_size do
      for j = 1, output_size do
        bias_grad[1][j] = bias_grad[1][j] + current_delta[b][j]
      end
    end
    bias_gradients[i] = bias_grad

    -- Propagate error to previous layer if not at input
    if i > 1 then
      local weight_t = nn_core.transpose(weights[i])
      local delta_prop = nn_core.matmul(current_delta, weight_t)
      table.insert(deltas, 1, delta_prop)
    end
  end

  return weight_gradients, bias_gradients, gamma_gradients, beta_gradients
end

--- Calculate learning rate warmup factor
--- Returns a factor to multiply learning rate by, implementing linear warmup
---@param timestep number Current timestep (1-indexed)
---@param warmup_steps number Number of steps to warm up (0 = no warmup)
---@param start_factor number Starting learning rate factor (default 0.1)
---@return number warmup_factor Factor to multiply learning rate by (1.0 = full LR)
local function calculate_warmup_factor(timestep, warmup_steps, start_factor)
  if warmup_steps <= 0 or timestep > warmup_steps then
    return 1.0 -- No warmup or past warmup phase
  end

  -- Linear warmup: factor = (t / warmup_steps) * (1 - start_factor) + start_factor
  -- At t=1: factor = start_factor (e.g., 0.1 = 10% of LR)
  -- At t=warmup_steps: factor = 1.0 (100% of LR)
  local progress = timestep / warmup_steps
  return progress * (1.0 - start_factor) + start_factor
end

--- Initialize optimizer state based on optimizer type and network architecture
---@param optimizer_type string Optimizer type ("sgd" or "adamw")
---@param architecture number[] Network architecture
---@return table? optimizer_state Initialized state for the optimizer (nil for SGD)
local function init_optimizer_state(optimizer_type, architecture)
  if optimizer_type == "sgd" then
    return nil -- SGD doesn't need optimizer state
  elseif optimizer_type == "adamw" then
    local optimizer_state = {
      timestep = 0,
      moments = {
        first = {
          weights = {},
          biases = {},
          gammas = {},
          betas = {},
        },
        second = {
          weights = {},
          biases = {},
          gammas = {},
          betas = {},
        },
      },
    }

    -- Initialize moments for each layer
    for i = 1, #architecture - 1 do
      local input_size = architecture[i]
      local output_size = architecture[i + 1]

      -- Initialize weight moments
      optimizer_state.moments.first.weights[i] = nn_core.zeros(input_size, output_size)
      optimizer_state.moments.second.weights[i] = nn_core.zeros(input_size, output_size)

      -- Initialize bias moments
      optimizer_state.moments.first.biases[i] = nn_core.zeros(1, output_size)
      optimizer_state.moments.second.biases[i] = nn_core.zeros(1, output_size)

      -- Initialize batch norm moments for hidden layers
      if i < #architecture - 1 then
        optimizer_state.moments.first.gammas[i] = nn_core.zeros(1, output_size)
        optimizer_state.moments.second.gammas[i] = nn_core.zeros(1, output_size)
        optimizer_state.moments.first.betas[i] = nn_core.zeros(1, output_size)
        optimizer_state.moments.second.betas[i] = nn_core.zeros(1, output_size)
      end
    end

    return optimizer_state
  else
    error("Unknown optimizer type: " .. tostring(optimizer_type))
  end
end

--- Update weights using SGD optimizer
---@param st table State table
---@param weights table Current weights
---@param biases table Current biases
---@param weight_gradients table Weight gradients
---@param bias_gradients table Bias gradients
---@param gammas table? Batch norm scale parameters
---@param betas table? Batch norm shift parameters
---@param gamma_gradients table? Gamma gradients
---@param beta_gradients table? Beta gradients
---@param learning_rate number Learning rate
---@param config table Configuration with weight_decay settings
local function update_parameters_sgd(
  st,
  weights,
  biases,
  weight_gradients,
  bias_gradients,
  gammas,
  betas,
  gamma_gradients,
  beta_gradients,
  learning_rate,
  config
)
  -- Initialize optimizer state if needed (for timestep tracking in warmup)
  if not st.optimizer_state then
    st.optimizer_state = { timestep = 0 }
  end

  -- Increment timestep
  st.optimizer_state.timestep = st.optimizer_state.timestep + 1
  local t = st.optimizer_state.timestep

  -- Apply learning rate warmup
  local warmup_steps = config.warmup_steps or 0
  local warmup_start_factor = config.warmup_start_factor or 0.1
  local warmup_factor = calculate_warmup_factor(t, warmup_steps, warmup_start_factor)
  local effective_lr = learning_rate * warmup_factor

  for i = 1, #weights do
    -- Apply weight decay (L2 regularization) to weight gradients
    local layer_decay = config.weight_decay or 0
    if config.layer_decay_multipliers and config.layer_decay_multipliers[i] then
      layer_decay = layer_decay * config.layer_decay_multipliers[i]
    end

    if layer_decay > 0 then
      -- Add decay term to weight gradient: ∇W = ∇W + λ*W
      weight_gradients[i] = nn_core.add(weight_gradients[i], nn_core.scalar_mul(weights[i], layer_decay))
    end

    -- W = W - α × ∇W
    weights[i] = nn_core.subtract(weights[i], nn_core.scalar_mul(weight_gradients[i], effective_lr))

    -- b = b - α × ∇b
    biases[i] = nn_core.subtract(biases[i], nn_core.scalar_mul(bias_gradients[i], effective_lr))

    -- Update batch norm parameters if present
    if gammas and gammas[i] and gamma_gradients and gamma_gradients[i] then
      gammas[i] = nn_core.subtract(gammas[i], nn_core.scalar_mul(gamma_gradients[i], effective_lr))
    end
    if betas and betas[i] and beta_gradients and beta_gradients[i] then
      betas[i] = nn_core.subtract(betas[i], nn_core.scalar_mul(beta_gradients[i], effective_lr))
    end
  end

  -- Calculate and store weight statistics
  if st.stats then
    st.stats.weight_norms = {}
    st.stats.avg_weight_magnitudes = {}
    for i = 1, #weights do
      local sum_squared = 0
      local sum_abs = 0
      local count = 0
      for j = 1, #weights[i] do
        for k = 1, #weights[i][j] do
          sum_squared = sum_squared + weights[i][j][k] * weights[i][j][k]
          sum_abs = sum_abs + math.abs(weights[i][j][k])
          count = count + 1
        end
      end
      st.stats.weight_norms[i] = math.sqrt(sum_squared)
      st.stats.avg_weight_magnitudes[i] = sum_abs / count
    end
  end
end

--- Update weights using AdamW optimizer
---@param st table State table
---@param weights table Current weights
---@param biases table Current biases
---@param weight_gradients table Weight gradients
---@param bias_gradients table Bias gradients
---@param gammas table? Batch norm scale parameters
---@param betas table? Batch norm shift parameters
---@param gamma_gradients table? Gamma gradients
---@param beta_gradients table? Beta gradients
---@param learning_rate number Learning rate
---@param config table Configuration with AdamW settings
local function update_parameters_adamw(
  st,
  weights,
  biases,
  weight_gradients,
  bias_gradients,
  gammas,
  betas,
  gamma_gradients,
  beta_gradients,
  learning_rate,
  config
)
  local beta1 = config.adam_beta1 or 0.9
  local beta2 = config.adam_beta2 or 0.999
  local epsilon = config.adam_epsilon or 1e-8

  -- Initialize optimizer state if needed
  if not st.optimizer_state then
    st.optimizer_state = init_optimizer_state("adamw", config.architecture)
  end

  -- Increment timestep
  st.optimizer_state.timestep = st.optimizer_state.timestep + 1
  local t = st.optimizer_state.timestep

  -- Apply learning rate warmup
  local warmup_steps = config.warmup_steps or 0
  local warmup_start_factor = config.warmup_start_factor or 0.1
  local warmup_factor = calculate_warmup_factor(t, warmup_steps, warmup_start_factor)
  local effective_lr = learning_rate * warmup_factor

  -- Bias correction factors
  local bias_correction1 = 1 - beta1 ^ t
  local bias_correction2 = 1 - beta2 ^ t

  -- Update each layer
  for i = 1, #weights do
    -- Get moments for weights
    local m_w = st.optimizer_state.moments.first.weights[i]
    local v_w = st.optimizer_state.moments.second.weights[i]

    -- Update first moment: m = β1 * m + (1 - β1) * g
    m_w = nn_core.add(nn_core.scalar_mul(m_w, beta1), nn_core.scalar_mul(weight_gradients[i], 1 - beta1))

    -- Update second moment: v = β2 * v + (1 - β2) * g²
    local g_squared = nn_core.element_wise(weight_gradients[i], function(x)
      return x * x
    end)
    v_w = nn_core.add(nn_core.scalar_mul(v_w, beta2), nn_core.scalar_mul(g_squared, 1 - beta2))

    -- Bias-corrected moments
    local m_hat = nn_core.scalar_mul(m_w, 1 / bias_correction1)
    local v_hat = nn_core.scalar_mul(v_w, 1 / bias_correction2)

    -- AdamW update: m_hat / (sqrt(v_hat) + ε)
    local v_sqrt_eps = nn_core.element_wise(v_hat, function(x)
      return math.sqrt(x) + epsilon
    end)
    local update = nn_core.element_wise2(m_hat, v_sqrt_eps, function(m, v)
      return m / v
    end)

    -- Apply decoupled weight decay directly to weights
    local layer_decay = config.weight_decay or 0
    if config.layer_decay_multipliers and config.layer_decay_multipliers[i] then
      layer_decay = layer_decay * config.layer_decay_multipliers[i]
    end

    if layer_decay > 0 then
      update = nn_core.add(update, nn_core.scalar_mul(weights[i], layer_decay))
    end

    -- Update weights: W = W - α * update
    weights[i] = nn_core.subtract(weights[i], nn_core.scalar_mul(update, effective_lr))

    -- Store updated moments
    st.optimizer_state.moments.first.weights[i] = m_w
    st.optimizer_state.moments.second.weights[i] = v_w

    -- Update biases
    local m_b = st.optimizer_state.moments.first.biases[i]
    local v_b = st.optimizer_state.moments.second.biases[i]

    m_b = nn_core.add(nn_core.scalar_mul(m_b, beta1), nn_core.scalar_mul(bias_gradients[i], 1 - beta1))

    local b_g_squared = nn_core.element_wise(bias_gradients[i], function(x)
      return x * x
    end)
    v_b = nn_core.add(nn_core.scalar_mul(v_b, beta2), nn_core.scalar_mul(b_g_squared, 1 - beta2))

    local m_b_hat = nn_core.scalar_mul(m_b, 1 / bias_correction1)
    local v_b_hat = nn_core.scalar_mul(v_b, 1 / bias_correction2)

    local v_b_sqrt_eps = nn_core.element_wise(v_b_hat, function(x)
      return math.sqrt(x) + epsilon
    end)
    local b_update = nn_core.element_wise2(m_b_hat, v_b_sqrt_eps, function(m, v)
      return m / v
    end)

    biases[i] = nn_core.subtract(biases[i], nn_core.scalar_mul(b_update, effective_lr))

    st.optimizer_state.moments.first.biases[i] = m_b
    st.optimizer_state.moments.second.biases[i] = v_b

    -- Update batch norm parameters if present
    if gammas and gammas[i] and gamma_gradients and gamma_gradients[i] then
      local m_g = st.optimizer_state.moments.first.gammas[i]
      local v_g = st.optimizer_state.moments.second.gammas[i]

      m_g = nn_core.add(nn_core.scalar_mul(m_g, beta1), nn_core.scalar_mul(gamma_gradients[i], 1 - beta1))

      local g_g_squared = nn_core.element_wise(gamma_gradients[i], function(x)
        return x * x
      end)
      v_g = nn_core.add(nn_core.scalar_mul(v_g, beta2), nn_core.scalar_mul(g_g_squared, 1 - beta2))

      local m_g_hat = nn_core.scalar_mul(m_g, 1 / bias_correction1)
      local v_g_hat = nn_core.scalar_mul(v_g, 1 / bias_correction2)

      local v_g_sqrt_eps = nn_core.element_wise(v_g_hat, function(x)
        return math.sqrt(x) + epsilon
      end)
      local g_update = nn_core.element_wise2(m_g_hat, v_g_sqrt_eps, function(m, v)
        return m / v
      end)

      gammas[i] = nn_core.subtract(gammas[i], nn_core.scalar_mul(g_update, effective_lr))

      st.optimizer_state.moments.first.gammas[i] = m_g
      st.optimizer_state.moments.second.gammas[i] = v_g
    end

    if betas and betas[i] and beta_gradients and beta_gradients[i] then
      local m_beta = st.optimizer_state.moments.first.betas[i]
      local v_beta = st.optimizer_state.moments.second.betas[i]

      m_beta = nn_core.add(nn_core.scalar_mul(m_beta, beta1), nn_core.scalar_mul(beta_gradients[i], 1 - beta1))

      local beta_g_squared = nn_core.element_wise(beta_gradients[i], function(x)
        return x * x
      end)
      v_beta = nn_core.add(nn_core.scalar_mul(v_beta, beta2), nn_core.scalar_mul(beta_g_squared, 1 - beta2))

      local m_beta_hat = nn_core.scalar_mul(m_beta, 1 / bias_correction1)
      local v_beta_hat = nn_core.scalar_mul(v_beta, 1 / bias_correction2)

      local v_beta_sqrt_eps = nn_core.element_wise(v_beta_hat, function(x)
        return math.sqrt(x) + epsilon
      end)
      local beta_update = nn_core.element_wise2(m_beta_hat, v_beta_sqrt_eps, function(m, v)
        return m / v
      end)

      betas[i] = nn_core.subtract(betas[i], nn_core.scalar_mul(beta_update, effective_lr))

      st.optimizer_state.moments.first.betas[i] = m_beta
      st.optimizer_state.moments.second.betas[i] = v_beta
    end
  end

  -- Calculate and store weight statistics
  if st.stats then
    st.stats.weight_norms = {}
    st.stats.avg_weight_magnitudes = {}
    for i = 1, #weights do
      local sum_squared = 0
      local sum_abs = 0
      local count = 0
      for j = 1, #weights[i] do
        for k = 1, #weights[i][j] do
          sum_squared = sum_squared + weights[i][j][k] * weights[i][j][k]
          sum_abs = sum_abs + math.abs(weights[i][j][k])
          count = count + 1
        end
      end
      st.stats.weight_norms[i] = math.sqrt(sum_squared)
      st.stats.avg_weight_magnitudes[i] = sum_abs / count
    end
  end
end

--- Update weights using configured optimizer (dispatcher function)
---@param st table State table
---@param weights table Current weights
---@param biases table Current biases
---@param weight_gradients table Weight gradients
---@param bias_gradients table Bias gradients
---@param gammas table? Batch norm scale parameters
---@param betas table? Batch norm shift parameters
---@param gamma_gradients table? Gamma gradients
---@param beta_gradients table? Beta gradients
---@param learning_rate number Learning rate
---@param config table Configuration with optimizer settings
local function update_parameters(
  st,
  weights,
  biases,
  weight_gradients,
  bias_gradients,
  gammas,
  betas,
  gamma_gradients,
  beta_gradients,
  learning_rate,
  config
)
  if st.optimizer_type == "adamw" then
    return update_parameters_adamw(
      st,
      weights,
      biases,
      weight_gradients,
      bias_gradients,
      gammas,
      betas,
      gamma_gradients,
      beta_gradients,
      learning_rate,
      config
    )
  else
    return update_parameters_sgd(
      st,
      weights,
      biases,
      weight_gradients,
      bias_gradients,
      gammas,
      betas,
      gamma_gradients,
      beta_gradients,
      learning_rate,
      config
    )
  end
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

--- Convert a flat input buffer to matrix format with optional match dropout
---@param input_buf number[] Flat array of normalized features in canonical order
---@param drop_match_features? boolean Whether to zero out match/virtual_name features
---@param match_idx? number Index of match feature in input_buf
---@param virtual_name_idx? number Index of virtual_name feature (nil for item pickers)
---@return table Input matrix
local function features_to_input(input_buf, drop_match_features, match_idx, virtual_name_idx)
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
---@param feature_names? string[] Feature names to use (defaults to DEFAULT_FEATURE_NAMES)
---@return table<string, number>
local function input_buf_to_features(input_buf, feature_names)
  local names = feature_names or DEFAULT_FEATURE_NAMES
  local features = {}
  for i, name in ipairs(names) do
    features[name] = input_buf[i]
  end
  return features
end

--- Construct multiple batches from pairs and history as pair batches
---@param current_pairs table Current training pairs from user selection
---@param history table Training history (array of pairs)
---@param batch_size number Target number of PAIRS per batch
---@param num_batches number Number of batches to create
---@return table Array of pair batches
local function construct_batches(current_pairs, history, batch_size, num_batches)
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
---@return number Average loss across all batches
local function train_on_batches(st, batches, latency_ctx)
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
      local combined_activations, combined_pre_activations, combined_bn_cache, combined_dropout_masks = forward_pass(
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
        backward_pass_pairwise(
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
        update_parameters(
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

--- Ensure config is available on the given state table.
--- Config is always set via init_state() / create_instance().
---@param st table State table
---@return NosNNConfig The current configuration
local function ensure_config(st)
  if not st.config then
    error("NN algorithm not properly initialized - call create_instance() first")
  end
  return st.config
end

--- Prepare fused weights/biases for fast inference by folding batch normalization
--- into the weight matrices. This eliminates all batch norm computation at inference time.
--- Also transposes weight matrices for cache-friendly access patterns.
---@param st table State table
local function prepare_inference_cache(st)
  if not st.weights or #st.weights == 0 then
    return
  end

  local num_layers = #st.weights
  local weights_t = {}
  local biases = {}
  local buffers = {}

  for i = 1, num_layers do
    local w = st.weights[i]
    local b = st.biases[i]
    local in_size = #w
    local out_size = #w[1]

    if i < num_layers and st.gammas and st.gammas[i] and st.betas and st.betas[i] then
      -- Hidden layer: fuse batch norm into weights and biases
      local gamma = st.gammas[i][1]
      local beta = st.betas[i][1]
      local mean = st.running_means[i][1]
      local var = st.running_vars[i][1]

      -- Compute scale and build transposed fused weight matrix + flat fused bias
      local wt_layer = {}
      local b_layer = {}
      for j = 1, out_size do
        local scale = gamma[j] / math.sqrt(var[j] + 1e-8)
        b_layer[j] = scale * (b[1][j] - mean[j]) + beta[j]
        local wt_j = {}
        for k = 1, in_size do
          wt_j[k] = w[k][j] * scale
        end
        wt_layer[j] = wt_j
      end
      weights_t[i] = wt_layer
      biases[i] = b_layer
    else
      -- Output layer (or layer without batch norm): transpose weights, flatten bias
      local wt_layer = {}
      local b_layer = {}
      for j = 1, out_size do
        b_layer[j] = b[1][j]
        local wt_j = {}
        for k = 1, in_size do
          wt_j[k] = w[k][j]
        end
        wt_layer[j] = wt_j
      end
      weights_t[i] = wt_layer
      biases[i] = b_layer
    end

    -- Pre-allocate output buffer for this layer
    local buf = {}
    for j = 1, out_size do
      buf[j] = 0
    end
    buffers[i] = buf
  end

  -- Pre-allocate input buffer
  local input_size = #st.weights[1] -- rows of first weight matrix = input features
  local input_buf = {}
  for j = 1, input_size do
    input_buf[j] = 0
  end

  -- Precompute input sizes per layer (eliminates #current in hot loop)
  local input_sizes = {}
  input_sizes[1] = input_size
  for i = 2, num_layers do
    input_sizes[i] = #biases[i - 1]
  end

  st.inference_cache = {
    weights_t = weights_t,
    biases = biases,
    buffers = buffers,
    input_buf = input_buf,
    num_layers = num_layers,
    input_sizes = input_sizes,
  }
end

--- Migrate the first layer when saved weights have fewer inputs than the current architecture.
--- Expands weight rows with Xavier initialization and backfills training history using
--- MIGRATION_DEFAULTS (static values or per-row heuristic functions keyed by feature name).
---@param st table State table (must have st.weights loaded)
---@param config table NN config with architecture
---@param training_history table? Training history pairs to backfill
---@return { migrated: boolean, output_size: number }
local function migrate_input_size(st, config, training_history)
  if not (st.weights and st.weights[1]) then
    return { migrated = false, output_size = 0 }
  end

  local saved_input_size = #st.weights[1]
  local expected_input_size = config.architecture[1]

  if saved_input_size >= expected_input_size then
    return { migrated = false, output_size = 0 }
  end

  local output_size = #st.weights[1][1]

  -- Append new rows with Xavier-scale random values for each new input
  for _ = saved_input_size + 1, expected_input_size do
    local new_row = nn_core.xavier_init(1, output_size, expected_input_size)[1]
    st.weights[1][#st.weights[1] + 1] = new_row
  end

  -- Backfill training history: expand existing pairs to match new input size
  -- Training history stores inputs as matrices: { {v1, v2, ..., vN} }
  -- so we must index into [1] (the inner row vector) for length checks and mutations
  if training_history then
    local feature_names = get_feature_names(st)

    -- Build name->index map for OLD features (what the row already contains)
    local old_feature_indices = {}
    for i = 1, saved_input_size do
      old_feature_indices[feature_names[i]] = i
    end

    -- Resolve static defaults once; collect functions for per-row evaluation
    local new_col_defaults = {}
    local new_col_fns = {}
    for col = saved_input_size + 1, expected_input_size do
      local default = MIGRATION_DEFAULTS[feature_names[col]]
      if type(default) == "function" then
        new_col_fns[col] = default
        new_col_defaults[col] = 0.0 -- placeholder, overwritten per-row
      else
        new_col_defaults[col] = default or 0.0
      end
    end

    local has_fns = next(new_col_fns) ~= nil

    local function backfill_row(row)
      for col = saved_input_size + 1, expected_input_size do
        if has_fns and new_col_fns[col] then
          row[col] = new_col_fns[col](row, old_feature_indices)
        else
          row[col] = new_col_defaults[col]
        end
      end
    end

    for _, pair in ipairs(training_history) do
      local pos_row = pair.positive_input and pair.positive_input[1]
      if pos_row and #pos_row == saved_input_size then
        backfill_row(pos_row)
      end
      local neg_row = pair.negative_input and pair.negative_input[1]
      if neg_row and #neg_row == saved_input_size then
        backfill_row(neg_row)
      end
    end
  end

  vim.notify(
    string.format(
      "neural-open: Migrated NN input layer from %d to %d features. "
        .. "First-layer weights expanded, optimizer moments reset.",
      saved_input_size,
      expected_input_size
    ),
    vim.log.levels.INFO
  )

  return { migrated = true, output_size = output_size }
end

--- Save the current NN state to disk.
---@param st table State table
---@param latency_ctx? table Optional latency context
local function save_state(st, latency_ctx)
  local weights_module = require("neural-open.weights")
  weights_module.save_weights("nn", {
    version = "2.0-hinge",
    network = {
      weights = st.weights,
      biases = st.biases,
      gammas = st.gammas,
      betas = st.betas,
      running_means = st.running_means,
      running_vars = st.running_vars,
    },
    training_history = st.training_history,
    stats = st.stats,
    optimizer_type = st.optimizer_type,
    optimizer_state = st.optimizer_state,
  }, latency_ctx, st.config and st.config.picker_name)
end

--- Ensure the network state is initialized
---@param st table State table
---@param force_reload boolean? Force reload weights even if already loaded
local function ensure_weights(st, force_reload)
  if not st.weights or force_reload then
    -- Invalidate inference cache when reloading weights
    st.inference_cache = nil

    -- Ensure config is loaded
    local config = ensure_config(st)

    -- Try to load from storage first
    local weights_module = require("neural-open.weights")
    local algorithm_weights = weights_module.get_weights("nn", st.config and st.config.picker_name)

    if algorithm_weights then
      -- Auto-migrate from old double-nested format: { nn = { version, network, ... } }
      -- to flat format: { version, network, ... }
      if algorithm_weights.nn and algorithm_weights.nn.network then
        algorithm_weights = algorithm_weights.nn
      end

      -- Load network weights (flat: algorithm_weights.network)
      if algorithm_weights.network then
        st.weights = algorithm_weights.network.weights
        st.biases = algorithm_weights.network.biases
        st.gammas = algorithm_weights.network.gammas
        st.betas = algorithm_weights.network.betas
        st.running_means = algorithm_weights.network.running_means
        st.running_vars = algorithm_weights.network.running_vars
      end

      -- Handle input-size migration: expand first layer if config expects more inputs
      local migration = migrate_input_size(st, config, algorithm_weights.training_history)

      -- Load history and stats
      st.training_history = algorithm_weights.training_history or {}
      if algorithm_weights.stats then
        st.stats = vim.tbl_extend("force", st.stats, algorithm_weights.stats)
        st.stats.batch_timings = st.stats.batch_timings or {}
        st.stats.avg_batch_timing = st.stats.avg_batch_timing or nil
        st.stats.loss_history = st.stats.loss_history or {}
      end

      -- Load optimizer state
      local saved_optimizer_type = algorithm_weights.optimizer_type or "sgd"
      local current_optimizer_type = config.optimizer or "sgd"

      if saved_optimizer_type ~= current_optimizer_type then
        -- Optimizer changed - reset optimizer state but keep network weights
        st.optimizer_type = current_optimizer_type
        st.optimizer_state = nil
        vim.notify(
          "neural-open: Optimizer changed to "
            .. current_optimizer_type
            .. ". Optimizer state reset, but network weights preserved.",
          vim.log.levels.INFO
        )
      else
        st.optimizer_type = saved_optimizer_type
        st.optimizer_state = algorithm_weights.optimizer_state
      end

      -- Initialize optimizer state if needed
      if st.optimizer_type == "adamw" and not st.optimizer_state then
        st.optimizer_state = init_optimizer_state("adamw", config.architecture)
      elseif st.optimizer_type == "sgd" and not st.optimizer_state then
        st.optimizer_state = { timestep = 0 }
      end

      -- Reset first-layer optimizer moments after input-size migration.
      -- This must run after optimizer state is loaded from disk.
      if migration.migrated and st.optimizer_state and st.optimizer_state.moments then
        local expected_input_size = config.architecture[1]
        local m = st.optimizer_state.moments
        if m.first and m.first.weights and m.first.weights[1] then
          m.first.weights[1] = nn_core.zeros(expected_input_size, migration.output_size)
        end
        if m.second and m.second.weights and m.second.weights[1] then
          m.second.weights[1] = nn_core.zeros(expected_input_size, migration.output_size)
        end
      end

      -- Persist migrated weights immediately so future loads don't re-trigger migration
      if migration.migrated then
        save_state(st)
      end
    end

    -- If still no weights, try loading from bundled defaults
    if not st.weights then
      -- Default weights architecture (must match nn_default_weights.lua)
      local default_architecture = { 11, 16, 16, 8, 1 }
      local architecture_matches = vim.deep_equal(config.architecture, default_architecture)

      local default_weights = nil
      if architecture_matches then
        local ok, loaded = pcall(require, "neural-open.algorithms.nn_default_weights")
        if ok and loaded and loaded.network then
          default_weights = loaded
        end
      end

      if default_weights then
        st.weights = default_weights.network.weights
        st.biases = default_weights.network.biases
        st.gammas = default_weights.network.gammas
        st.betas = default_weights.network.betas
        st.running_means = default_weights.network.running_means
        st.running_vars = default_weights.network.running_vars
      else
        -- Fallback to random initialization if defaults unavailable or architecture differs
        st.weights, st.biases, st.gammas, st.betas, st.running_means, st.running_vars =
          nn_core.init_network(config.architecture)
      end
      st.training_history = {}

      -- Set optimizer type from config
      st.optimizer_type = config.optimizer or "sgd"
      if st.optimizer_type == "adamw" then
        st.optimizer_state = init_optimizer_state("adamw", config.architecture)
      else
        -- SGD needs state for timestep tracking (warmup)
        st.optimizer_state = { timestep = 0 }
      end
    end

    -- Ensure running statistics exist (initialize if missing from loaded weights)
    if not st.running_means or not st.running_vars then
      local _, _, _, _, running_means, running_vars = nn_core.init_network(config.architecture)
      st.running_means = running_means
      st.running_vars = running_vars
    end

    -- Build fused inference cache for fast calculate_score()
    prepare_inference_cache(st)
  end
end

--- Implementation: Calculate score using neural network from a flat input buffer
---@param st table State table
---@param input_buf number[] Flat array of normalized features in canonical order
---@return number Score in [0, 100]
local function calculate_score_impl(st, input_buf)
  -- Lazy-load weights if not yet initialized (hot path skips this after first call)
  if not st.inference_cache then
    ensure_weights(st)
  end

  if not st.inference_cache then
    -- Fallback to general forward_pass when inference cache unavailable (e.g., empty/invalid weights)
    local input = features_to_input(input_buf, false, st.match_idx, st.virtual_name_idx)
    local activations = forward_pass(
      input,
      st.weights,
      st.biases,
      st.gammas,
      st.betas,
      st.running_means,
      st.running_vars,
      false,
      nil,
      false
    )
    return activations[#activations][1][1] * 100
  end

  local cache = st.inference_cache --[[@as table]]
  local num_layers = cache.num_layers
  local input_sizes = cache.input_sizes
  local current = input_buf -- Use directly as first-layer input (read-only)

  for layer = 1, num_layers do
    local wt = cache.weights_t[layer]
    local b = cache.biases[layer]
    local buf = cache.buffers[layer]
    local out_size = #b
    local in_size = input_sizes[layer]

    for j = 1, out_size do
      local wt_j = wt[j]
      local sum = b[j]
      for k = 1, in_size do
        sum = sum + current[k] * wt_j[k]
      end

      if layer < num_layers then
        -- Leaky ReLU inline (alpha=0.01)
        buf[j] = sum > 0 and sum or 0.01 * sum
      else
        -- Sigmoid inline for output layer
        if sum < -500 then
          buf[j] = 0
        elseif sum > 500 then
          buf[j] = 1
        else
          buf[j] = 1 / (1 + math_exp(-sum))
        end
      end
    end

    current = buf
  end

  return current[1] * 100
end

--- Implementation: Update neural network weights based on user selection
---@param st table State table
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@param latency_ctx? table Optional latency context
local function update_weights_impl(st, selected_item, ranked_items, latency_ctx)
  local latency = require("neural-open.latency")
  ensure_weights(st, true)

  local config = ensure_config(st)

  -- Find selected item's rank
  local selected_rank = 1
  local selected_id = scorer.get_item_identity(selected_item)
  for i, item in ipairs(ranked_items) do
    local id = scorer.get_item_identity(item)
    if selected_id and id and selected_id == id then
      selected_rank = i
      break
    end
  end

  -- Measure pair construction
  local pairs = latency.measure(latency_ctx, "nn.pair_construction", function()
    local pairs_result = {}
    -- Get match_dropout configuration
    local match_dropout_rate = config.match_dropout or 0.25

    -- Check if selected item has input_buf
    if not (selected_item.nos and selected_item.nos.input_buf) then
      return pairs_result -- Cannot train without features
    end

    local positive_input_buf = selected_item.nos.input_buf

    -- Collect top-10 items as hard negatives, excluding the selected item
    local candidate_pool = {}
    for _, item in ipairs(ranked_items) do
      -- Skip the selected item itself
      local item_id = scorer.get_item_identity(item)
      if not selected_id or not item_id or item_id ~= selected_id then
        table.insert(candidate_pool, item)
        -- Stop when we have 10 hard negatives
        if #candidate_pool >= 10 then
          break
        end
      end
    end

    -- Create pairs: selected vs. all hard negatives
    for _, neg_item in ipairs(candidate_pool) do
      local neg_input_buf = neg_item.nos and neg_item.nos.input_buf
      if neg_input_buf then
        -- Decide once per pair whether to drop match features (applies to both positive and negative)
        local drop_match = math.random() < match_dropout_rate
        table.insert(pairs_result, {
          positive_input = features_to_input(positive_input_buf, drop_match, st.match_idx, st.virtual_name_idx),
          negative_input = features_to_input(neg_input_buf, drop_match, st.match_idx, st.virtual_name_idx),
          positive_file = selected_item.nos.normalized_path or selected_item.nos.item_id,
          negative_file = neg_item.nos.normalized_path or neg_item.nos.item_id,
        })
        st.stats.samples_processed = st.stats.samples_processed + 1
      end
    end

    return pairs_result
  end, "async.weight_update")

  latency.add_metadata(latency_ctx, "nn.pair_construction", {
    pairs_created = #pairs,
    selected_rank = selected_rank,
  })

  -- Measure batch construction
  local batches = latency.measure(latency_ctx, "nn.batch_construction", function()
    -- Construct batches BEFORE adding current pairs to history to avoid duplicates
    -- This ensures the first batch uses current pairs + old history (not current pairs twice)
    return construct_batches(pairs, st.training_history, config.batch_size, config.batches_per_update)
  end, "async.weight_update")

  latency.add_metadata(latency_ctx, "nn.batch_construction", {
    num_batches = #batches,
    history_size = #st.training_history,
  })

  -- Add current pairs to history (they go at the end as newest)
  for _, pair in ipairs(pairs) do
    table.insert(st.training_history, pair)
  end

  -- Maintain history size limit by removing oldest pairs
  while #st.training_history > config.history_size do
    table.remove(st.training_history, 1) -- Remove oldest (from beginning)
  end

  -- Train the network on all batches
  if #batches > 0 then
    latency.start(latency_ctx, "nn.training", "async.weight_update")
    train_on_batches(st, batches, latency_ctx)
    latency.finish(latency_ctx, "nn.training")

    -- Rebuild inference cache after training modified weights/batch norm parameters
    prepare_inference_cache(st)

    -- Inject training phase metrics from existing stats (post-hoc metadata)
    if st.stats.avg_batch_timing then
      latency.add_metadata(latency_ctx, "nn.training", {
        num_batches = #batches,
        total_pairs = math.floor(st.stats.samples_per_batch * #batches),
        avg_forward_ms = st.stats.avg_batch_timing.forward_ms,
        avg_backward_ms = st.stats.avg_batch_timing.backward_ms,
        avg_update_ms = st.stats.avg_batch_timing.update_ms,
        avg_loss = st.stats.last_loss,
        optimizer = st.optimizer_type,
      })
    end
  end

  -- Save weights (even if no training happened, state may have changed)
  latency.start(latency_ctx, "nn.save_weights", "async.weight_update")
  save_state(st, latency_ctx)
  latency.finish(latency_ctx, "nn.save_weights")
end

--- Calculate loss averages from history
---@param st table State table
---@return table<number, number?> averages Table with keys 1, 10, 100, 1000
local function calculate_loss_averages(st)
  local history = st.stats and st.stats.loss_history
  if not history or #history == 0 then
    return {}
  end

  local averages = {}
  local history_size = #history

  -- Calculate averages for different window sizes
  -- Only calculate averages when we have enough samples to make them meaningful
  local windows = { 1, 10, 100, 1000 }

  for _, window_size in ipairs(windows) do
    if history_size >= math.min(window_size, 2) then
      -- For window size 1, just use the last value
      if window_size == 1 then
        averages[1] = history[history_size]
      else
        -- Calculate average for the window (or all available if less than window)
        local actual_window = math.min(window_size, history_size)
        local start_idx = history_size - actual_window + 1
        local sum = 0
        for i = start_idx, history_size do
          sum = sum + history[i]
        end
        averages[actual_window] = sum / actual_window
      end
    end
  end

  return averages
end

--- Calculate ranking accuracy averages from history
---@param st table State table
---@return table<number, {correct_pct: number, margin_pct: number}> averages Table with keys 1, 10, 100, 1000
local function calculate_accuracy_averages(st)
  local history = st.stats and st.stats.ranking_accuracy_history
  if not history or #history == 0 then
    return {}
  end

  local averages = {}
  local history_size = #history
  local windows = { 1, 10, 100, 1000 }

  for _, window_size in ipairs(windows) do
    if history_size >= math.min(window_size, 2) then
      if window_size == 1 then
        -- For window size 1, use the last value
        local last = history[history_size]
        if last.total > 0 then
          averages[1] = {
            correct_pct = (last.correct / last.total) * 100,
            margin_pct = (last.margin_correct / last.total) * 100,
          }
        else
          averages[1] = { correct_pct = 0, margin_pct = 0 }
        end
      else
        -- Calculate average for the window
        local actual_window = math.min(window_size, history_size)
        local start_idx = history_size - actual_window + 1

        local total_correct = 0
        local total_margin_correct = 0
        local total_pairs = 0

        for i = start_idx, history_size do
          total_correct = total_correct + history[i].correct
          total_margin_correct = total_margin_correct + history[i].margin_correct
          total_pairs = total_pairs + history[i].total
        end

        if total_pairs > 0 then
          averages[actual_window] = {
            correct_pct = (total_correct / total_pairs) * 100,
            margin_pct = (total_margin_correct / total_pairs) * 100,
          }
        else
          averages[actual_window] = { correct_pct = 0, margin_pct = 0 }
        end
      end
    end
  end

  return averages
end

local fmt = require("neural-open.debug_fmt")

--- Implementation: Generate debug view for neural network algorithm
---@param st table State table
---@param item NeuralOpenItem
---@param all_items NeuralOpenItem[]?
---@return string[], table[]
local function debug_view_impl(st, item, all_items)
  local lines = {}
  local hl = {}

  fmt.add_title(lines, hl, "Neural Network Algorithm")
  table.insert(lines, "")

  local config = ensure_config(st)
  fmt.add_label(lines, hl, "Architecture", table.concat(config.architecture, " -> "))
  fmt.add_label(lines, hl, "Learning Rate", string.format("%.4f", config.learning_rate))
  fmt.add_label(lines, hl, "Weight Decay", string.format("%.6f", config.weight_decay or 0))
  fmt.add_label(lines, hl, "Margin", string.format("%.2f", config.margin or 1.0))

  -- Optimizer information
  local optimizer_name = st.optimizer_type == "adamw" and "AdamW" or "SGD"
  fmt.add_label(lines, hl, "Optimizer", optimizer_name)

  -- Step counter with warmup info
  if st.optimizer_state then
    local current_timestep = st.optimizer_state.timestep or 0
    local warmup_steps = config.warmup_steps or 0
    local warmup_start_factor = config.warmup_start_factor or 0.1

    if warmup_steps > 0 and current_timestep > 0 and current_timestep <= warmup_steps then
      local warmup_factor = calculate_warmup_factor(current_timestep, warmup_steps, warmup_start_factor)
      fmt.add_label(
        lines,
        hl,
        "Step",
        string.format("%d/%d (warmup, LR factor: %.1f%%)", current_timestep, warmup_steps, warmup_factor * 100)
      )
    else
      fmt.add_label(lines, hl, "Step", string.format("%d", current_timestep))
    end

    if st.optimizer_type == "adamw" then
      fmt.add_label(
        lines,
        hl,
        "Beta1/Beta2",
        string.format("%.3f / %.3f", config.adam_beta1 or 0.9, config.adam_beta2 or 0.999),
        4
      )
    end
  end

  -- AdamW moment statistics
  if st.optimizer_type == "adamw" and st.optimizer_state then
    if st.optimizer_state.moments and st.optimizer_state.moments.first.weights[1] then
      local m_w = st.optimizer_state.moments.first.weights[1]
      local v_w = st.optimizer_state.moments.second.weights[1]

      local m_sum, v_sum, count = 0, 0, 0
      for i = 1, #m_w do
        for j = 1, #m_w[i] do
          m_sum = m_sum + math.abs(m_w[i][j])
          v_sum = v_sum + v_w[i][j]
          count = count + 1
        end
      end

      if count > 0 then
        fmt.add_label(
          lines,
          hl,
          "Avg Moments (L1)",
          string.format("1st: %.6f, 2nd: %.6f", m_sum / count, v_sum / count),
          4
        )
      end
    end
  end

  -- Dropout configuration
  if config.dropout_rates and #config.dropout_rates > 0 then
    local dropout_str = {}
    for i, rate in ipairs(config.dropout_rates) do
      table.insert(dropout_str, string.format("L%d: %.1f%%", i, rate * 100))
    end
    fmt.add_label(lines, hl, "Dropout Rates", table.concat(dropout_str, ", "))

    if st.stats.dropout_active_rates and next(st.stats.dropout_active_rates) then
      local active_str = {}
      for i, rate in pairs(st.stats.dropout_active_rates) do
        if rate ~= nil and rate > 0 then
          table.insert(active_str, string.format("L%d: %.1f%%", i, rate))
        end
      end
      if #active_str > 0 then
        fmt.add_label(lines, hl, "Active Neurons (last batch)", table.concat(active_str, ", "))
      end
    end
  end

  -- Match dropout configuration
  if config.match_dropout and config.match_dropout > 0 then
    local dropout_desc = st.virtual_name_idx and "match/virtual_name features" or "match features"
    fmt.add_label(lines, hl, "Match Dropout", string.format("%.1f%% (%s)", config.match_dropout * 100, dropout_desc))
  end

  table.insert(lines, "")

  -- Training statistics
  fmt.add_title(lines, hl, "Training Statistics")
  table.insert(lines, "")

  -- Training metrics table (loss, accuracy, margin accuracy) -- shown first
  local loss_averages = calculate_loss_averages(st)
  local accuracy_averages = calculate_accuracy_averages(st)
  local loss_history_size = st.stats.loss_history and #st.stats.loss_history or 0
  local accuracy_history_size = st.stats.ranking_accuracy_history and #st.stats.ranking_accuracy_history or 0

  if loss_history_size > 0 or accuracy_history_size > 0 then
    local windows = { 1, 10, 100, 1000 }
    local shown_windows = {}
    for _, w in ipairs(windows) do
      local has_loss = loss_averages[w] or (w > loss_history_size and loss_averages[loss_history_size])
      local has_acc = accuracy_averages[w] or (w > accuracy_history_size and accuracy_averages[accuracy_history_size])
      if has_loss or has_acc then
        table.insert(shown_windows, w)
      end
    end

    if #shown_windows > 0 then
      -- Build metrics table dynamically based on available windows
      local header = { "" }
      for _, w in ipairs(shown_windows) do
        table.insert(header, w == 1 and "Last" or string.format("Avg %d", w))
      end

      local metrics_rows = { header }

      -- Loss row
      local loss_row = { "Loss:" }
      for _, w in ipairs(shown_windows) do
        local val = loss_averages[w] or loss_averages[loss_history_size]
        table.insert(loss_row, val and string.format("%.4f", val) or "-")
      end
      table.insert(metrics_rows, loss_row)

      -- Accuracy rows
      if accuracy_history_size > 0 then
        local acc_row = { "Accuracy:" }
        local margin_row = { "Margin:" }
        for _, w in ipairs(shown_windows) do
          local val = accuracy_averages[w] or accuracy_averages[accuracy_history_size]
          table.insert(acc_row, val and string.format("%.1f%%", val.correct_pct) or "-")
          table.insert(margin_row, val and string.format("%.1f%%", val.margin_pct) or "-")
        end
        table.insert(metrics_rows, acc_row)
        table.insert(metrics_rows, margin_row)
      end

      fmt.format_table(lines, hl, metrics_rows)
      table.insert(lines, "")
    end
  else
    fmt.add_label(lines, hl, "Last Hinge Loss", string.format("%.6f", st.stats.last_loss or 0))
  end

  fmt.add_label(lines, hl, "Samples Processed", string.format("%d", st.stats.samples_processed or 0))
  fmt.add_label(lines, hl, "Batches Trained", string.format("%d", st.stats.batches_trained or 0))

  fmt.add_label(
    lines,
    hl,
    "History Size",
    string.format("%d/%d pairs", st.training_history and #st.training_history or 0, config.history_size)
  )
  fmt.add_label(lines, hl, "Training Mode", "Pairwise Ranking (Hinge Loss)")

  -- Batch timing statistics
  local avg_timing = st.stats and st.stats.avg_batch_timing
  if avg_timing and avg_timing.total_ms then
    fmt.add_label(
      lines,
      hl,
      "Avg Batch Time (last 10)",
      string.format(
        "%.2fms (fwd: %.2fms, back: %.2fms, upd: %.2fms)",
        avg_timing.total_ms,
        avg_timing.forward_ms or 0,
        avg_timing.backward_ms or 0,
        avg_timing.update_ms or 0
      )
    )
  end

  -- Loss interpretation
  if st.stats.last_loss and st.stats.last_loss > 0 then
    fmt.add_label(lines, hl, "Loss Interpretation", string.format("Avg margin violation of %.4f", st.stats.last_loss))
    table.insert(lines, string.format("      (Loss=0 means all pairs satisfy margin of %.2f)", config.margin or 1.0))
  end

  table.insert(lines, "")

  if item.nos and item.nos.neural_score then
    fmt.add_label(lines, hl, "Current Score", string.format("%.4f", item.nos.neural_score))
  end

  -- Combined feature table (importance + current values), sorted by importance
  local feature_names = get_feature_names(st)
  local normalized_features = nil
  if item.nos and item.nos.input_buf then
    normalized_features = input_buf_to_features(item.nos.input_buf, feature_names)
  end

  if st.weights and st.weights[1] then
    local feature_order = feature_names

    -- Compute importance (L1 norm of first-layer weights per feature)
    local feature_data = {}
    for i, name in ipairs(feature_order) do
      local weight_sum = 0
      for j = 1, #st.weights[1][i] do
        weight_sum = weight_sum + math.abs(st.weights[1][i][j])
      end
      table.insert(feature_data, {
        name = name,
        importance = weight_sum / #st.weights[1][i],
        value = normalized_features and normalized_features[name] or nil,
      })
    end

    table.sort(feature_data, function(a, b)
      return a.importance > b.importance
    end)

    local feature_rows = { { "Features:", "Weight", "Value" } }
    for _, f in ipairs(feature_data) do
      table.insert(feature_rows, {
        fmt.format_feature_name(f.name),
        string.format("%.4f", f.importance),
        f.value and string.format("%.4f", f.value) or "-",
      })
    end

    table.insert(lines, "")
    fmt.format_table(lines, hl, feature_rows)
    table.insert(lines, "")
  elseif normalized_features then
    fmt.append_feature_value_table(lines, hl, normalized_features)
  end

  -- Network prediction
  if normalized_features and st.weights then
    fmt.add_title(lines, hl, "Network Prediction")
    table.insert(lines, "")

    -- No match dropout during inference/debug
    local input = features_to_input(item.nos.input_buf, false, st.match_idx, st.virtual_name_idx)

    -- Get logit output for debug info
    local activations_logit = forward_pass(
      input,
      st.weights,
      st.biases,
      st.gammas,
      st.betas,
      st.running_means,
      st.running_vars,
      false, -- inference mode
      nil, -- no dropout
      true -- return logits
    )
    local logit = activations_logit[#activations_logit][1][1]

    -- Get sigmoid output for probability
    local activations = forward_pass(
      input,
      st.weights,
      st.biases,
      st.gammas,
      st.betas,
      st.running_means,
      st.running_vars,
      false, -- inference mode
      nil, -- no dropout
      false -- return sigmoid
    )

    -- Show activation patterns
    for i = 2, #activations do
      local layer_name = i < #activations and string.format("Hidden Layer %d", i - 1) or "Output"
      local activation = activations[i][1]

      if i < #activations then
        local active_count = 0
        for j = 1, #activation do
          if activation[j] > 0 then
            active_count = active_count + 1
          end
        end
        fmt.add_label(lines, hl, layer_name, string.format("%d/%d neurons active", active_count, #activation))

        if st.gammas and st.gammas[i - 1] then
          fmt.add_label(
            lines,
            hl,
            "BatchNorm",
            string.format("enabled (mean: %.4f, bias: %.4f)", st.gammas[i - 1][1][1], st.betas[i - 1][1][1]),
            4
          )
        end
      else
        fmt.add_label(lines, hl, layer_name .. " Logit", string.format("%.4f", logit))
        fmt.add_label(lines, hl, layer_name .. " Probability", string.format("%.4f (sigmoid)", activation[1]))
      end
    end
  end

  -- Weight statistics
  if st.stats.weight_norms and #st.stats.weight_norms > 0 then
    table.insert(lines, "")
    fmt.add_title(lines, hl, "Weight Statistics")
    table.insert(lines, "")
    for i = 1, #st.stats.weight_norms do
      local layer_name = i < #st.stats.weight_norms and string.format("Layer %d", i) or "Output Layer"
      fmt.add_label(lines, hl, layer_name, "")
      fmt.add_label(lines, hl, "L2 Norm", string.format("%.4f", st.stats.weight_norms[i] or 0), 4)
      fmt.add_label(lines, hl, "Avg Magnitude", string.format("%.4f", st.stats.avg_weight_magnitudes[i] or 0), 4)
    end
  end

  return lines, hl
end

--- Implementation: Load the latest weights
---@param st table State table
local function load_weights_impl(st)
  ensure_weights(st, true)
end

--- Create a per-picker instance with isolated state
---@param config NosNNConfig
---@return Algorithm
function M.create_instance(config)
  local picker_name = config.picker_name or "__default__"
  local st = states[picker_name]
  if not st then
    st = new_state()
    states[picker_name] = st
  end
  init_state(st, config)
  local instance = {
    calculate_score = function(input_buf)
      return calculate_score_impl(st, input_buf)
    end,
    update_weights = function(selected_item, ranked_items, latency_ctx)
      return update_weights_impl(st, selected_item, ranked_items, latency_ctx)
    end,
    load_weights = function()
      return load_weights_impl(st)
    end,
    debug_view = function(item, all_items)
      return debug_view_impl(st, item, all_items)
    end,
    get_name = function()
      return "nn"
    end,
    init = function() end, -- no-op; config already set via init_state
  }
  ---@diagnostic disable-next-line: undefined-field
  if _G._TEST then
    instance._get_training_history = function()
      return st.training_history
    end
    instance._get_weights = function()
      return st.weights
    end
    instance._get_stats = function()
      return st.stats
    end
    instance._get_optimizer_state = function()
      return st.optimizer_state
    end
    instance._forward_pass = function(input)
      return forward_pass(
        input,
        st.weights,
        st.biases,
        st.gammas,
        st.betas,
        st.running_means,
        st.running_vars,
        false, -- inference mode
        nil, -- no dropout
        false, -- return sigmoid output
        st.stats
      )
    end
    instance._features_to_input = features_to_input
  end
  return instance
end

-- Testing helpers (only available in test environment)
---@diagnostic disable-next-line: undefined-field
if _G._TEST then
  function M._reset_states()
    states = {}
  end
end

return M
