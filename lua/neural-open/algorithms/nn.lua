--- Neural Network scoring algorithm
local nn_core = require("neural-open.algorithms.nn_core")
local math_exp = math.exp
local M = {}

-- Module state
local state = {
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
  return_logits
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
        if state and state.stats then
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
          if not state.stats.dropout_active_rates then
            state.stats.dropout_active_rates = {}
          end
          state.stats.dropout_active_rates[i] = (active_count / total_count) * 100
        end
      elseif state and state.stats and training then
        -- Initialize with 0 for layers without dropout to prevent nils
        if not state.stats.dropout_active_rates then
          state.stats.dropout_active_rates = {}
        end
        state.stats.dropout_active_rates[i] = 0
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
---@return table weight_gradients, table bias_gradients, table gamma_gradients, table beta_gradients
local function backward_pass_pairwise(
  activations,
  pre_activations,
  output_grad,
  weights,
  gammas,
  bn_cache,
  dropout_masks
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
        local dropout_rate = state.config.dropout_rates and state.config.dropout_rates[i] or 0
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
  if not state.optimizer_state then
    state.optimizer_state = { timestep = 0 }
  end

  -- Increment timestep
  state.optimizer_state.timestep = state.optimizer_state.timestep + 1
  local t = state.optimizer_state.timestep

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
  if state and state.stats then
    state.stats.weight_norms = {}
    state.stats.avg_weight_magnitudes = {}
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
      state.stats.weight_norms[i] = math.sqrt(sum_squared)
      state.stats.avg_weight_magnitudes[i] = sum_abs / count
    end
  end
end

--- Update weights using AdamW optimizer
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
  if not state.optimizer_state then
    state.optimizer_state = init_optimizer_state("adamw", config.architecture)
  end

  -- Increment timestep
  state.optimizer_state.timestep = state.optimizer_state.timestep + 1
  local t = state.optimizer_state.timestep

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
    local m_w = state.optimizer_state.moments.first.weights[i]
    local v_w = state.optimizer_state.moments.second.weights[i]

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
    state.optimizer_state.moments.first.weights[i] = m_w
    state.optimizer_state.moments.second.weights[i] = v_w

    -- Update biases
    local m_b = state.optimizer_state.moments.first.biases[i]
    local v_b = state.optimizer_state.moments.second.biases[i]

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

    state.optimizer_state.moments.first.biases[i] = m_b
    state.optimizer_state.moments.second.biases[i] = v_b

    -- Update batch norm parameters if present
    if gammas and gammas[i] and gamma_gradients and gamma_gradients[i] then
      local m_g = state.optimizer_state.moments.first.gammas[i]
      local v_g = state.optimizer_state.moments.second.gammas[i]

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

      state.optimizer_state.moments.first.gammas[i] = m_g
      state.optimizer_state.moments.second.gammas[i] = v_g
    end

    if betas and betas[i] and beta_gradients and beta_gradients[i] then
      local m_beta = state.optimizer_state.moments.first.betas[i]
      local v_beta = state.optimizer_state.moments.second.betas[i]

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

      state.optimizer_state.moments.first.betas[i] = m_beta
      state.optimizer_state.moments.second.betas[i] = v_beta
    end
  end

  -- Calculate and store weight statistics
  if state and state.stats then
    state.stats.weight_norms = {}
    state.stats.avg_weight_magnitudes = {}
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
      state.stats.weight_norms[i] = math.sqrt(sum_squared)
      state.stats.avg_weight_magnitudes[i] = sum_abs / count
    end
  end
end

--- Update weights using configured optimizer (dispatcher function)
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
  if state.optimizer_type == "adamw" then
    return update_parameters_adamw(
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

--- Canonical feature names in input order (shared from scorer)
local FEATURE_NAMES = require("neural-open.scorer").FEATURE_NAMES

--- Convert a flat input buffer to matrix format with optional match dropout
---@param input_buf number[] Flat array of normalized features in canonical order
---@param drop_match_features? boolean Whether to zero out match/virtual_name features
---@return table Input matrix
local function features_to_input(input_buf, drop_match_features)
  -- Copy is required: input_buf is a shared mutable buffer; dropout would corrupt it
  local input = {}
  for i = 1, #input_buf do
    input[i] = input_buf[i]
  end
  if drop_match_features then
    input[1] = 0 -- match
    input[2] = 0 -- virtual_name
  end
  return { input } -- nn_core matrix format: { {v1, v2, ...} }
end

--- Convert input_buf flat array to named features table (for debug/display only)
---@param input_buf number[] Flat array of normalized features in canonical order
---@return table<string, number>
local function input_buf_to_features(input_buf)
  local features = {}
  for i, name in ipairs(FEATURE_NAMES) do
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

  -- First batch always includes current pairs
  local first_batch_pairs = {}
  for _, pair in ipairs(current_pairs) do
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
---@param batches table Array of pair batches {pairs}
---@param latency_ctx table|nil Optional latency context for performance tracking
---@return number Average loss across all batches
local function train_on_batches(batches, latency_ctx)
  if not state.weights or #batches == 0 then
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
  state.stats.loss_history = state.stats.loss_history or {}

  -- Initialize ranking accuracy history if needed
  state.stats.ranking_accuracy_history = state.stats.ranking_accuracy_history or {}

  -- Get margin from config
  local margin = state.config.margin or 1.0

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
        state.weights,
        state.biases,
        state.gammas,
        state.betas,
        state.running_means,
        state.running_vars,
        true, -- training mode
        state.config.dropout_rates,
        true -- return logits for training
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
          state.weights,
          state.gammas,
          combined_bn_cache,
          combined_dropout_masks
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
          state.weights,
          state.biases,
          weight_grads_accumulated,
          bias_grads_accumulated,
          state.gammas,
          state.betas,
          gamma_grads_accumulated,
          beta_grads_accumulated,
          state.config.learning_rate,
          state.config
        )
        batch_timing.update_ms = (vim.loop.hrtime() - update_start) / 1e6
      else
        -- No gradients (all pairs had loss = 0)
        batch_timing.update_ms = 0
      end

      -- STORE TIMING in circular buffer
      table.insert(state.stats.batch_timings, batch_timing)
      if #state.stats.batch_timings > 10 then
        table.remove(state.stats.batch_timings, 1)
      end

      -- Calculate average timing
      if #state.stats.batch_timings > 0 then
        local avg_forward, avg_backward, avg_update = 0, 0, 0
        for _, timing in ipairs(state.stats.batch_timings) do
          avg_forward = avg_forward + timing.forward_ms
          avg_backward = avg_backward + timing.backward_ms
          avg_update = avg_update + timing.update_ms
        end
        local n = #state.stats.batch_timings
        state.stats.avg_batch_timing = {
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
      state.stats.batches_trained = state.stats.batches_trained + 1

      -- Add individual batch loss to history
      table.insert(state.stats.loss_history, batch_loss)
      if #state.stats.loss_history > 1000 then
        table.remove(state.stats.loss_history, 1)
      end

      -- Track ranking accuracy history
      table.insert(state.stats.ranking_accuracy_history, {
        correct = batch_correct,
        margin_correct = batch_margin_correct,
        total = batch_size,
      })
      if #state.stats.ranking_accuracy_history > 1000 then
        table.remove(state.stats.ranking_accuracy_history, 1)
      end
    end
  end

  -- Update statistics
  if #batches > 0 then
    state.stats.last_loss = total_loss / #batches
    state.stats.samples_per_batch = total_pairs / #batches
  else
    state.stats.last_loss = 0
    state.stats.samples_per_batch = 0
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

  return state.stats.last_loss
end

--- Ensure config is loaded from the main module
--- Config comes from init.lua (single source of truth) via registry.get_algorithm()
---@return NosNNConfig The current configuration
local function ensure_config()
  if not state.config then
    -- Load config from main module (user config already merged with defaults in init.lua)
    local main_config = require("neural-open").config
    local algo_config = main_config.algorithm_config and main_config.algorithm_config.nn or {}
    state.config = vim.deepcopy(algo_config)

    -- Validate critical fields exist
    if not state.config.architecture then
      error("NN algorithm not properly initialized - missing configuration")
    end
  end
  return state.config
end

--- Prepare fused weights/biases for fast inference by folding batch normalization
--- into the weight matrices. This eliminates all batch norm computation at inference time.
--- Also transposes weight matrices for cache-friendly access patterns.
local function prepare_inference_cache()
  if not state.weights or #state.weights == 0 then
    return
  end

  local num_layers = #state.weights
  local weights_t = {}
  local biases = {}
  local buffers = {}

  for i = 1, num_layers do
    local w = state.weights[i]
    local b = state.biases[i]
    local in_size = #w
    local out_size = #w[1]

    if i < num_layers and state.gammas and state.gammas[i] and state.betas and state.betas[i] then
      -- Hidden layer: fuse batch norm into weights and biases
      local gamma = state.gammas[i][1]
      local beta = state.betas[i][1]
      local mean = state.running_means[i][1]
      local var = state.running_vars[i][1]

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
  local input_size = #state.weights[1] -- rows of first weight matrix = input features
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

  state.inference_cache = {
    weights_t = weights_t,
    biases = biases,
    buffers = buffers,
    input_buf = input_buf,
    num_layers = num_layers,
    input_sizes = input_sizes,
  }
end

--- Ensure the network state is initialized
---@param force_reload boolean? Force reload weights even if already loaded
local function ensure_weights(force_reload)
  if not state.weights or force_reload then
    -- Invalidate inference cache when reloading weights
    state.inference_cache = nil

    -- Ensure config is loaded
    local config = ensure_config()

    -- Try to load from storage first
    local weights_module = require("neural-open.weights")
    local algorithm_weights = weights_module.get_weights("nn")

    -- Detect and handle migration from old BCE format to new pairwise hinge loss format
    local needs_migration = false
    local old_version = false

    if algorithm_weights and algorithm_weights.nn then
      -- Check for version field
      local version = algorithm_weights.nn.version

      if not version or version == "1.0-bce" then
        -- Old BCE format detected
        old_version = true
        needs_migration = true
      elseif version ~= "2.0-hinge" then
        -- Unknown version, treat as needing migration to be safe
        needs_migration = true
      end
      -- If version == "2.0-hinge", no migration needed (needs_migration stays false)

      -- Load network weights (preserve learned weights)
      if algorithm_weights.nn.network then
        state.weights = algorithm_weights.nn.network.weights
        state.biases = algorithm_weights.nn.network.biases
        state.gammas = algorithm_weights.nn.network.gammas
        state.betas = algorithm_weights.nn.network.betas
        state.running_means = algorithm_weights.nn.network.running_means
        state.running_vars = algorithm_weights.nn.network.running_vars
      end

      -- Handle input-size migration: expand first layer if config expects more inputs
      local input_size_migrated = false
      local migrated_output_size = 0
      if state.weights and state.weights[1] then
        local saved_input_size = #state.weights[1]
        local expected_input_size = config.architecture[1]
        if saved_input_size ~= expected_input_size and saved_input_size < expected_input_size then
          input_size_migrated = true
          local output_size = #state.weights[1][1]
          migrated_output_size = output_size
          -- Append new rows with Xavier-scale random values for each new input
          for _ = saved_input_size + 1, expected_input_size do
            local new_row = nn_core.xavier_init(1, output_size, expected_input_size)[1]
            state.weights[1][#state.weights[1] + 1] = new_row
          end

          -- Note: first-layer optimizer moments are reset later, after optimizer state
          -- is loaded from disk (see "Reset first-layer optimizer moments" below).

          -- Backfill training history: append not_current feature to existing pairs
          -- Training history stores inputs as matrices: { {v1, v2, ..., vN} }
          -- so we must index into [1] (the inner row vector) for length checks and mutations
          if algorithm_weights.nn.training_history then
            for _, pair in ipairs(algorithm_weights.nn.training_history) do
              local pos_row = pair.positive_input and pair.positive_input[1]
              if pos_row and #pos_row == saved_input_size then
                -- Heuristic: trigram >= 0.99 AND proximity == 1.0 suggests current file
                local is_current = pos_row[9] >= 0.99 and pos_row[6] == 1.0
                pos_row[expected_input_size] = is_current and 0.0 or 1.0
              end
              local neg_row = pair.negative_input and pair.negative_input[1]
              if neg_row and #neg_row == saved_input_size then
                local is_current = neg_row[9] >= 0.99 and neg_row[6] == 1.0
                neg_row[expected_input_size] = is_current and 0.0 or 1.0
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
        end
      end

      -- Handle migration: clear incompatible training data, reset optimizer state
      if needs_migration then
        -- Old format: training_history contains {input, target, file} samples
        -- New format: training_history contains {positive_input, negative_input, positive_file, negative_file} pairs
        -- Cannot convert old samples to pairs → clear history
        state.training_history = {}

        -- Clear loss history (loss magnitude changes from BCE to hinge)
        state.stats.loss_history = {}

        -- Keep other stats (samples_processed, batches_trained for informational purposes)
        if algorithm_weights.nn.stats then
          state.stats.samples_processed = algorithm_weights.nn.stats.samples_processed or 0
          state.stats.batches_trained = algorithm_weights.nn.stats.batches_trained or 0
          state.stats.batch_timings = {} -- Clear timing stats
          state.stats.avg_batch_timing = nil
        end

        -- Reset optimizer state (gradient statistics become invalid with hinge loss)
        state.optimizer_type = config.optimizer or "sgd"
        state.optimizer_state = nil

        -- Notify user of migration
        if old_version then
          vim.notify(
            "neural-open: Upgraded to pairwise hinge loss (v2.0). "
              .. "Network weights preserved, optimizer state and training history reset. "
              .. "Loss magnitude will differ from previous BCE loss.",
            vim.log.levels.INFO
          )
        end
      else
        -- No migration needed, load history and stats normally
        state.training_history = algorithm_weights.nn.training_history or {}
        if algorithm_weights.nn.stats then
          state.stats = vim.tbl_extend("force", state.stats, algorithm_weights.nn.stats)
          -- Ensure timing fields exist
          state.stats.batch_timings = state.stats.batch_timings or {}
          state.stats.avg_batch_timing = state.stats.avg_batch_timing or nil
          state.stats.loss_history = state.stats.loss_history or {}
        end

        -- Handle optimizer state migration and loading (only if not migrating loss function)
        local saved_optimizer_type = algorithm_weights.nn.optimizer_type
        local current_optimizer_type = config.optimizer or "sgd"

        if not saved_optimizer_type then
          -- Legacy weight file without optimizer_type - default to SGD
          state.optimizer_type = "sgd"
          state.optimizer_state = nil
        elseif saved_optimizer_type ~= current_optimizer_type then
          -- Optimizer changed - reset optimizer state but keep network weights
          state.optimizer_type = current_optimizer_type
          state.optimizer_state = nil
          vim.notify(
            "neural-open: Optimizer changed to "
              .. current_optimizer_type
              .. ". Optimizer state reset, but network weights preserved.",
            vim.log.levels.INFO
          )
        else
          -- Same optimizer - load optimizer state
          state.optimizer_type = saved_optimizer_type
          state.optimizer_state = algorithm_weights.nn.optimizer_state
        end
      end

      -- Initialize optimizer state if needed
      if state.optimizer_type == "adamw" and not state.optimizer_state then
        state.optimizer_state = init_optimizer_state("adamw", config.architecture)
      elseif state.optimizer_type == "sgd" and not state.optimizer_state then
        -- SGD now needs state for timestep tracking (warmup)
        state.optimizer_state = { timestep = 0 }
      end

      -- Reset first-layer optimizer moments after input-size migration.
      -- This runs here (not in the migration block above) because optimizer state
      -- is loaded from disk after the weight expansion, so it isn't available earlier.
      if input_size_migrated and state.optimizer_state and state.optimizer_state.moments then
        local expected_input_size = config.architecture[1]
        local m = state.optimizer_state.moments
        if m.first and m.first.weights and m.first.weights[1] then
          m.first.weights[1] = nn_core.zeros(expected_input_size, migrated_output_size)
        end
        if m.second and m.second.weights and m.second.weights[1] then
          m.second.weights[1] = nn_core.zeros(expected_input_size, migrated_output_size)
        end
      end
    end

    -- If still no weights, try loading from bundled defaults
    if not state.weights then
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
        state.weights = default_weights.network.weights
        state.biases = default_weights.network.biases
        state.gammas = default_weights.network.gammas
        state.betas = default_weights.network.betas
        state.running_means = default_weights.network.running_means
        state.running_vars = default_weights.network.running_vars
      else
        -- Fallback to random initialization if defaults unavailable or architecture differs
        state.weights, state.biases, state.gammas, state.betas, state.running_means, state.running_vars =
          nn_core.init_network(config.architecture)
      end
      state.training_history = {}

      -- Set optimizer type from config
      state.optimizer_type = config.optimizer or "sgd"
      if state.optimizer_type == "adamw" then
        state.optimizer_state = init_optimizer_state("adamw", config.architecture)
      else
        -- SGD needs state for timestep tracking (warmup)
        state.optimizer_state = { timestep = 0 }
      end
    end

    -- Ensure running statistics exist (initialize if missing from loaded weights)
    if not state.running_means or not state.running_vars then
      local _, _, _, _, running_means, running_vars = nn_core.init_network(config.architecture)
      state.running_means = running_means
      state.running_vars = running_vars
    end

    -- Build fused inference cache for fast calculate_score()
    prepare_inference_cache()
  end
end

--- Calculate score using neural network from a flat input buffer.
--- The input_buf is used directly as the first layer's input (read-only, not modified).
--- IMPORTANT: load_weights() must be called before first use (done in capture_context)
---@param input_buf number[] Flat array of 11 normalized features in canonical order
---@return number Score in [0, 100]
function M.calculate_score(input_buf)
  -- Lazy-load weights if not yet initialized (hot path skips this after first call)
  if not state.inference_cache then
    ensure_weights()
  end

  if not state.inference_cache then
    -- Fallback to general forward_pass when inference cache unavailable (e.g., empty/invalid weights)
    local input = features_to_input(input_buf, false)
    local activations = forward_pass(
      input,
      state.weights,
      state.biases,
      state.gammas,
      state.betas,
      state.running_means,
      state.running_vars,
      false,
      nil,
      false
    )
    return activations[#activations][1][1] * 100
  end

  local cache = state.inference_cache --[[@as table]]
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

--- Update neural network weights based on user selection (with optional latency tracking)
---@param selected_item NeuralOpenItem
---@param ranked_items NeuralOpenItem[]
---@param latency_ctx? table Optional latency context
function M.update_weights(selected_item, ranked_items, latency_ctx)
  local latency = require("neural-open.latency")
  ensure_weights()

  local config = ensure_config()

  -- Find selected item's rank
  local selected_rank = 1
  for i, item in ipairs(ranked_items) do
    if item.file == selected_item.file then
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
      if item.file ~= selected_item.file then
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
          positive_input = features_to_input(positive_input_buf, drop_match),
          negative_input = features_to_input(neg_input_buf, drop_match),
          positive_file = selected_item.file,
          negative_file = neg_item.file,
        })
        state.stats.samples_processed = state.stats.samples_processed + 1
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
    return construct_batches(pairs, state.training_history, config.batch_size, config.batches_per_update)
  end, "async.weight_update")

  latency.add_metadata(latency_ctx, "nn.batch_construction", {
    num_batches = #batches,
    history_size = #state.training_history,
  })

  -- Add current pairs to history (they go at the end as newest)
  for _, pair in ipairs(pairs) do
    table.insert(state.training_history, pair)
  end

  -- Maintain history size limit by removing oldest pairs
  while #state.training_history > config.history_size do
    table.remove(state.training_history, 1) -- Remove oldest (from beginning)
  end

  -- Train the network on all batches
  if #batches > 0 then
    latency.start(latency_ctx, "nn.training", "async.weight_update")
    train_on_batches(batches, latency_ctx)
    latency.finish(latency_ctx, "nn.training")

    -- Rebuild inference cache after training modified weights/batch norm parameters
    prepare_inference_cache()

    -- Inject training phase metrics from existing stats (post-hoc metadata)
    if state.stats.avg_batch_timing then
      latency.add_metadata(latency_ctx, "nn.training", {
        num_batches = #batches,
        total_pairs = math.floor(state.stats.samples_per_batch * #batches),
        avg_forward_ms = state.stats.avg_batch_timing.forward_ms,
        avg_backward_ms = state.stats.avg_batch_timing.backward_ms,
        avg_update_ms = state.stats.avg_batch_timing.update_ms,
        avg_loss = state.stats.last_loss,
        optimizer = state.optimizer_type,
      })
    end
  end

  -- Save weights (even if no training happened, state may have changed)
  latency.start(latency_ctx, "nn.save_weights", "async.weight_update")
  local weights_module = require("neural-open.weights")
  weights_module.save_weights("nn", {
    nn = {
      version = "2.0-hinge", -- Version field for pairwise hinge loss format
      network = {
        weights = state.weights,
        biases = state.biases,
        gammas = state.gammas,
        betas = state.betas,
        running_means = state.running_means,
        running_vars = state.running_vars,
      },
      training_history = state.training_history,
      stats = state.stats,
      optimizer_type = state.optimizer_type,
      optimizer_state = state.optimizer_state,
    },
  }, latency_ctx)
  latency.finish(latency_ctx, "nn.save_weights")
end

--- Calculate loss averages from history
---@return table<number, number?> averages Table with keys 1, 10, 100, 1000
local function calculate_loss_averages()
  local history = state.stats and state.stats.loss_history
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
---@return table<number, {correct_pct: number, margin_pct: number}> averages Table with keys 1, 10, 100, 1000
local function calculate_accuracy_averages()
  local history = state.stats and state.stats.ranking_accuracy_history
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

--- Generate debug view for neural network algorithm
---@param item NeuralOpenItem
---@param all_items NeuralOpenItem[]?
---@return string[]
function M.debug_view(item, all_items)
  local lines = {}

  table.insert(lines, "🧠 Neural Network Algorithm")
  table.insert(lines, "")

  local config = ensure_config()
  table.insert(lines, "Architecture: " .. table.concat(config.architecture, " → "))
  table.insert(lines, string.format("Learning Rate: %.4f", config.learning_rate))
  table.insert(lines, string.format("Weight Decay: %.6f", config.weight_decay or 0))
  table.insert(lines, string.format("Margin: %.2f", config.margin or 1.0))

  -- Optimizer information
  local optimizer_name = state.optimizer_type == "adamw" and "AdamW" or "SGD"
  table.insert(lines, "Optimizer: " .. optimizer_name)

  -- Warmup configuration and status
  local warmup_steps = config.warmup_steps or 0
  local warmup_start_factor = config.warmup_start_factor or 0.1
  if warmup_steps > 0 then
    table.insert(
      lines,
      string.format("Warmup: %d steps (start factor: %.1f%%)", warmup_steps, warmup_start_factor * 100)
    )

    -- Show current warmup factor if in warmup phase
    if state.optimizer_state then
      local current_timestep = state.optimizer_state.timestep or 0
      if current_timestep > 0 and current_timestep <= warmup_steps then
        local warmup_factor = calculate_warmup_factor(current_timestep, warmup_steps, warmup_start_factor)
        table.insert(
          lines,
          string.format(
            "  Current: step %d/%d (LR factor: %.1f%%)",
            current_timestep,
            warmup_steps,
            warmup_factor * 100
          )
        )
      elseif current_timestep > warmup_steps then
        table.insert(lines, string.format("  Status: completed (at step %d)", current_timestep))
      end
    end
  end

  -- Optimizer-specific stats
  if state.optimizer_state then
    if state.optimizer_type == "sgd" then
      table.insert(lines, string.format("  Timestep: %d", state.optimizer_state.timestep or 0))
    elseif state.optimizer_type == "adamw" then
      table.insert(lines, string.format("  Timestep: %d", state.optimizer_state.timestep))
      table.insert(
        lines,
        string.format("  Beta1: %.3f, Beta2: %.3f", config.adam_beta1 or 0.9, config.adam_beta2 or 0.999)
      )
    end
  end

  -- AdamW moment statistics
  if state.optimizer_type == "adamw" and state.optimizer_state then
    -- Calculate average moment magnitudes for first layer (as a representative sample)
    if state.optimizer_state.moments and state.optimizer_state.moments.first.weights[1] then
      local m_w = state.optimizer_state.moments.first.weights[1]
      local v_w = state.optimizer_state.moments.second.weights[1]

      local m_sum, v_sum, count = 0, 0, 0
      for i = 1, #m_w do
        for j = 1, #m_w[i] do
          m_sum = m_sum + math.abs(m_w[i][j])
          v_sum = v_sum + v_w[i][j]
          count = count + 1
        end
      end

      if count > 0 then
        table.insert(
          lines,
          string.format("  Avg 1st Moment: %.6f, Avg 2nd Moment: %.6f (L1)", m_sum / count, v_sum / count)
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
    table.insert(lines, "Dropout Rates: " .. table.concat(dropout_str, ", "))

    -- Show active neuron percentages during training
    if state.stats.dropout_active_rates and next(state.stats.dropout_active_rates) then
      local active_str = {}
      for i, rate in pairs(state.stats.dropout_active_rates) do
        if rate ~= nil and rate > 0 then
          -- Only show layers that actually have dropout applied
          table.insert(active_str, string.format("L%d: %.1f%%", i, rate))
        end
      end
      if #active_str > 0 then
        table.insert(lines, "Active Neurons (last batch): " .. table.concat(active_str, ", "))
      end
    end
  end

  -- Match dropout configuration
  if config.match_dropout and config.match_dropout > 0 then
    table.insert(
      lines,
      string.format("Match Dropout: %.1f%% (match/virtual_name features)", config.match_dropout * 100)
    )
  end

  table.insert(lines, "")

  -- Training statistics
  table.insert(lines, "Training Statistics:")
  table.insert(lines, string.format("  Samples Processed: %d", state.stats.samples_processed or 0))
  table.insert(lines, string.format("  Batches Trained: %d", state.stats.batches_trained or 0))

  -- Calculate and display loss averages on a single line
  local loss_averages = calculate_loss_averages()
  local history_size = state.stats.loss_history and #state.stats.loss_history or 0

  if history_size > 0 then
    local loss_parts = {}

    -- Add averages in order, showing partial buckets when appropriate
    local windows = { 1, 10, 100, 1000 }
    for _, window in ipairs(windows) do
      if loss_averages[window] then
        table.insert(loss_parts, string.format("[%d] %.6f", window, loss_averages[window]))
      elseif window > history_size then
        -- We've reached a window larger than our history size
        -- Show the average of all available samples as a partial bucket
        if loss_averages[history_size] then
          table.insert(loss_parts, string.format("[%d] %.6f", history_size, loss_averages[history_size]))
        end
        break -- Don't show larger windows
      end
    end

    if #loss_parts > 0 then
      table.insert(lines, "  Avg Hinge Loss: " .. table.concat(loss_parts, " "))
    end
  else
    table.insert(lines, string.format("  Last Hinge Loss: %.6f", state.stats.last_loss or 0))
  end

  -- Display ranking accuracy
  local accuracy_averages = calculate_accuracy_averages()
  local accuracy_history_size = state.stats.ranking_accuracy_history and #state.stats.ranking_accuracy_history or 0

  if accuracy_history_size > 0 then
    local accuracy_parts = {}
    local windows = { 1, 10, 100, 1000 }

    for _, window in ipairs(windows) do
      if accuracy_averages[window] then
        table.insert(
          accuracy_parts,
          string.format(
            "[%d] %.2f%% (%.2f%%)",
            window,
            accuracy_averages[window].correct_pct,
            accuracy_averages[window].margin_pct
          )
        )
      elseif window > accuracy_history_size and accuracy_averages[accuracy_history_size] then
        -- Show partial bucket
        table.insert(
          accuracy_parts,
          string.format(
            "[%d] %.2f%% (%.2f%%)",
            accuracy_history_size,
            accuracy_averages[accuracy_history_size].correct_pct,
            accuracy_averages[accuracy_history_size].margin_pct
          )
        )
        break
      end
    end

    if #accuracy_parts > 0 then
      table.insert(lines, "  Ranking Accuracy: " .. table.concat(accuracy_parts, " "))
    end
  end

  table.insert(
    lines,
    string.format(
      "  History Size: %d/%d pairs",
      state.training_history and #state.training_history or 0,
      config.history_size
    )
  )
  table.insert(lines, "  Training Mode: Pairwise Ranking (Hinge Loss)")

  -- Batch timing statistics
  local avg_timing = state.stats and state.stats.avg_batch_timing
  if avg_timing and avg_timing.total_ms then
    table.insert(
      lines,
      string.format(
        "  Avg Batch Time (last 10): %.2fms (fwd: %.2fms, back: %.2fms, upd: %.2fms)",
        avg_timing.total_ms,
        avg_timing.forward_ms or 0,
        avg_timing.backward_ms or 0,
        avg_timing.update_ms or 0
      )
    )
  end

  -- Loss interpretation
  if state.stats.last_loss and state.stats.last_loss > 0 then
    table.insert(lines, string.format("  Loss Interpretation: Avg margin violation of %.4f", state.stats.last_loss))
    table.insert(lines, string.format("    (Loss=0 means all pairs satisfy margin of %.2f)", config.margin or 1.0))
  end

  table.insert(lines, "")

  if item.nos and item.nos.neural_score then
    table.insert(lines, string.format("Current Score: %.4f", item.nos.neural_score))
    table.insert(lines, "")
  end

  -- Feature importance (based on first layer weights)
  if state.weights and state.weights[1] then
    table.insert(lines, "Feature Importance (first layer weights):")
    local feature_order = FEATURE_NAMES

    local importance = {}
    for i, name in ipairs(feature_order) do
      local weight_sum = 0
      for j = 1, #state.weights[1][i] do
        weight_sum = weight_sum + math.abs(state.weights[1][i][j])
      end
      importance[name] = weight_sum / #state.weights[1][i]
    end

    -- Sort by importance
    local sorted_importance = {}
    for name, value in pairs(importance) do
      table.insert(sorted_importance, { name = name, value = value })
    end
    table.sort(sorted_importance, function(a, b)
      return a.value > b.value
    end)

    for _, feature in ipairs(sorted_importance) do
      local formatted_name = feature.name:gsub("_", " "):gsub("(%l)(%u)", "%1 %2")
      formatted_name = formatted_name:sub(1, 1):upper() .. formatted_name:sub(2)
      table.insert(lines, string.format("  %-15s: %.4f", formatted_name, feature.value))
    end
    table.insert(lines, "")
  end

  -- Current item features from input_buf
  local normalized_features = nil
  if item.nos and item.nos.input_buf then
    normalized_features = input_buf_to_features(item.nos.input_buf)
  end
  if normalized_features then
    table.insert(lines, "Input Features (normalized):")

    local sorted_features = {}
    for name, value in pairs(normalized_features) do
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

  -- Network prediction
  if normalized_features and state.weights then
    table.insert(lines, "")
    table.insert(lines, "Network Prediction:")

    -- No match dropout during inference/debug
    local input = features_to_input(item.nos.input_buf, false)

    -- Get logit output for debug info
    local activations_logit = forward_pass(
      input,
      state.weights,
      state.biases,
      state.gammas,
      state.betas,
      state.running_means,
      state.running_vars,
      false, -- inference mode
      nil, -- no dropout
      true -- return logits
    )
    local logit = activations_logit[#activations_logit][1][1]

    -- Get sigmoid output for probability
    local activations = forward_pass(
      input,
      state.weights,
      state.biases,
      state.gammas,
      state.betas,
      state.running_means,
      state.running_vars,
      false, -- inference mode
      nil, -- no dropout
      false -- return sigmoid
    )

    -- Show activation patterns
    for i = 2, #activations do
      local layer_name = i < #activations and string.format("Hidden Layer %d", i - 1) or "Output"
      local activation = activations[i][1]

      if i < #activations then
        -- For hidden layers, show activation pattern
        local active_count = 0
        for j = 1, #activation do
          if activation[j] > 0 then
            active_count = active_count + 1
          end
        end
        table.insert(lines, string.format("  %s: %d/%d neurons active", layer_name, active_count, #activation))

        -- Show batch norm status if available
        if state.gammas and state.gammas[i - 1] then
          table.insert(
            lines,
            string.format(
              "    BatchNorm: enabled (γ mean: %.4f, β mean: %.4f)",
              state.gammas[i - 1][1][1],
              state.betas[i - 1][1][1]
            )
          )
        end
      else
        -- For output layer, show both logit and probability
        table.insert(lines, string.format("  %s Logit: %.4f", layer_name, logit))
        table.insert(lines, string.format("  %s Probability: %.4f (sigmoid)", layer_name, activation[1]))
      end
    end
  end

  -- Weight statistics
  if state.stats.weight_norms and #state.stats.weight_norms > 0 then
    table.insert(lines, "")
    table.insert(lines, "Weight Statistics:")
    for i = 1, #state.stats.weight_norms do
      local layer_name = i < #state.stats.weight_norms and string.format("Layer %d", i) or "Output Layer"
      table.insert(lines, string.format("  %s:", layer_name))
      table.insert(lines, string.format("    L2 Norm: %.4f", state.stats.weight_norms[i] or 0))
      table.insert(lines, string.format("    Avg Magnitude: %.4f", state.stats.avg_weight_magnitudes[i] or 0))
    end
  end

  return lines
end

--- Get algorithm name
---@return AlgorithmName
function M.get_name()
  return "nn"
end

--- Initialize algorithm
---@param config NosNNConfig
function M.init(config)
  state.config = vim.deepcopy(config or {})

  -- Validate required fields exist
  if not state.config.architecture then
    error("NN algorithm: config.architecture is required")
  end
  if not state.config.optimizer then
    error("NN algorithm: config.optimizer is required")
  end

  -- Set optimizer type from config
  state.optimizer_type = state.config.optimizer

  -- Validate optimizer type
  if state.optimizer_type ~= "sgd" and state.optimizer_type ~= "adamw" then
    error(string.format("Invalid optimizer type: %s. Must be 'sgd' or 'adamw'", state.optimizer_type))
  end

  -- Validate dropout configuration
  if state.config.dropout_rates then
    local hidden_layer_count = #state.config.architecture - 2 -- Exclude input and output layers
    if #state.config.dropout_rates ~= hidden_layer_count then
      error(
        string.format(
          "Dropout rates array length (%d) must match number of hidden layers (%d)",
          #state.config.dropout_rates,
          hidden_layer_count
        )
      )
    end

    -- Validate each dropout rate is in [0, 1) range
    for i, rate in ipairs(state.config.dropout_rates) do
      if rate < 0 or rate >= 1 then
        error(string.format("Dropout rate for layer %d must be in [0, 1) range, got %.2f", i, rate))
      end
    end
  end

  -- Initialize random seed for reproducibility
  math.randomseed(os.time())
end

--- Load the latest weights from the weights module
function M.load_weights()
  ensure_weights(true)
end

-- Testing helpers (only available in test environment)
---@diagnostic disable-next-line: undefined-field
if _G._TEST then
  function M._get_training_history()
    return state.training_history
  end

  function M._get_weights()
    return state.weights
  end

  function M._get_stats()
    return state.stats
  end

  function M._get_optimizer_state()
    return state.optimizer_state
  end

  function M._forward_pass(input)
    return forward_pass(
      input,
      state.weights,
      state.biases,
      state.gammas,
      state.betas,
      state.running_means,
      state.running_vars,
      false, -- inference mode
      nil, -- no dropout
      false -- return sigmoid output
    )
  end

  M._features_to_input = features_to_input
end

return M
