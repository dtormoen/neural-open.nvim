-- Neural network core functionality for neural-open
-- Provides matrix operations, activation functions, and network primitives

local M = {}

-- Latency tracking context (module-level)
local _latency_ctx = nil
local _call_stats = {}

--- Set latency context for tracking performance
---@param ctx table|nil Latency context from latency module
function M.set_latency_context(ctx)
  _latency_ctx = ctx
end

--- Reset call statistics
function M.reset_call_stats()
  _call_stats = {
    matmul_count = 0,
    matmul_ops = 0,
    element_wise_count = 0,
    element_wise_ops = 0,
    element_wise2_count = 0,
    element_wise2_ops = 0,
    add_count = 0,
    subtract_count = 0,
    scalar_mul_count = 0,
    batch_norm_count = 0,
  }
end

--- Get call statistics
---@return table stats Call statistics
function M.get_call_stats()
  return _call_stats
end

-- Matrix operations

--- Create a matrix filled with zeros
---@param rows number
---@param cols number
---@return table
function M.zeros(rows, cols)
  local matrix = {}
  for i = 1, rows do
    matrix[i] = {}
    for j = 1, cols do
      matrix[i][j] = 0
    end
  end
  return matrix
end

--- Create a matrix with Xavier/He initialization
---@param rows number
---@param cols number
---@param prev_size number Size of previous layer (for He initialization)
---@return table
function M.xavier_init(rows, cols, prev_size)
  local matrix = {}
  local scale = math.sqrt(2.0 / prev_size) -- He initialization for ReLU
  for i = 1, rows do
    matrix[i] = {}
    for j = 1, cols do
      -- Random normal approximation using Box-Muller transform
      local u1 = math.random()
      local u2 = math.random()
      local z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
      matrix[i][j] = z * scale
    end
  end
  return matrix
end

--- Matrix multiplication (A × B)
---@param a table Matrix A (m×n)
---@param b table Matrix B (n×p)
---@return table Result matrix (m×p)
function M.matmul(a, b)
  local m = #a
  local n = #a[1]
  local p = #b[1]

  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.matmul_count = _call_stats.matmul_count + 1
    _call_stats.matmul_ops = _call_stats.matmul_ops + (m * n * p)
  end

  local latency = require("neural-open.latency")
  local result = latency.measure(_latency_ctx, "core.matmul", function()
    local mat = M.zeros(m, p)
    for i = 1, m do
      for j = 1, p do
        local sum = 0
        for k = 1, n do
          sum = sum + a[i][k] * b[k][j]
        end
        mat[i][j] = sum
      end
    end
    return mat
  end)
  return result
end

--- Matrix addition with broadcasting for bias (A + b)
---@param matrix table Matrix (m×n)
---@param bias table Bias vector (1×n)
---@return table Result matrix (m×n)
function M.add_bias(matrix, bias)
  local m = #matrix
  local n = #matrix[1]
  local result = M.zeros(m, n)

  for i = 1, m do
    for j = 1, n do
      result[i][j] = matrix[i][j] + bias[1][j]
    end
  end
  return result
end

--- Element-wise matrix operation
---@param matrix table Input matrix
---@param operation function Operation to apply
---@return table Result matrix
function M.element_wise(matrix, operation)
  local m = #matrix
  local n = #matrix[1]

  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.element_wise_count = _call_stats.element_wise_count + 1
    _call_stats.element_wise_ops = _call_stats.element_wise_ops + (m * n)
  end

  local latency = require("neural-open.latency")
  local result = latency.measure(_latency_ctx, "core.element_wise", function()
    local mat = M.zeros(m, n)
    for i = 1, m do
      for j = 1, n do
        mat[i][j] = operation(matrix[i][j])
      end
    end
    return mat
  end)
  return result
end

--- Element-wise binary operation on two matrices
---@param a table First matrix
---@param b table Second matrix
---@param operation function Binary operation to apply
---@return table Result matrix
function M.element_wise2(a, b, operation)
  local m = #a
  local n = #a[1]

  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.element_wise2_count = _call_stats.element_wise2_count + 1
    _call_stats.element_wise2_ops = _call_stats.element_wise2_ops + (m * n)
  end

  local latency = require("neural-open.latency")
  local result = latency.measure(_latency_ctx, "core.element_wise2", function()
    local mat = M.zeros(m, n)
    for i = 1, m do
      for j = 1, n do
        mat[i][j] = operation(a[i][j], b[i][j])
      end
    end
    return mat
  end)
  return result
end

--- Matrix transpose
---@param matrix table Input matrix (m×n)
---@return table Transposed matrix (n×m)
function M.transpose(matrix)
  local m = #matrix
  local n = #matrix[1]
  local result = M.zeros(n, m)

  for i = 1, m do
    for j = 1, n do
      result[j][i] = matrix[i][j]
    end
  end
  return result
end

--- Element-wise matrix multiplication (Hadamard product)
---@param a table Matrix A
---@param b table Matrix B
---@return table Result matrix
function M.hadamard(a, b)
  local m = #a
  local n = #a[1]
  local result = M.zeros(m, n)

  for i = 1, m do
    for j = 1, n do
      result[i][j] = a[i][j] * b[i][j]
    end
  end
  return result
end

--- Scalar multiplication
---@param matrix table Input matrix
---@param scalar number Scalar value
---@return table Result matrix
function M.scalar_mul(matrix, scalar)
  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.scalar_mul_count = _call_stats.scalar_mul_count + 1
  end

  return M.element_wise(matrix, function(x)
    return x * scalar
  end)
end

--- Matrix addition (A + B)
---@param a table Matrix A
---@param b table Matrix B
---@return table Result matrix
function M.add(a, b)
  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.add_count = _call_stats.add_count + 1
  end

  local m = #a
  local n = #a[1]
  local result = M.zeros(m, n)

  for i = 1, m do
    for j = 1, n do
      result[i][j] = a[i][j] + b[i][j]
    end
  end
  return result
end

--- Matrix subtraction (A - B)
---@param a table Matrix A
---@param b table Matrix B
---@return table Result matrix
function M.subtract(a, b)
  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.subtract_count = _call_stats.subtract_count + 1
  end

  local m = #a
  local n = #a[1]
  local result = M.zeros(m, n)

  for i = 1, m do
    for j = 1, n do
      result[i][j] = a[i][j] - b[i][j]
    end
  end
  return result
end

-- Activation functions

--- ReLU activation function
---@param x number Input value
---@return number Activated value
function M.relu(x)
  return math.max(0, x)
end

--- ReLU derivative
---@param x number Input value
---@return number Derivative value
function M.relu_derivative(x)
  return x > 0 and 1 or 0
end

--- Leaky ReLU activation function
---@param x number Input value
---@param alpha? number Leak coefficient (default 0.01)
---@return number Activated value
function M.leaky_relu(x, alpha)
  alpha = alpha or 0.01
  return x > 0 and x or alpha * x
end

--- Leaky ReLU derivative
---@param x number Input value
---@param alpha? number Leak coefficient (default 0.01)
---@return number Derivative value
function M.leaky_relu_derivative(x, alpha)
  alpha = alpha or 0.01
  return x >= 0 and 1 or alpha
end

--- Sigmoid activation function
---@param x number Input value
---@return number Activated value (0 to 1)
function M.sigmoid(x)
  -- Numerical stability for large negative values
  if x < -500 then
    return 0
  elseif x > 500 then
    return 1
  end
  return 1 / (1 + math.exp(-x))
end

-- Loss functions

--- Pairwise hinge loss for ranking
---@param score_pos number Positive item score (0-1)
---@param score_neg number Negative item score (0-1)
---@param margin number Margin hyperparameter (default 1.0)
---@return number loss Hinge loss value (0 or positive)
function M.pairwise_hinge_loss(score_pos, score_neg, margin)
  local diff = score_pos - score_neg
  return math.max(0, margin - diff)
end

-- Utility functions

--- Deep copy a matrix
---@param matrix table Input matrix
---@return table Copied matrix
function M.copy_matrix(matrix)
  local copy = {}
  for i = 1, #matrix do
    copy[i] = {}
    for j = 1, #matrix[i] do
      copy[i][j] = matrix[i][j]
    end
  end
  return copy
end

--- Convert vector to matrix (for batch processing)
---@param vector table Input vector
---@return table Matrix (1×n)
function M.vector_to_matrix(vector)
  return { vector }
end

--- Convert matrix to vector (extract first row)
---@param matrix table Input matrix
---@return table Vector
function M.matrix_to_vector(matrix)
  return matrix[1]
end

-- Batch processing operations

--- Compute mean along specified axis
---@param matrix table Input matrix (batch_size × features)
---@param axis number Axis to compute mean (0 for columns, 1 for rows)
---@return table Mean vector or scalar
function M.mean(matrix, axis)
  if axis == 0 then
    -- Mean along columns (result is 1 × features)
    local m = #matrix
    local n = #matrix[1]
    local result = M.zeros(1, n)

    for j = 1, n do
      local sum = 0
      for i = 1, m do
        sum = sum + matrix[i][j]
      end
      result[1][j] = sum / m
    end
    return result
  elseif axis == 1 then
    -- Mean along rows (result is batch_size × 1)
    local m = #matrix
    local result = {}

    for i = 1, m do
      local sum = 0
      for j = 1, #matrix[i] do
        sum = sum + matrix[i][j]
      end
      result[i] = { sum / #matrix[i] }
    end
    return result
  else
    error("Invalid axis: must be 0 or 1")
  end
end

--- Compute variance along specified axis
---@param matrix table Input matrix (batch_size × features)
---@param mean table Precomputed mean (1 × features for axis=0)
---@param axis number Axis to compute variance (0 for columns)
---@return table Variance vector
function M.variance(matrix, mean, axis)
  if axis == 0 then
    -- Variance along columns (result is 1 × features)
    local m = #matrix
    local n = #matrix[1]
    local result = M.zeros(1, n)

    for j = 1, n do
      local sum_sq = 0
      for i = 1, m do
        local diff = matrix[i][j] - mean[1][j]
        sum_sq = sum_sq + diff * diff
      end
      result[1][j] = sum_sq / m
    end
    return result
  else
    error("Variance along axis 1 not implemented")
  end
end

--- Apply batch normalization to a batch of inputs
---@param matrix table Input matrix (batch_size × features)
---@param gamma table Scale parameters (1 × features)
---@param beta table Shift parameters (1 × features)
---@param epsilon? number Small constant for numerical stability (default 1e-5)
---@param training? boolean Whether in training mode (default true)
---@param running_mean? table Running mean for inference (1 × features)
---@param running_var? table Running variance for inference (1 × features)
---@param momentum? number Momentum for updating running stats (default 0.1)
---@return table normalized, table mean, table var - Normalized output and statistics
function M.batch_normalize(matrix, gamma, beta, epsilon, training, running_mean, running_var, momentum)
  local m = #matrix
  local n = #matrix[1]

  -- Track statistics if latency context is active
  if _latency_ctx then
    _call_stats.batch_norm_count = _call_stats.batch_norm_count + 1
  end

  epsilon = epsilon or 1e-5 -- Increased from 1e-8 for better stability
  training = training == nil and true or training -- Default to training mode
  momentum = momentum or 0.1 -- Standard momentum for batch norm

  -- Manual timing for functions with multiple return values
  local latency = require("neural-open.latency")
  latency.start(_latency_ctx, "core.batch_normalize")

  local batch_mean, batch_var
  local use_mean, use_var

  if training then
    -- Training mode: compute batch statistics
    batch_mean = M.mean(matrix, 0)
    batch_var = M.variance(matrix, batch_mean, 0)

    -- Clamp variance to prevent numerical issues with binary features
    for j = 1, #batch_var[1] do
      batch_var[1][j] = math.max(batch_var[1][j], 1e-5)
    end

    -- Update running statistics if provided (exponential moving average)
    if running_mean and running_var then
      for j = 1, #running_mean[1] do
        running_mean[1][j] = (1 - momentum) * running_mean[1][j] + momentum * batch_mean[1][j]
        running_var[1][j] = (1 - momentum) * running_var[1][j] + momentum * batch_var[1][j]
      end
    end

    use_mean = batch_mean
    use_var = batch_var
  else
    -- Inference mode: use running statistics if available
    if running_mean and running_var then
      use_mean = running_mean
      use_var = running_var
    else
      -- Fallback to batch statistics if running stats not available
      batch_mean = M.mean(matrix, 0)
      batch_var = M.variance(matrix, batch_mean, 0)
      for j = 1, #batch_var[1] do
        batch_var[1][j] = math.max(batch_var[1][j], 1e-5)
      end
      use_mean = batch_mean
      use_var = batch_var
    end
  end

  -- Normalize
  local normalized = M.zeros(m, n)

  for i = 1, m do
    for j = 1, n do
      -- x_norm = (x - mean) / sqrt(var + epsilon)
      local x_norm = (matrix[i][j] - use_mean[1][j]) / math.sqrt(use_var[1][j] + epsilon)
      -- y = gamma * x_norm + beta
      normalized[i][j] = gamma[1][j] * x_norm + beta[1][j]
    end
  end

  latency.finish(_latency_ctx, "core.batch_normalize")

  -- Return batch statistics for backward pass (even in inference mode for debugging)
  return normalized, batch_mean or use_mean, batch_var or use_var
end

--- Compute gradients for batch normalization backward pass
---@param grad_out table Gradient from next layer (batch_size × features)
---@param input table Original input to batch norm (batch_size × features)
---@param gamma table Scale parameters (1 × features)
---@param mean table Batch mean (1 × features)
---@param var table Batch variance (1 × features)
---@param epsilon number Small constant for numerical stability
---@return table grad_input, table grad_gamma, table grad_beta
function M.batch_normalize_backward(grad_out, input, gamma, mean, var, epsilon)
  epsilon = epsilon or 1e-8
  local m = #input -- batch size
  local n = #input[1] -- features

  -- Initialize gradients
  local grad_gamma = M.zeros(1, n)
  local grad_beta = M.zeros(1, n)
  local grad_input = M.zeros(m, n)

  -- Compute intermediate values
  local x_norm = M.zeros(m, n)
  for i = 1, m do
    for j = 1, n do
      x_norm[i][j] = (input[i][j] - mean[1][j]) / math.sqrt(var[1][j] + epsilon)
    end
  end

  -- Compute gradients for gamma and beta
  for j = 1, n do
    local sum_grad_beta = 0
    local sum_grad_gamma = 0
    for i = 1, m do
      sum_grad_beta = sum_grad_beta + grad_out[i][j]
      sum_grad_gamma = sum_grad_gamma + grad_out[i][j] * x_norm[i][j]
    end
    grad_beta[1][j] = sum_grad_beta
    grad_gamma[1][j] = sum_grad_gamma
  end

  -- Compute gradient for input
  for j = 1, n do
    local std = math.sqrt(var[1][j] + epsilon)
    local inv_std = 1.0 / std

    -- Compute intermediate sums
    local sum1 = 0
    local sum2 = 0
    for i = 1, m do
      sum1 = sum1 + grad_out[i][j] * gamma[1][j]
      sum2 = sum2 + grad_out[i][j] * gamma[1][j] * (input[i][j] - mean[1][j])
    end

    -- Apply chain rule
    for i = 1, m do
      grad_input[i][j] = (1.0 / m)
        * gamma[1][j]
        * inv_std
        * (m * grad_out[i][j] - sum1 - (input[i][j] - mean[1][j]) * inv_std * inv_std * sum2)
    end
  end

  return grad_input, grad_gamma, grad_beta
end

--- Create a vector of ones
---@param size number Size of the vector
---@return table Vector of ones (1 × size)
function M.ones(size)
  local vector = {}
  vector[1] = {}
  for i = 1, size do
    vector[1][i] = 1
  end
  return vector
end

--- Initialize network weights
---@param architecture table Array of layer sizes
---@return table weights, table biases, table gammas, table betas, table running_means, table running_vars
function M.init_network(architecture)
  local weights = {}
  local biases = {}
  local gammas = {}
  local betas = {}
  local running_means = {}
  local running_vars = {}

  for i = 1, #architecture - 1 do
    local input_size = architecture[i]
    local output_size = architecture[i + 1]

    -- Initialize weights with He initialization
    weights[i] = M.xavier_init(input_size, output_size, input_size)

    -- Initialize biases to small values
    biases[i] = M.zeros(1, output_size)

    -- Initialize batch norm parameters for hidden layers (not output layer)
    if i < #architecture - 1 then
      gammas[i] = M.ones(output_size) -- Scale to 1
      betas[i] = M.zeros(1, output_size) -- Shift to 0
      running_means[i] = M.zeros(1, output_size) -- Running mean
      running_vars[i] = M.ones(output_size) -- Running variance (init to 1)
    end
  end

  return weights, biases, gammas, betas, running_means, running_vars
end

-- Dropout functions

--- Generate a binary dropout mask
---@param rows number Number of rows
---@param cols number Number of columns
---@param rate number Dropout rate (probability of dropping a neuron)
---@return table Binary mask matrix (1 = keep, 0 = drop)
function M.dropout_mask(rows, cols, rate)
  local mask = M.zeros(rows, cols)
  for i = 1, rows do
    for j = 1, cols do
      -- Keep neuron with probability (1 - rate)
      mask[i][j] = math.random() > rate and 1 or 0
    end
  end
  return mask
end

--- Apply dropout to a matrix of activations
---@param matrix table Input matrix of activations
---@param rate number Dropout rate (0-1)
---@param training boolean Whether in training mode
---@return table result, table|nil mask - Result matrix and dropout mask (if training)
function M.dropout(matrix, rate, training)
  if not training or rate <= 0 then
    return matrix, nil
  end

  local rows = #matrix
  local cols = #matrix[1]
  local mask = M.dropout_mask(rows, cols, rate)

  -- Apply mask and scale by inverted dropout factor
  local scale = 1.0 / (1.0 - rate)
  local result = M.hadamard(matrix, mask)
  result = M.scalar_mul(result, scale)

  return result, mask
end

--- Clip gradients by global norm to prevent gradient explosion
---@param gradients table List of gradient matrices
---@param max_norm number Maximum allowed global norm
---@return table clipped_gradients The clipped gradients
function M.clip_gradients(gradients, max_norm)
  -- Calculate global norm across all gradients
  local global_norm = 0
  for i = 1, #gradients do
    if gradients[i] then
      for j = 1, #gradients[i] do
        for k = 1, #gradients[i][j] do
          global_norm = global_norm + gradients[i][j][k] * gradients[i][j][k]
        end
      end
    end
  end
  global_norm = math.sqrt(global_norm)

  -- If norm exceeds max_norm, scale all gradients
  if global_norm > max_norm then
    local scale = max_norm / global_norm
    local clipped = {}
    for i = 1, #gradients do
      if gradients[i] then
        clipped[i] = M.scalar_mul(gradients[i], scale)
      else
        clipped[i] = gradients[i]
      end
    end
    return clipped
  end

  return gradients
end

return M
