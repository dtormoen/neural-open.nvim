-- Tests for batch processing and batch normalization in neural network
local nn_core = require("neural-open.algorithms.nn_core")

describe("Neural Network Batch Processing", function()
  describe("batch operations", function()
    it("computes mean along columns correctly", function()
      local matrix = {
        { 1, 2, 3 },
        { 4, 5, 6 },
        { 7, 8, 9 },
      }
      local mean = nn_core.mean(matrix, 0)
      assert.are.equal(3, #mean[1])
      assert.are.near(4, mean[1][1], 0.001)
      assert.are.near(5, mean[1][2], 0.001)
      assert.are.near(6, mean[1][3], 0.001)
    end)

    it("computes variance along columns correctly", function()
      local matrix = {
        { 1, 2, 3 },
        { 4, 5, 6 },
        { 7, 8, 9 },
      }
      local mean = nn_core.mean(matrix, 0)
      local var = nn_core.variance(matrix, mean, 0)
      assert.are.equal(3, #var[1])
      assert.are.near(6, var[1][1], 0.001) -- Variance of [1,4,7]
      assert.are.near(6, var[1][2], 0.001) -- Variance of [2,5,8]
      assert.are.near(6, var[1][3], 0.001) -- Variance of [3,6,9]
    end)

    it("creates ones vector correctly", function()
      local ones = nn_core.ones(5)
      assert.are.equal(1, #ones)
      assert.are.equal(5, #ones[1])
      for i = 1, 5 do
        assert.are.equal(1, ones[1][i])
      end
    end)
  end)

  describe("batch normalization", function()
    it("normalizes batch correctly", function()
      local batch = {
        { 1, 2 },
        { 3, 4 },
        { 5, 6 },
      }
      local gamma = { { 1, 1 } }
      local beta = { { 0, 0 } }

      local normalized, _, _ = nn_core.batch_normalize(batch, gamma, beta)

      -- Check dimensions
      assert.are.equal(3, #normalized)
      assert.are.equal(2, #normalized[1])

      -- Check mean is approximately 0
      local norm_mean = nn_core.mean(normalized, 0)
      assert.are.near(0, norm_mean[1][1], 0.001)
      assert.are.near(0, norm_mean[1][2], 0.001)

      -- Check variance is approximately 1
      local norm_var = nn_core.variance(normalized, norm_mean, 0)
      assert.are.near(1, norm_var[1][1], 0.001)
      assert.are.near(1, norm_var[1][2], 0.001)
    end)

    it("applies scale and shift correctly", function()
      local batch = {
        { 1, 2 },
        { 3, 4 },
        { 5, 6 },
      }
      local gamma = { { 2, 3 } } -- Scale
      local beta = { { 0.5, -0.5 } } -- Shift

      local normalized = nn_core.batch_normalize(batch, gamma, beta)

      -- With gamma=2 and beta=0.5, normalized values should be scaled and shifted
      assert.are.equal(3, #normalized)
      assert.are.equal(2, #normalized[1])

      -- Check that scale and shift were applied
      local mean = nn_core.mean(normalized, 0)
      assert.are.near(0.5, mean[1][1], 0.001) -- Shifted by beta
      assert.are.near(-0.5, mean[1][2], 0.001)
    end)

    it("handles backward pass correctly", function()
      local batch = {
        { 1, 2, 3 },
        { 4, 5, 6 },
      }
      local gamma = { { 1, 1, 1 } }
      local beta = { { 0, 0, 0 } }

      -- Forward pass
      local _, mean, var = nn_core.batch_normalize(batch, gamma, beta)

      -- Mock gradient from next layer
      local grad_out = {
        { 0.1, 0.2, 0.3 },
        { 0.4, 0.5, 0.6 },
      }

      -- Backward pass
      local grad_input, grad_gamma, grad_beta = nn_core.batch_normalize_backward(grad_out, batch, gamma, mean, var)

      -- Check dimensions
      assert.are.equal(2, #grad_input)
      assert.are.equal(3, #grad_input[1])
      assert.are.equal(1, #grad_gamma)
      assert.are.equal(3, #grad_gamma[1])
      assert.are.equal(1, #grad_beta)
      assert.are.equal(3, #grad_beta[1])

      -- Gradient for beta should be sum of grad_out
      assert.are.near(0.5, grad_beta[1][1], 0.001) -- 0.1 + 0.4
      assert.are.near(0.7, grad_beta[1][2], 0.001) -- 0.2 + 0.5
      assert.are.near(0.9, grad_beta[1][3], 0.001) -- 0.3 + 0.6
    end)

    it("handles binary features with numerical stability", function()
      -- Create batch with mostly binary features (simulating the real use case)
      local batch = {
        { 0, 1, 0, 1, 0 },
        { 1, 1, 0, 0, 1 },
        { 0, 1, 1, 1, 0 },
        { 1, 0, 0, 1, 1 },
      }
      local gamma = { { 1, 1, 1, 1, 1 } }
      local beta = { { 0, 0, 0, 0, 0 } }

      -- This should not cause numerical issues even with binary features
      local normalized, _, var = nn_core.batch_normalize(batch, gamma, beta)

      -- Check that variance is properly clamped (should be at least 1e-5)
      for j = 1, #var[1] do
        assert.is_true(var[1][j] >= 1e-5, "Variance should be clamped to prevent numerical issues")
      end

      -- Check that normalized values are finite
      for i = 1, #normalized do
        for j = 1, #normalized[i] do
          assert.is_true(normalized[i][j] == normalized[i][j], "NaN detected in normalized output")
          assert.is_true(math.abs(normalized[i][j]) < 1e10, "Infinite or very large value detected")
        end
      end
    end)

    it("handles near-zero variance correctly", function()
      -- Create batch where all values in a column are identical (zero variance)
      local batch = {
        { 1, 2, 5 },
        { 1, 3, 5 },
        { 1, 4, 5 },
      }
      local gamma = { { 1, 1, 1 } }
      local beta = { { 0, 0, 0 } }

      local normalized, _, var = nn_core.batch_normalize(batch, gamma, beta)

      -- Column 1 and 3 have zero natural variance, should be clamped
      assert.is_true(var[1][1] >= 1e-5, "Zero variance should be clamped")
      assert.is_true(var[1][3] >= 1e-5, "Zero variance should be clamped")

      -- All normalized values should be finite
      for i = 1, #normalized do
        for j = 1, #normalized[i] do
          assert.is_true(normalized[i][j] == normalized[i][j], "NaN detected")
          assert.is_true(math.abs(normalized[i][j]) < 1e10, "Infinite value detected")
        end
      end
    end)
  end)

  describe("network initialization", function()
    it("initializes batch norm parameters", function()
      local architecture = { 3, 4, 2, 1 }
      local weights, biases, gammas, betas = nn_core.init_network(architecture)

      -- Check weights and biases
      assert.are.equal(3, #weights)
      assert.are.equal(3, #biases)

      -- Check batch norm parameters (only for hidden layers)
      assert.are.equal(2, #gammas) -- Two hidden layers
      assert.are.equal(2, #betas)

      -- First hidden layer batch norm
      assert.are.equal(1, #gammas[1])
      assert.are.equal(4, #gammas[1][1]) -- 4 neurons in first hidden layer
      for i = 1, 4 do
        assert.are.equal(1, gammas[1][1][i]) -- Initialized to 1
        assert.are.equal(0, betas[1][1][i]) -- Initialized to 0
      end

      -- Second hidden layer batch norm
      assert.are.equal(1, #gammas[2])
      assert.are.equal(2, #gammas[2][1]) -- 2 neurons in second hidden layer
      for i = 1, 2 do
        assert.are.equal(1, gammas[2][1][i])
        assert.are.equal(0, betas[2][1][i])
      end
    end)
  end)

  describe("batch forward pass", function()
    it("processes batches through the network", function()
      local architecture = { 2, 3, 1 }
      local weights, biases, gammas, betas = nn_core.init_network(architecture)

      -- Create a batch of inputs
      local batch = {
        { 0.5, 0.3 },
        { 0.7, 0.2 },
        { 0.1, 0.9 },
      }

      -- Mock forward pass (simplified without actual nn module)
      local activations = { batch }

      -- First layer
      local z1 = nn_core.matmul(batch, weights[1])
      z1 = nn_core.add_bias(z1, biases[1])

      -- Apply batch norm
      local z1_norm = nn_core.batch_normalize(z1, gammas[1], betas[1])

      -- Apply ReLU
      local a1 = nn_core.element_wise(z1_norm, nn_core.relu)
      table.insert(activations, a1)

      -- Output layer
      local z2 = nn_core.matmul(a1, weights[2])
      z2 = nn_core.add_bias(z2, biases[2])
      local a2 = nn_core.element_wise(z2, nn_core.sigmoid)
      table.insert(activations, a2)

      -- Check output dimensions
      assert.are.equal(3, #a2) -- Batch size
      assert.are.equal(1, #a2[1]) -- Output size

      -- Check outputs are in valid range [0, 1]
      for i = 1, 3 do
        assert.is_true(a2[i][1] >= 0 and a2[i][1] <= 1)
      end
    end)
  end)

  describe("numerical gradient checking", function()
    it("verifies batch norm gradients numerically", function()
      local batch = {
        { 1.0, 2.0 },
        { 3.0, 1.0 },
        { 2.0, 3.0 },
      }
      local gamma = { { 1.5, 0.8 } }
      local beta = { { 0.2, -0.3 } }
      local epsilon = 1e-8

      -- Forward pass
      local output, mean, var = nn_core.batch_normalize(batch, gamma, beta, epsilon)

      -- Mock loss (sum of squares for simplicity)
      local function compute_loss(normalized)
        local loss = 0
        for i = 1, #normalized do
          for j = 1, #normalized[i] do
            loss = loss + normalized[i][j] * normalized[i][j]
          end
        end
        return loss
      end

      local loss = compute_loss(output)

      -- Gradient w.r.t output (dL/dy = 2y)
      local grad_out = {}
      for i = 1, #output do
        grad_out[i] = {}
        for j = 1, #output[i] do
          grad_out[i][j] = 2 * output[i][j]
        end
      end

      -- Analytical gradients
      local _, grad_gamma, _ = nn_core.batch_normalize_backward(grad_out, batch, gamma, mean, var, epsilon)

      -- Numerical gradient for gamma[1][1]
      local h = 1e-5
      gamma[1][1] = gamma[1][1] + h
      local output_plus = nn_core.batch_normalize(batch, gamma, beta, epsilon)
      local loss_plus = compute_loss(output_plus)
      gamma[1][1] = gamma[1][1] - h

      local numerical_grad_gamma = (loss_plus - loss) / h

      -- Check that analytical and numerical gradients are close
      assert.are.near(numerical_grad_gamma, grad_gamma[1][1], 0.01)
    end)
  end)

  describe("integration with nn module", function()
    local orig_nn, orig_weights

    before_each(function()
      orig_nn = package.loaded["neural-open.algorithms.nn"]
      orig_weights = package.loaded["neural-open.weights"]
    end)

    after_each(function()
      package.loaded["neural-open.algorithms.nn"] = orig_nn
      package.loaded["neural-open.weights"] = orig_weights
    end)

    it("trains on batches correctly", function()
      -- Fresh nn module: clear cached module to reset all internal state
      -- (prevents interference from other test files that may have modified nn state)
      package.loaded["neural-open.algorithms.nn"] = nil

      -- Mock weights module BEFORE requiring nn (so ensure_weights uses our mock)
      local saved_weights = nil
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_weights or {}
        end,
        save_weights = function(_, weights)
          saved_weights = weights
        end,
      }

      local nn_test = require("neural-open.algorithms.nn")

      nn_test.init({
        architecture = { 11, 4, 1 },
        optimizer = "sgd",
        learning_rate = 0.1,
        batch_size = 2,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        batches_per_update = 1,
        history_size = 100,
        margin = 1.0,
        match_dropout = 0,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
        weight_decay = 0.0001,
        dropout_rates = { 0 },
      })

      -- Fixed seed AFTER init (overrides os.time() seed) so ensure_weights() initializes deterministically
      math.randomseed(42)

      local input_buf1 = { 0.8, 0.2, 0.5, 1.0, 0.0, 0.7, 1.0, 0.3, 0.6, 0.0, 1.0 }
      local input_buf2 = { 0.3, 0.9, 0.2, 0.0, 1.0, 0.4, 0.0, 0.8, 0.1, 0.0, 1.0 }

      -- Calculate initial scores (triggers ensure_weights → deterministic random init)
      local score1_before = nn_test.calculate_score(input_buf1)
      local score2_before = nn_test.calculate_score(input_buf2)

      local selected_item = {
        file = "test1.lua",
        nos = { input_buf = input_buf1, normalized_path = "test1.lua" },
      }

      local ranked_items = {
        { file = "test2.lua", nos = { input_buf = input_buf2, normalized_path = "test2.lua" } },
        selected_item,
      }

      -- Train for multiple iterations to ensure convergence
      for _ = 1, 10 do
        nn_test.update_weights(selected_item, ranked_items)
      end

      -- Verify weights were saved
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn)
      assert.is_not_nil(saved_weights.nn.network)
      assert.is_not_nil(saved_weights.nn.network.weights)
      assert.is_not_nil(saved_weights.nn.network.gammas)
      assert.is_not_nil(saved_weights.nn.network.betas)

      -- Load the updated weights and rebuild inference cache
      nn_test.load_weights()

      -- Calculate scores after training
      local score1_after = nn_test.calculate_score(input_buf1)
      local score2_after = nn_test.calculate_score(input_buf2)

      -- Selected item should have higher score after training
      assert.is_true(
        score1_after > score2_after or (score1_after - score1_before) > (score2_after - score2_before),
        string.format(
          "Selected item score should improve more than non-selected (s1: %.4f→%.4f, s2: %.4f→%.4f)",
          score1_before,
          score1_after,
          score2_before,
          score2_after
        )
      )
    end)
  end)
end)
