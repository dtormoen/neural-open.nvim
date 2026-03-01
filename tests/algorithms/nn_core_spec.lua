describe("Neural Network Core", function()
  local nn_core = require("neural-open.algorithms.nn_core")

  describe("Matrix Operations", function()
    it("creates zero matrices", function()
      local m = nn_core.zeros(2, 3)
      assert.equals(2, #m)
      assert.equals(3, #m[1])
      assert.equals(0, m[1][1])
      assert.equals(0, m[2][3])
    end)

    it("performs matrix multiplication", function()
      local a = { { 1, 2 }, { 3, 4 } }
      local b = { { 5, 6 }, { 7, 8 } }
      local result = nn_core.matmul(a, b)

      -- [1,2] × [5,6] = [1×5+2×7, 1×6+2×8] = [19, 22]
      -- [3,4]   [7,8]   [3×5+4×7, 3×6+4×8]   [43, 50]
      assert.equals(19, result[1][1])
      assert.equals(22, result[1][2])
      assert.equals(43, result[2][1])
      assert.equals(50, result[2][2])
    end)

    it("adds bias to matrix", function()
      local matrix = { { 1, 2 }, { 3, 4 } }
      local bias = { { 10, 20 } }
      local result = nn_core.add_bias(matrix, bias)

      assert.equals(11, result[1][1])
      assert.equals(22, result[1][2])
      assert.equals(13, result[2][1])
      assert.equals(24, result[2][2])
    end)

    it("transposes matrices", function()
      local matrix = { { 1, 2, 3 }, { 4, 5, 6 } }
      local result = nn_core.transpose(matrix)

      assert.equals(3, #result)
      assert.equals(2, #result[1])
      assert.equals(1, result[1][1])
      assert.equals(4, result[1][2])
      assert.equals(3, result[3][1])
      assert.equals(6, result[3][2])
    end)

    it("performs element-wise operations", function()
      local matrix = { { -1, 2 }, { 0, -3 } }
      local result = nn_core.element_wise(matrix, nn_core.relu)

      assert.equals(0, result[1][1])
      assert.equals(2, result[1][2])
      assert.equals(0, result[2][1])
      assert.equals(0, result[2][2])
    end)

    it("performs Hadamard product", function()
      local a = { { 1, 2 }, { 3, 4 } }
      local b = { { 5, 6 }, { 7, 8 } }
      local result = nn_core.hadamard(a, b)

      assert.equals(5, result[1][1])
      assert.equals(12, result[1][2])
      assert.equals(21, result[2][1])
      assert.equals(32, result[2][2])
    end)

    it("performs scalar multiplication", function()
      local matrix = { { 1, 2 }, { 3, 4 } }
      local result = nn_core.scalar_mul(matrix, 0.5)

      assert.equals(0.5, result[1][1])
      assert.equals(1.0, result[1][2])
      assert.equals(1.5, result[2][1])
      assert.equals(2.0, result[2][2])
    end)

    it("performs matrix subtraction", function()
      local a = { { 5, 6 }, { 7, 8 } }
      local b = { { 1, 2 }, { 3, 4 } }
      local result = nn_core.subtract(a, b)

      assert.equals(4, result[1][1])
      assert.equals(4, result[1][2])
      assert.equals(4, result[2][1])
      assert.equals(4, result[2][2])
    end)
  end)

  describe("Activation Functions", function()
    it("applies ReLU activation", function()
      assert.equals(0, nn_core.relu(-5))
      assert.equals(0, nn_core.relu(0))
      assert.equals(5, nn_core.relu(5))
    end)

    it("computes ReLU derivative", function()
      assert.equals(0, nn_core.relu_derivative(-5))
      assert.equals(0, nn_core.relu_derivative(0))
      assert.equals(1, nn_core.relu_derivative(5))
    end)

    it("applies sigmoid activation", function()
      local s0 = nn_core.sigmoid(0)
      assert.is_true(math.abs(s0 - 0.5) < 0.001)

      local s_large = nn_core.sigmoid(10)
      assert.is_true(s_large > 0.999)

      local s_small = nn_core.sigmoid(-10)
      assert.is_true(s_small < 0.001)
    end)

    it("applies leaky ReLU activation", function()
      -- Positive values pass through unchanged
      assert.equals(5, nn_core.leaky_relu(5))
      assert.equals(0.5, nn_core.leaky_relu(0.5))

      -- Negative values are scaled by alpha
      assert.equals(-0.01, nn_core.leaky_relu(-1, 0.01))
      assert.equals(-0.05, nn_core.leaky_relu(-5, 0.01))

      -- Custom alpha
      assert.equals(-0.2, nn_core.leaky_relu(-1, 0.2))
      assert.equals(-1.0, nn_core.leaky_relu(-5, 0.2))

      -- Zero should return zero
      assert.equals(0, nn_core.leaky_relu(0))
    end)

    it("computes leaky ReLU derivative", function()
      -- Positive values have derivative 1
      assert.equals(1, nn_core.leaky_relu_derivative(5))
      assert.equals(1, nn_core.leaky_relu_derivative(0.1))

      -- Negative values have derivative alpha
      assert.equals(0.01, nn_core.leaky_relu_derivative(-1, 0.01))
      assert.equals(0.01, nn_core.leaky_relu_derivative(-5, 0.01))

      -- Custom alpha
      assert.equals(0.2, nn_core.leaky_relu_derivative(-1, 0.2))

      -- Zero edge case (treated as positive)
      assert.equals(1, nn_core.leaky_relu_derivative(0))
    end)
  end)

  describe("Network Initialization", function()
    it("initializes network weights and biases", function()
      local architecture = { 3, 4, 2, 1 }
      local weights, biases = nn_core.init_network(architecture)

      -- Check number of weight matrices
      assert.equals(3, #weights)

      -- Check dimensions of weight matrices
      assert.equals(3, #weights[1]) -- 3×4 matrix
      assert.equals(4, #weights[1][1])
      assert.equals(4, #weights[2]) -- 4×2 matrix
      assert.equals(2, #weights[2][1])
      assert.equals(2, #weights[3]) -- 2×1 matrix
      assert.equals(1, #weights[3][1])

      -- Check dimensions of bias vectors
      assert.equals(3, #biases)
      assert.equals(1, #biases[1]) -- 1×4 vector
      assert.equals(4, #biases[1][1])
      assert.equals(1, #biases[2]) -- 1×2 vector
      assert.equals(2, #biases[2][1])
      assert.equals(1, #biases[3]) -- 1×1 vector
      assert.equals(1, #biases[3][1])

      -- Check that weights are initialized (not all zeros)
      local has_non_zero = false
      for i = 1, #weights[1] do
        for j = 1, #weights[1][i] do
          if weights[1][i][j] ~= 0 then
            has_non_zero = true
            break
          end
        end
      end
      assert.is_true(has_non_zero)
    end)
  end)

  describe("Utility Functions", function()
    it("deep copies matrices", function()
      local original = { { 1, 2 }, { 3, 4 } }
      local copy = nn_core.copy_matrix(original)

      -- Verify values are copied
      assert.equals(1, copy[1][1])
      assert.equals(4, copy[2][2])

      -- Verify it's a deep copy
      copy[1][1] = 99
      assert.equals(1, original[1][1])
    end)

    it("converts vectors to matrices", function()
      local vector = { 1, 2, 3 }
      local matrix = nn_core.vector_to_matrix(vector)

      assert.equals(1, #matrix)
      assert.equals(3, #matrix[1])
      assert.equals(1, matrix[1][1])
      assert.equals(3, matrix[1][3])
    end)

    it("converts matrices to vectors", function()
      local matrix = { { 1, 2, 3 } }
      local vector = nn_core.matrix_to_vector(matrix)

      assert.equals(3, #vector)
      assert.equals(1, vector[1])
      assert.equals(3, vector[3])
    end)
  end)

  describe("Dropout Functions", function()
    it("generates dropout mask with correct proportion", function()
      math.randomseed(42) -- Set seed for reproducibility
      local mask = nn_core.dropout_mask(10, 10, 0.5)

      -- Count active neurons
      local active_count = 0
      for i = 1, 10 do
        for j = 1, 10 do
          if mask[i][j] == 1 then
            active_count = active_count + 1
          end
        end
      end

      -- Should be approximately 50% active (with some variance)
      assert.is_true(active_count >= 30 and active_count <= 70)
    end)

    it("returns original matrix when not training", function()
      local matrix = { { 1, 2, 3 }, { 4, 5, 6 } }
      local result, mask = nn_core.dropout(matrix, 0.5, false)

      -- Should return original matrix unchanged
      assert.same(matrix, result)
      assert.is_nil(mask)
    end)

    it("returns original matrix when dropout rate is 0", function()
      local matrix = { { 1, 2, 3 }, { 4, 5, 6 } }
      local result, mask = nn_core.dropout(matrix, 0, true)

      -- Should return original matrix unchanged
      assert.same(matrix, result)
      assert.is_nil(mask)
    end)

    it("applies dropout and inverted scaling during training", function()
      math.randomseed(42) -- Set seed for reproducibility
      local matrix = { { 2, 4, 6 }, { 8, 10, 12 } }
      local dropout_rate = 0.5
      local result, mask = nn_core.dropout(matrix, dropout_rate, true)

      assert.not_nil(mask)
      assert.equals(#matrix, #mask)
      assert.equals(#matrix[1], #mask[1])

      -- Check inverted dropout scaling
      local scale = 1.0 / (1.0 - dropout_rate)
      for i = 1, #matrix do
        for j = 1, #matrix[i] do
          if mask[i][j] == 1 then
            -- Active neurons should be scaled
            assert.equals(matrix[i][j] * scale, result[i][j])
          else
            -- Dropped neurons should be zero
            assert.equals(0, result[i][j])
          end
        end
      end
    end)

    it("maintains expected value with inverted dropout", function()
      math.randomseed(42)
      local matrix = { { 100, 100, 100, 100 } }
      local dropout_rate = 0.5

      -- Run dropout many times and check average
      local sum = { 0, 0, 0, 0 }
      local num_trials = 1000

      for _ = 1, num_trials do
        local result = nn_core.dropout(matrix, dropout_rate, true)
        for j = 1, 4 do
          sum[j] = sum[j] + result[1][j]
        end
      end

      -- Average should be close to original values
      for j = 1, 4 do
        local avg = sum[j] / num_trials
        -- Allow 10% variance
        assert.is_true(math.abs(avg - 100) < 10, "Expected value not maintained")
      end
    end)

    it("handles different dropout rates correctly", function()
      math.randomseed(42)

      -- Test with high dropout rate (0.9)
      local mask_high = nn_core.dropout_mask(10, 10, 0.9)
      local active_high = 0
      for i = 1, 10 do
        for j = 1, 10 do
          if mask_high[i][j] == 1 then
            active_high = active_high + 1
          end
        end
      end
      assert.is_true(active_high < 30, "Too many active neurons for 0.9 dropout rate")

      -- Test with low dropout rate (0.1)
      local mask_low = nn_core.dropout_mask(10, 10, 0.1)
      local active_low = 0
      for i = 1, 10 do
        for j = 1, 10 do
          if mask_low[i][j] == 1 then
            active_low = active_low + 1
          end
        end
      end
      assert.is_true(active_low > 70, "Too few active neurons for 0.1 dropout rate")
    end)
  end)

  describe("Gradient Clipping", function()
    it("clips gradients when norm exceeds max_norm", function()
      local gradients = {
        { { 3, 4 }, { 0, 0 } }, -- norm = 5
        { { 12, 0 }, { 0, 0 } }, -- norm = 12, total = 13
      }

      local max_norm = 5.0
      local clipped = nn_core.clip_gradients(gradients, max_norm)

      -- Calculate the scaling factor: 5/13
      local scale = 5.0 / 13.0

      -- Check that gradients are scaled correctly
      assert.is_true(math.abs(clipped[1][1][1] - 3 * scale) < 0.001)
      assert.is_true(math.abs(clipped[1][1][2] - 4 * scale) < 0.001)
      assert.is_true(math.abs(clipped[2][1][1] - 12 * scale) < 0.001)

      -- Verify the new norm is approximately max_norm
      local new_norm = 0
      for i = 1, #clipped do
        for j = 1, #clipped[i] do
          for k = 1, #clipped[i][j] do
            new_norm = new_norm + clipped[i][j][k] * clipped[i][j][k]
          end
        end
      end
      new_norm = math.sqrt(new_norm)
      assert.is_true(math.abs(new_norm - max_norm) < 0.001)
    end)

    it("does not clip gradients when norm is below max_norm", function()
      local gradients = {
        { { 1, 1 }, { 1, 1 } }, -- norm = 2
        { { 0, 0 }, { 0, 1 } }, -- norm = 1, total = sqrt(5) ≈ 2.236
      }

      local max_norm = 10.0
      local clipped = nn_core.clip_gradients(gradients, max_norm)

      -- Gradients should remain unchanged
      assert.equals(1, clipped[1][1][1])
      assert.equals(1, clipped[1][1][2])
      assert.equals(1, clipped[1][2][1])
      assert.equals(1, clipped[1][2][2])
      assert.equals(0, clipped[2][1][1])
      assert.equals(1, clipped[2][2][2])
    end)

    it("handles nil gradients gracefully", function()
      local gradients = {
        { { 3, 4 } },
        nil,
        { { 0, 5 } },
      }

      local max_norm = 10.0
      local clipped = nn_core.clip_gradients(gradients, max_norm)

      -- Should handle nil without error
      assert.equals(3, clipped[1][1][1])
      assert.equals(4, clipped[1][1][2])
      assert.is_nil(clipped[2])
      assert.equals(0, clipped[3][1][1])
      assert.equals(5, clipped[3][1][2])
    end)
  end)

  describe("Batch Operations", function()
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
      assert.are.near(6, var[1][1], 0.001)
      assert.are.near(6, var[1][2], 0.001)
      assert.are.near(6, var[1][3], 0.001)
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

  describe("Batch Normalization", function()
    it("normalizes batch correctly", function()
      local batch = {
        { 1, 2 },
        { 3, 4 },
        { 5, 6 },
      }
      local gamma = { { 1, 1 } }
      local beta = { { 0, 0 } }

      local normalized, _, _ = nn_core.batch_normalize(batch, gamma, beta)

      assert.are.equal(3, #normalized)
      assert.are.equal(2, #normalized[1])

      local norm_mean = nn_core.mean(normalized, 0)
      assert.are.near(0, norm_mean[1][1], 0.001)
      assert.are.near(0, norm_mean[1][2], 0.001)

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
      local gamma = { { 2, 3 } }
      local beta = { { 0.5, -0.5 } }

      local normalized = nn_core.batch_normalize(batch, gamma, beta)

      assert.are.equal(3, #normalized)
      assert.are.equal(2, #normalized[1])

      local mean = nn_core.mean(normalized, 0)
      assert.are.near(0.5, mean[1][1], 0.001)
      assert.are.near(-0.5, mean[1][2], 0.001)
    end)

    it("handles backward pass correctly", function()
      local batch = {
        { 1, 2, 3 },
        { 4, 5, 6 },
      }
      local gamma = { { 1, 1, 1 } }
      local beta = { { 0, 0, 0 } }

      local _, mean, var = nn_core.batch_normalize(batch, gamma, beta)

      local grad_out = {
        { 0.1, 0.2, 0.3 },
        { 0.4, 0.5, 0.6 },
      }

      local grad_input, grad_gamma, grad_beta = nn_core.batch_normalize_backward(grad_out, batch, gamma, mean, var)

      assert.are.equal(2, #grad_input)
      assert.are.equal(3, #grad_input[1])
      assert.are.equal(1, #grad_gamma)
      assert.are.equal(3, #grad_gamma[1])
      assert.are.equal(1, #grad_beta)
      assert.are.equal(3, #grad_beta[1])

      assert.are.near(0.5, grad_beta[1][1], 0.001)
      assert.are.near(0.7, grad_beta[1][2], 0.001)
      assert.are.near(0.9, grad_beta[1][3], 0.001)
    end)

    it("handles binary features with numerical stability", function()
      local batch = {
        { 0, 1, 0, 1, 0 },
        { 1, 1, 0, 0, 1 },
        { 0, 1, 1, 1, 0 },
        { 1, 0, 0, 1, 1 },
      }
      local gamma = { { 1, 1, 1, 1, 1 } }
      local beta = { { 0, 0, 0, 0, 0 } }

      local normalized, _, var = nn_core.batch_normalize(batch, gamma, beta)

      for j = 1, #var[1] do
        assert.is_true(var[1][j] >= 1e-5, "Variance should be clamped to prevent numerical issues")
      end

      for i = 1, #normalized do
        for j = 1, #normalized[i] do
          assert.is_true(normalized[i][j] == normalized[i][j], "NaN detected in normalized output")
          assert.is_true(math.abs(normalized[i][j]) < 1e10, "Infinite or very large value detected")
        end
      end
    end)

    it("handles near-zero variance correctly", function()
      local batch = {
        { 1, 2, 5 },
        { 1, 3, 5 },
        { 1, 4, 5 },
      }
      local gamma = { { 1, 1, 1 } }
      local beta = { { 0, 0, 0 } }

      local normalized, _, var = nn_core.batch_normalize(batch, gamma, beta)

      assert.is_true(var[1][1] >= 1e-5, "Zero variance should be clamped")
      assert.is_true(var[1][3] >= 1e-5, "Zero variance should be clamped")

      for i = 1, #normalized do
        for j = 1, #normalized[i] do
          assert.is_true(normalized[i][j] == normalized[i][j], "NaN detected")
          assert.is_true(math.abs(normalized[i][j]) < 1e10, "Infinite value detected")
        end
      end
    end)
  end)

  describe("Batch Forward Pass", function()
    it("processes batches through the network", function()
      local architecture = { 2, 3, 1 }
      local weights, biases, gammas, betas = nn_core.init_network(architecture)

      local batch = {
        { 0.5, 0.3 },
        { 0.7, 0.2 },
        { 0.1, 0.9 },
      }

      local activations = { batch }

      local z1 = nn_core.matmul(batch, weights[1])
      z1 = nn_core.add_bias(z1, biases[1])

      local z1_norm = nn_core.batch_normalize(z1, gammas[1], betas[1])

      local a1 = nn_core.element_wise(z1_norm, nn_core.relu)
      table.insert(activations, a1)

      local z2 = nn_core.matmul(a1, weights[2])
      z2 = nn_core.add_bias(z2, biases[2])
      local a2 = nn_core.element_wise(z2, nn_core.sigmoid)
      table.insert(activations, a2)

      assert.are.equal(3, #a2)
      assert.are.equal(1, #a2[1])

      for i = 1, 3 do
        assert.is_true(a2[i][1] >= 0 and a2[i][1] <= 1)
      end
    end)
  end)

  describe("Numerical Gradient Checking", function()
    it("verifies batch norm gradients numerically", function()
      local batch = {
        { 1.0, 2.0 },
        { 3.0, 1.0 },
        { 2.0, 3.0 },
      }
      local gamma = { { 1.5, 0.8 } }
      local beta = { { 0.2, -0.3 } }
      local epsilon = 1e-8

      local output, mean, var = nn_core.batch_normalize(batch, gamma, beta, epsilon)

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

      local grad_out = {}
      for i = 1, #output do
        grad_out[i] = {}
        for j = 1, #output[i] do
          grad_out[i][j] = 2 * output[i][j]
        end
      end

      local _, grad_gamma, _ = nn_core.batch_normalize_backward(grad_out, batch, gamma, mean, var, epsilon)

      local h = 1e-5
      gamma[1][1] = gamma[1][1] + h
      local output_plus = nn_core.batch_normalize(batch, gamma, beta, epsilon)
      local loss_plus = compute_loss(output_plus)
      gamma[1][1] = gamma[1][1] - h

      local numerical_grad_gamma = (loss_plus - loss) / h

      assert.are.near(numerical_grad_gamma, grad_gamma[1][1], 0.01)
    end)
  end)

  describe("pairwise_hinge_loss", function()
    it("returns zero when margin is satisfied", function()
      local loss = nn_core.pairwise_hinge_loss(0.8, 0.3, 0.5)
      assert.equals(0, loss)
    end)

    it("returns positive loss when margin is violated", function()
      local loss = nn_core.pairwise_hinge_loss(0.6, 0.5, 0.5)
      assert.is_true(math.abs(loss - 0.4) < 0.0001)
    end)

    it("returns maximum penalty when negative ranks higher", function()
      local loss = nn_core.pairwise_hinge_loss(0.3, 0.8, 1.0)
      assert.is_true(math.abs(loss - 1.5) < 0.0001)
    end)

    it("handles equal scores correctly", function()
      local loss = nn_core.pairwise_hinge_loss(0.5, 0.5, 1.0)
      assert.is_true(math.abs(loss - 1.0) < 0.0001)
    end)

    it("handles edge cases with extreme values", function()
      local loss1 = nn_core.pairwise_hinge_loss(1.0, 1.0, 1.0)
      assert.is_true(math.abs(loss1 - 1.0) < 0.0001)

      local loss2 = nn_core.pairwise_hinge_loss(0.0, 0.0, 1.0)
      assert.is_true(math.abs(loss2 - 1.0) < 0.0001)

      local loss3 = nn_core.pairwise_hinge_loss(1.0, 0.0, 1.0)
      assert.equals(0, loss3)
    end)

    it("works with different margin values", function()
      local loss1 = nn_core.pairwise_hinge_loss(0.8, 0.3, 0.3)
      assert.equals(0, loss1)

      local loss2 = nn_core.pairwise_hinge_loss(0.8, 0.3, 0.7)
      assert.is_true(math.abs(loss2 - 0.2) < 0.0001)
    end)

    describe("Gradient Correctness", function()
      it("has zero gradient when margin is satisfied", function()
        local loss = nn_core.pairwise_hinge_loss(0.9, 0.2, 0.5)
        assert.equals(0, loss)
      end)

      it("has non-zero gradient when margin is violated", function()
        local loss = nn_core.pairwise_hinge_loss(0.6, 0.5, 0.5)
        assert.is_true(loss > 0)
      end)

      it("gradient magnitude is constant when margin is violated", function()
        local margin = 1.0

        local loss1 = nn_core.pairwise_hinge_loss(0.6, 0.1, margin)
        assert.is_true(math.abs(loss1 - 0.5) < 0.0001)

        local loss2 = nn_core.pairwise_hinge_loss(0.4, 0.1, margin)
        assert.is_true(math.abs(loss2 - 0.7) < 0.0001)

        local loss_diff = loss2 - loss1
        local expected_diff = 0.2
        assert.is_true(math.abs(loss_diff - expected_diff) < 0.0001)
      end)
    end)

    describe("Edge Case Handling", function()
      it("handles very small score differences", function()
        local loss = nn_core.pairwise_hinge_loss(0.500001, 0.5, 1.0)
        assert.is_true(math.abs(loss - 0.999999) < 0.0001)
      end)

      it("handles very large margins", function()
        local loss = nn_core.pairwise_hinge_loss(0.9, 0.1, 10.0)
        assert.is_true(math.abs(loss - 9.2) < 0.0001)
      end)

      it("handles zero margin", function()
        local loss1 = nn_core.pairwise_hinge_loss(0.6, 0.5, 0.0)
        assert.equals(0, loss1)

        local loss2 = nn_core.pairwise_hinge_loss(0.5, 0.6, 0.0)
        assert.is_true(math.abs(loss2 - 0.1) < 0.0001)
      end)

      it("handles negative margins (not recommended but should work)", function()
        local loss = nn_core.pairwise_hinge_loss(0.5, 0.5, -0.5)
        assert.equals(0, loss)
      end)
    end)

    it("loss equals max(0, margin - diff) for representative cases", function()
      local cases = {
        { 0.9, 0.2, 1.5 },
        { 0.3, 0.8, 0.5 },
        { 0.5, 0.5, 1.0 },
        { 0.7, 0.1, 0.3 },
        { 0.1, 0.9, 2.0 },
      }

      for _, case in ipairs(cases) do
        local score_pos, score_neg, margin = case[1], case[2], case[3]
        local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
        local expected = math.max(0, margin - (score_pos - score_neg))

        assert.is_true(
          math.abs(loss - expected) < 0.0001,
          string.format("Loss should equal max(0, margin - diff): expected=%.4f, got=%.4f", expected, loss)
        )
      end
    end)
  end)
end)
