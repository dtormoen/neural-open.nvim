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
end)
