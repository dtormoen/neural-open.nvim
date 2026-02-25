describe("classic algorithm", function()
  local classic
  local mock_weights
  local helpers

  before_each(function()
    -- Load helpers and get real default config
    helpers = require("tests.helpers")
    local default_config = helpers.get_default_config()

    -- Mock the weights module
    mock_weights = {
      weights = {
        classic = vim.deepcopy(default_config.algorithm_config.classic.default_weights),
      },
      default_weights = {
        classic = vim.deepcopy(default_config.algorithm_config.classic.default_weights),
      },
      get_weights = function(algo_name)
        return mock_weights.weights[algo_name] or {}
      end,
      get_default_weights = function(algo_name)
        return mock_weights.default_weights[algo_name] or {}
      end,
      save_weights = function(algo_name, weights)
        mock_weights.weights[algo_name] = weights
      end,
    }

    package.loaded["neural-open.weights"] = mock_weights
    package.loaded["neural-open.algorithms.classic"] = nil
    classic = require("neural-open.algorithms.classic")

    -- Initialize with real config (learning_rate is already default 0.6)
    local config = helpers.create_algorithm_config("classic")
    classic.init(config.algorithm_config.classic)

    -- Load weights after init
    classic.load_weights()
  end)

  after_each(function()
    package.loaded["neural-open.weights"] = nil
    package.loaded["neural-open.algorithms.classic"] = nil
  end)

  describe("calculate_score", function()
    -- Flat array order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition

    it("should calculate weighted sum of features", function()
      -- match=0.8, virtual_name=0, frecency=0.5, open=0, alt=0, proximity=0.3, project=0, recency=0.2, trigram=0, transition=0
      local input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0.2, 0, 0 }

      -- Expected: 0.8*140 + 0.5*17 + 0.3*13 + 0.2*9 = 112 + 8.5 + 3.9 + 1.8 = 126.2
      local score = classic.calculate_score(input_buf)
      assert.are.equal(126.2, score)
    end)

    it("should handle zero values", function()
      -- match=0.8, virtual_name=0, frecency=0, open=0, alt=0, proximity=0, project=0, recency=0.2, trigram=0, transition=0
      local input_buf = { 0.8, 0, 0, 0, 0, 0, 0, 0.2, 0, 0 }

      -- Expected: 0.8*140 + 0*17 + 0*13 + 0.2*9 = 112 + 0 + 0 + 1.8 = 113.8
      local score = classic.calculate_score(input_buf)
      assert.are.equal(113.8, score)
    end)

    it("should handle all-zero features", function()
      local input_buf = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

      -- Expected: 0
      local score = classic.calculate_score(input_buf)
      assert.are.equal(0, score)
    end)
  end)

  describe("update_weights", function()
    it("should not update weights when item is rank 1", function()
      -- Flat array: match=0.8, rest zeros
      local selected_item = {
        neural_rank = 1,
        nos = {
          input_buf = { 0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        },
      }
      local ranked_items = { selected_item }

      local original_weights = vim.deepcopy(mock_weights.weights.classic)
      classic.update_weights(selected_item, ranked_items)

      -- Weights should not have changed
      assert.are.same(original_weights, mock_weights.weights.classic)
    end)

    it("should update weights when selecting lower-ranked item", function()
      -- Flat order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition
      local selected_item = {
        neural_rank = 2,
        nos = {
          input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0.2, 0, 0 },
        },
      }

      local higher_item = {
        neural_rank = 1,
        nos = {
          input_buf = { 0.7, 0, 0.6, 0, 0, 0.3, 0, 0.1, 0, 0 },
        },
      }

      local ranked_items = { higher_item, selected_item }

      -- update_weights now saves internally
      classic.update_weights(selected_item, ranked_items)

      -- Match weight should increase (selected had higher match score)
      assert.is_true(mock_weights.weights.classic.match > 140)
      -- Frecency weight should decrease (higher item had better frecency)
      assert.is_true(mock_weights.weights.classic.frecency < 17)
    end)
  end)

  describe("simulate_weight_adjustments", function()
    it("should simulate without applying changes", function()
      -- Flat order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition
      local selected_item = {
        neural_rank = 2,
        nos = {
          input_buf = { 0.8, 0, 0.5, 0, 0, 0, 0, 0, 0, 0 },
        },
      }

      local higher_item = {
        neural_rank = 1,
        nos = {
          input_buf = { 0.7, 0, 0.6, 0, 0, 0, 0, 0, 0, 0 },
        },
      }

      local ranked_items = { higher_item, selected_item }

      local original_weights = vim.deepcopy(mock_weights.weights.classic)
      local simulation = classic.simulate_weight_adjustments(selected_item, ranked_items)

      -- Weights should not have changed
      assert.are.same(original_weights, mock_weights.weights.classic)

      -- Should return simulation results
      assert.is_not_nil(simulation)
      assert.is_not_nil(simulation.changes)
      assert.is_not_nil(simulation.new_weights)
    end)
  end)

  describe("debug_view", function()
    it("should return detailed debug information", function()
      -- Flat order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition
      local item = {
        nos = {
          neural_score = 129,
          raw_features = {
            match = 80,
            frecency = 25,
          },
          input_buf = { 0.8, 0, 0.5, 0, 0, 0, 0, 0, 0, 0 },
        },
      }

      local lines = classic.debug_view(item)

      assert.is_table(lines)
      assert.is_true(#lines > 0)

      local content = table.concat(lines, "\n")
      assert.is_true(content:find("Classic Algorithm") ~= nil)
      assert.is_true(content:find("129") ~= nil) -- Score
      assert.is_true(content:find("0.6") ~= nil) -- Learning rate
    end)

    it("should display transition feature in debug output", function()
      -- Flat order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition
      local item = {
        nos = {
          neural_score = 100,
          raw_features = {
            match = 80,
            frecency = 25,
            transition = 0.5,
          },
          input_buf = { 0.8, 0, 0.5, 0, 0, 0, 0, 0, 0, 0.5 },
        },
      }

      local lines = classic.debug_view(item)
      local content = table.concat(lines, "\n")

      -- Verify transition appears in the raw→normalized features section
      assert.is_true(content:find("Transition") ~= nil, "Debug view should display transition feature")
      -- Verify the normalized value is shown
      assert.is_true(
        content:find("0.5000") ~= nil or content:find("0.50") ~= nil,
        "Debug view should show transition value"
      )
    end)
  end)

  describe("get_name", function()
    it("should return algorithm name", function()
      assert.are.equal("classic", classic.get_name())
    end)
  end)
end)
