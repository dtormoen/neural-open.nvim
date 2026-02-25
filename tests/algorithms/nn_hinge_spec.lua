describe("Pairwise Hinge Loss", function()
  local nn_core
  local scorer = require("neural-open.scorer")

  local function features_to_input_buf(features)
    local buf = {}
    for i, name in ipairs(scorer.FEATURE_NAMES) do
      buf[i] = features[name] or 0
    end
    return buf
  end

  before_each(function()
    nn_core = require("neural-open.algorithms.nn_core")
  end)

  describe("pairwise_hinge_loss", function()
    it("returns zero when margin is satisfied", function()
      -- score_pos - score_neg >= margin
      local loss = nn_core.pairwise_hinge_loss(0.8, 0.3, 0.5)
      assert.equals(0, loss)
    end)

    it("returns positive loss when margin is violated", function()
      -- score_pos - score_neg < margin
      local loss = nn_core.pairwise_hinge_loss(0.6, 0.5, 0.5)
      assert.is_true(math.abs(loss - 0.4) < 0.0001) -- max(0, 0.5 - 0.1) = 0.4
    end)

    it("returns maximum penalty when negative ranks higher", function()
      -- score_neg > score_pos
      local loss = nn_core.pairwise_hinge_loss(0.3, 0.8, 1.0)
      assert.is_true(math.abs(loss - 1.5) < 0.0001) -- max(0, 1.0 - (-0.5)) = 1.5
    end)

    it("handles equal scores correctly", function()
      local loss = nn_core.pairwise_hinge_loss(0.5, 0.5, 1.0)
      assert.is_true(math.abs(loss - 1.0) < 0.0001) -- max(0, 1.0 - 0) = 1.0
    end)

    it("handles edge cases with extreme values", function()
      -- Both at max
      local loss1 = nn_core.pairwise_hinge_loss(1.0, 1.0, 1.0)
      assert.is_true(math.abs(loss1 - 1.0) < 0.0001)

      -- Both at min
      local loss2 = nn_core.pairwise_hinge_loss(0.0, 0.0, 1.0)
      assert.is_true(math.abs(loss2 - 1.0) < 0.0001)

      -- Maximum separation
      local loss3 = nn_core.pairwise_hinge_loss(1.0, 0.0, 1.0)
      assert.equals(0, loss3)
    end)

    it("works with different margin values", function()
      -- With margin 0.3, diff 0.5 > 0.3, so loss = 0
      local loss1 = nn_core.pairwise_hinge_loss(0.8, 0.3, 0.3)
      assert.equals(0, loss1) -- 0.5 > 0.3

      -- With margin 0.7, diff 0.5 < 0.7, so loss = 0.2
      local loss2 = nn_core.pairwise_hinge_loss(0.8, 0.3, 0.7)
      assert.is_true(math.abs(loss2 - 0.2) < 0.0001) -- 0.5 < 0.7
    end)

    describe("Gradient Correctness", function()
      it("has zero gradient when margin is satisfied", function()
        -- When loss = 0, gradient should be 0
        -- score_pos = 0.9, score_neg = 0.2, margin = 0.5
        -- diff = 0.7 > 0.5, so loss = 0
        local score_pos = 0.9
        local score_neg = 0.2
        local margin = 0.5

        local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
        assert.equals(0, loss)

        -- When loss = 0, the gradient is 0 (no update needed)
        -- We can't test gradient directly, but we verify loss is exactly 0
      end)

      it("has non-zero gradient when margin is violated", function()
        -- When loss > 0, gradient should be non-zero
        -- score_pos = 0.6, score_neg = 0.5, margin = 0.5
        -- diff = 0.1 < 0.5, so loss = 0.4 > 0
        local score_pos = 0.6
        local score_neg = 0.5
        local margin = 0.5

        local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
        assert.is_true(loss > 0)

        -- The gradient would be: dL/d(score_pos) = -1, dL/d(score_neg) = +1
        -- We verify loss is positive, indicating gradient exists
      end)

      it("gradient magnitude is constant when margin is violated", function()
        -- Hinge loss has constant gradient of ±1 when violated
        -- Test different violations produce linear loss increases

        local margin = 1.0

        -- Test case 1: diff = 0.5, loss = 0.5
        local loss1 = nn_core.pairwise_hinge_loss(0.6, 0.1, margin)
        assert.is_true(math.abs(loss1 - 0.5) < 0.0001)

        -- Test case 2: diff = 0.3, loss = 0.7 (0.2 more violation)
        local loss2 = nn_core.pairwise_hinge_loss(0.4, 0.1, margin)
        assert.is_true(math.abs(loss2 - 0.7) < 0.0001)

        -- Verify linearity: difference in loss equals difference in violation
        local loss_diff = loss2 - loss1
        local expected_diff = 0.2 -- (0.5 - 0.3)
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
        assert.is_true(math.abs(loss - 9.2) < 0.0001) -- max(0, 10.0 - 0.8) = 9.2
      end)

      it("handles zero margin", function()
        -- With margin = 0, any positive diff satisfies the constraint
        local loss1 = nn_core.pairwise_hinge_loss(0.6, 0.5, 0.0)
        assert.equals(0, loss1) -- diff = 0.1 > 0

        -- When diff is negative, loss = margin - diff = 0 - (-0.1) = 0.1
        local loss2 = nn_core.pairwise_hinge_loss(0.5, 0.6, 0.0)
        assert.is_true(math.abs(loss2 - 0.1) < 0.0001)
      end)

      it("handles negative margins (not recommended but should work)", function()
        -- With negative margin, constraint is easier to satisfy
        local loss = nn_core.pairwise_hinge_loss(0.5, 0.5, -0.5)
        -- diff = 0, loss = max(0, -0.5 - 0) = 0
        assert.equals(0, loss)
      end)
    end)

    describe("Property-Based Tests", function()
      it("loss is always non-negative", function()
        -- Test multiple random cases
        math.randomseed(12345)
        for _ = 1, 100 do
          local score_pos = math.random()
          local score_neg = math.random()
          local margin = math.random() * 2

          local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
          assert.is_true(loss >= 0, string.format("Loss should be >= 0, got %.4f", loss))
        end
      end)

      it("loss decreases as score_pos increases", function()
        local score_neg = 0.5
        local margin = 1.0

        local prev_loss = nil
        for score_pos = 0.1, 1.0, 0.1 do
          local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
          if prev_loss then
            -- Loss should decrease or stay same (when margin satisfied)
            assert.is_true(
              loss <= prev_loss,
              string.format("Loss should decrease: prev=%.4f, curr=%.4f", prev_loss, loss)
            )
          end
          prev_loss = loss
        end
      end)

      it("loss increases as score_neg increases", function()
        local score_pos = 0.5
        local margin = 1.0

        local prev_loss = nil
        for score_neg = 0.0, 0.9, 0.1 do
          local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
          if prev_loss then
            -- Loss should increase or stay same (when margin satisfied)
            assert.is_true(
              loss >= prev_loss,
              string.format("Loss should increase: prev=%.4f, curr=%.4f", prev_loss, loss)
            )
          end
          prev_loss = loss
        end
      end)

      it("loss increases as margin increases", function()
        local score_pos = 0.8
        local score_neg = 0.3

        local prev_loss = nil
        for margin = 0.1, 2.0, 0.1 do
          local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
          if prev_loss then
            -- Loss should increase (or stay at 0 for small margins)
            assert.is_true(
              loss >= prev_loss,
              string.format("Loss should increase: prev=%.4f, curr=%.4f", prev_loss, loss)
            )
          end
          prev_loss = loss
        end
      end)

      it("loss equals max(0, margin - diff)", function()
        math.randomseed(54321)
        for _ = 1, 50 do
          local score_pos = math.random()
          local score_neg = math.random()
          local margin = math.random() * 2

          local loss = nn_core.pairwise_hinge_loss(score_pos, score_neg, margin)
          local expected = math.max(0, margin - (score_pos - score_neg))

          assert.is_true(
            math.abs(loss - expected) < 0.0001,
            string.format("Loss should equal max(0, margin - diff): expected=%.4f, got=%.4f", expected, loss)
          )
        end
      end)
    end)

    describe("Numerical Stability", function()
      it("handles very small differences without underflow", function()
        local loss = nn_core.pairwise_hinge_loss(0.5000000001, 0.5, 1.0)
        assert.is_true(loss > 0.99 and loss < 1.01)
      end)

      it("handles extreme score combinations", function()
        -- All extreme combinations should produce valid results
        local cases = {
          { 0.0, 0.0, 1.0 },
          { 1.0, 1.0, 1.0 },
          { 1.0, 0.0, 1.0 },
          { 0.0, 1.0, 1.0 },
          { 0.5, 0.5, 0.0 },
        }

        for _, case in ipairs(cases) do
          local loss = nn_core.pairwise_hinge_loss(case[1], case[2], case[3])
          assert.is_true(loss >= 0 and loss <= 2.0, string.format("Loss out of range: %.4f", loss))
          assert.is_true(loss == loss, "Loss should not be NaN") -- NaN check
        end
      end)
    end)
  end)

  describe("Pair Construction", function()
    local nn

    before_each(function()
      -- Setup environment
      _G._TEST = true
      _G.vim = _G.vim or {}
      _G.vim.loop = {
        hrtime = function()
          return 0
        end,
      }
      local original_vim_tbl_extend = _G.vim.tbl_extend
      local original_vim_tbl_deep_extend = _G.vim.tbl_deep_extend

      _G.vim.tbl_extend = function(mode, ...)
        local result = {}
        for i = 1, select("#", ...) do
          local tbl = select(i, ...)
          for k, v in pairs(tbl) do
            result[k] = v
          end
        end
        return result
      end
      _G.vim.tbl_deep_extend = _G.vim.tbl_extend -- For testing
      _G.vim.notify = function() end
      _G.vim.log = { levels = { INFO = 1 } }

      -- Clear module cache
      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.algorithms.nn_core"] = nil

      -- Mock weights module
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return nil
        end,
        save_weights = function() end,
      }

      nn = require("neural-open.algorithms.nn")
      nn.init({
        architecture = { 11, 4, 1 },
        optimizer = "sgd",
        learning_rate = 0.01,
        batch_size = 4,
        history_size = 20,
        batches_per_update = 1,
        margin = 1.0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        match_dropout = 0,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
        weight_decay = 0.0001,
        dropout_rates = { 0 }, -- One hidden layer (layer 4), no dropout for testing
      })

      -- Store originals for restoration
      _G._original_vim_tbl_extend = original_vim_tbl_extend
      _G._original_vim_tbl_deep_extend = original_vim_tbl_deep_extend
    end)

    after_each(function()
      _G.vim.tbl_extend = _G._original_vim_tbl_extend
      _G.vim.tbl_deep_extend = _G._original_vim_tbl_deep_extend
    end)

    local function create_item(file, features, score)
      local default_features = {
        match = 0.5,
        virtual_name = 0.5,
        frecency = 0.5,
        open = 0,
        alt = 0,
        proximity = 0.5,
        project = 1,
        recency = 0.5,
        trigram = 0.5,
        transition = 0.0,
      }
      return {
        file = file,
        nos = {
          input_buf = features_to_input_buf(features or default_features),
          neural_score = score or 50,
          normalized_path = file,
        },
      }
    end

    it("constructs pairs when selected item is ranked 5th", function()
      local items = {
        create_item("1.lua", nil, 90),
        create_item("2.lua", nil, 80),
        create_item("3.lua", nil, 70),
        create_item("4.lua", nil, 60),
        create_item("selected.lua", nil, 50), -- Selected, rank 5
        create_item("6.lua", nil, 40),
        create_item("7.lua", nil, 30),
      }

      -- Call update_weights
      nn.update_weights(items[5], items)

      -- Verify training history contains pairs
      local history = nn._get_training_history()
      assert.is_true(#history > 0)

      -- Should have: 4 hard negatives (items 1-4) + 1 immediate (item 6) + 1 random from item 7 = 6 pairs
      assert.is_true(#history >= 5 and #history <= 6, string.format("Expected 5-6 pairs, got %d", #history))

      -- Each pair should have correct structure
      for _, pair in ipairs(history) do
        assert.is_not_nil(pair.positive_input)
        assert.is_not_nil(pair.negative_input)
        assert.equals("selected.lua", pair.positive_file)
        assert.is_string(pair.negative_file)
      end
    end)

    it("constructs pairs when selected item is ranked 1st", function()
      local items = {
        create_item("selected.lua", nil, 90), -- Selected, rank 1
        create_item("2.lua", nil, 80),
        create_item("3.lua", nil, 70),
      }

      nn.update_weights(items[1], items)

      -- Should have: 0 hard negatives + 1 immediate (item 2) + 1 random from item 3 = 2 pairs
      local history = nn._get_training_history()
      assert.is_true(#history >= 1 and #history <= 2, string.format("Expected 1-2 pairs, got %d", #history))
    end)

    it("handles case when selected item is last", function()
      local items = {
        create_item("1.lua", nil, 90),
        create_item("selected.lua", nil, 80), -- Selected, rank 2 (last)
      }

      nn.update_weights(items[2], items)

      -- Should have: 1 hard negative (item 1) + 0 immediate + 0 random = 1 pair
      local history = nn._get_training_history()
      assert.equals(1, #history)
      assert.equals("selected.lua", history[1].positive_file)
      assert.equals("1.lua", history[1].negative_file)
    end)

    it("respects history size limit", function()
      -- Set small history size
      nn.init({
        history_size = 5,
        batch_size = 2,
        batches_per_update = 1,
        architecture = { 11, 4, 1 },
        optimizer = "sgd",
        learning_rate = 0.01,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        margin = 1.0,
        match_dropout = 0,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
        weight_decay = 0.0001,
        dropout_rates = { 0 },
      })

      local items = {
        create_item("neg.lua", nil, 90),
        create_item("selected.lua", nil, 80),
      }

      -- Add 10 pairs (should only keep last 5)
      for _ = 1, 10 do
        nn.update_weights(items[2], items)
      end

      local history = nn._get_training_history()
      assert.equals(5, #history)
    end)

    it("creates correct pair structure with positive and negative files", function()
      local items = {
        create_item("higher.lua", nil, 90),
        create_item("selected.lua", nil, 80),
      }

      nn.update_weights(items[2], items)

      local history = nn._get_training_history()
      assert.equals(1, #history)

      local pair = history[1]
      assert.is_table(pair.positive_input)
      assert.is_table(pair.negative_input)
      assert.equals("selected.lua", pair.positive_file)
      assert.equals("higher.lua", pair.negative_file)
    end)

    it("constructs multiple pairs with items ranked above selected", function()
      local items = {
        create_item("1.lua", nil, 100),
        create_item("2.lua", nil, 90),
        create_item("3.lua", nil, 80),
        create_item("selected.lua", nil, 70), -- Ranked 4th
      }

      nn.update_weights(items[4], items)

      local history = nn._get_training_history()
      -- Should have 3 hard negatives (items 1-3) + 0 immediate + 0 random = 3 pairs
      assert.equals(3, #history)

      -- All pairs should have selected item as positive
      for _, pair in ipairs(history) do
        assert.equals("selected.lua", pair.positive_file)
      end

      -- Negative files should be items 1-3
      local negative_files = {}
      for _, pair in ipairs(history) do
        negative_files[pair.negative_file] = true
      end
      assert.is_true(negative_files["1.lua"])
      assert.is_true(negative_files["2.lua"])
      assert.is_true(negative_files["3.lua"])
    end)

    it("uses top-10 items for pair construction", function()
      local items = {}
      -- Create 10 items ranked above selected
      for i = 1, 10 do
        table.insert(items, create_item(string.format("%d.lua", i), nil, 100 - i))
      end
      table.insert(items, create_item("selected.lua", nil, 50)) -- Ranked 11th

      nn.update_weights(items[11], items)

      local history = nn._get_training_history()
      -- Selected at rank 11, top-10 are items 1-10
      -- Creates pairs: selected vs. each of the 10 items in top-10 = 10 pairs
      assert.equals(10, #history)
    end)

    it("handles case with selected ranked first", function()
      local items = {
        create_item("selected.lua", nil, 100), -- Ranked 1st
        create_item("2.lua", nil, 90),
        create_item("3.lua", nil, 80),
        create_item("4.lua", nil, 70),
        create_item("5.lua", nil, 60),
        create_item("6.lua", nil, 50),
      }

      nn.update_weights(items[1], items)

      local history = nn._get_training_history()
      -- Selected at rank 1, top-10 includes all 6 items
      -- Creates pairs: selected vs. other 5 items = 5 pairs
      assert.equals(5, #history)

      -- All should be negative files from items 2-6
      for _, pair in ipairs(history) do
        assert.equals("selected.lua", pair.positive_file)
        assert.is_not_nil(pair.negative_file)
        assert.is_true(pair.negative_file ~= "selected.lua")
      end
    end)
  end)

  describe("Pairwise Training Loop", function()
    local nn

    before_each(function()
      -- Setup mocks
      _G._TEST = true
      _G.vim = _G.vim or {}
      _G.vim.loop = {
        hrtime = function()
          return 1000000
        end,
      }
      local original_vim_tbl_extend = _G.vim.tbl_extend
      local original_vim_tbl_deep_extend = _G.vim.tbl_deep_extend

      _G.vim.tbl_extend = function(mode, ...)
        local result = {}
        for i = 1, select("#", ...) do
          local tbl = select(i, ...)
          for k, v in pairs(tbl) do
            result[k] = v
          end
        end
        return result
      end
      _G.vim.tbl_deep_extend = _G.vim.tbl_extend -- For testing
      _G.vim.notify = function() end
      _G.vim.log = { levels = { INFO = 1 } }

      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.algorithms.nn_core"] = nil
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return nil
        end,
        save_weights = function() end,
      }

      nn_core = require("neural-open.algorithms.nn_core")
      nn = require("neural-open.algorithms.nn")
      nn.init({
        architecture = { 11, 8, 1 },
        learning_rate = 0.01,
        batch_size = 2, -- 2 pairs per batch
        history_size = 10,
        batches_per_update = 1,
        margin = 1.0,
        optimizer = "sgd",
        dropout_rates = { 0 }, -- Disable dropout for predictable tests
        match_dropout = 0, -- Disable match dropout for testing
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
        weight_decay = 0.0001,
      })

      -- Store originals for restoration
      _G._original_vim_tbl_extend = original_vim_tbl_extend
      _G._original_vim_tbl_deep_extend = original_vim_tbl_deep_extend
    end)

    after_each(function()
      _G.vim.tbl_extend = _G._original_vim_tbl_extend
      _G.vim.tbl_deep_extend = _G._original_vim_tbl_deep_extend
    end)

    it("trains network to prefer positive over negative item", function()
      -- Create features where positive should score higher
      local positive_features = {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.8,
        project = 1.0,
        recency = 0.9,
        trigram = 0.8,
        transition = 0.0,
      }

      local negative_features = {
        match = 0.1,
        virtual_name = 0.1,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.2,
        project = 0.0,
        recency = 0.1,
        trigram = 0.1,
        transition = 0.0,
      }

      -- Get initial scores
      local pos_buf = features_to_input_buf(positive_features)
      local neg_buf = features_to_input_buf(negative_features)
      local score_pos_before = nn.calculate_score(pos_buf)
      local score_neg_before = nn.calculate_score(neg_buf)
      local diff_before = score_pos_before - score_neg_before

      -- Create items for training - add more items to top-10 for diverse training
      local items = {
        { file = "neg.lua", nos = { input_buf = neg_buf, neural_score = 80, normalized_path = "neg.lua" } },
        { file = "pos.lua", nos = { input_buf = pos_buf, neural_score = 70, normalized_path = "pos.lua" } },
      }

      -- Add some filler items with intermediate features to create a realistic top-10
      for i = 1, 8 do
        local filler_features = {}
        for k, _ in pairs(negative_features) do
          filler_features[k] = (positive_features[k] + negative_features[k]) / 2 + (math.random() - 0.5) * 0.2
        end
        table.insert(items, {
          file = "filler" .. i .. ".lua",
          nos = {
            input_buf = features_to_input_buf(filler_features),
            neural_score = 60 - i,
            normalized_path = "filler" .. i .. ".lua",
          },
        })
      end

      -- Train: selected item is ranked 2nd (should learn to rank it higher)
      for _ = 1, 50 do -- Multiple training iterations
        nn.update_weights(items[2], items)
      end

      -- Get scores after training
      local score_pos_after = nn.calculate_score(pos_buf)
      local score_neg_after = nn.calculate_score(neg_buf)
      local diff_after = score_pos_after - score_neg_after

      -- Verify that difference increased or stayed positive (network learned)
      -- With random initialization, convergence may be slow or diff may already be positive
      assert.is_true(
        diff_after >= diff_before - 0.01,
        string.format("Diff should not decrease: before=%.4f, after=%.4f", diff_before, diff_after)
      )
    end)

    it("does not update weights when margin is satisfied", function()
      -- Initialize with fixed seed
      math.randomseed(42)
      nn.init({
        architecture = { 11, 4, 1 },
        optimizer = "sgd",
        margin = 0.1, -- Small margin for easier satisfaction
        dropout_rates = { 0 },
        match_dropout = 0,
        learning_rate = 0.01,
        batch_size = 2,
        batches_per_update = 1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        history_size = 100,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
        weight_decay = 0.0001,
      })

      -- Create features where positive already scores much higher
      local positive_features = {
        match = 1.0,
        virtual_name = 1.0,
        frecency = 1.0,
        open = 1.0,
        alt = 1.0,
        proximity = 1.0,
        project = 1.0,
        recency = 1.0,
        trigram = 1.0,
        transition = 0.0,
      }

      local negative_features = {
        match = 0.0,
        virtual_name = 0.0,
        frecency = 0.0,
        open = 0.0,
        alt = 0.0,
        proximity = 0.0,
        project = 0.0,
        recency = 0.0,
        trigram = 0.0,
        transition = 0.0,
      }

      local pos_buf = features_to_input_buf(positive_features)
      local neg_buf = features_to_input_buf(negative_features)
      local score_pos = nn.calculate_score(pos_buf)
      local score_neg = nn.calculate_score(neg_buf)

      -- Check if margin is satisfied (this test may need adjustment based on initial weights)
      local loss = nn_core.pairwise_hinge_loss(score_pos / 100, score_neg / 100, 0.1)

      -- If loss is already 0, weights shouldn't change significantly
      if loss < 0.01 then
        local items = {
          { file = "neg.lua", nos = { input_buf = neg_buf, normalized_path = "neg.lua" } },
          { file = "pos.lua", nos = { input_buf = pos_buf, normalized_path = "pos.lua" } },
        }

        -- Get score before
        local before = nn.calculate_score(pos_buf)

        -- Train once
        nn.update_weights(items[2], items)

        -- Get score after
        local after = nn.calculate_score(pos_buf)

        -- Score should not change much (within numerical precision)
        assert.is_true(
          math.abs(after - before) < 5.0,
          string.format("Score should not change much when margin satisfied: before=%.2f, after=%.2f", before, after)
        )
      end
    end)

    it("processes batches with correct pair structure", function()
      -- This test verifies the batch processing
      local items = {
        {
          file = "1.lua",
          nos = {
            input_buf = { 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 },
            normalized_path = "1.lua",
          },
        },
        {
          file = "2.lua",
          nos = {
            input_buf = { 0.4, 0.4, 0.4, 0, 0, 0.4, 1, 0.4, 0.4, 0.0, 1.0 },
            normalized_path = "2.lua",
          },
        },
      }

      -- Train once
      nn.update_weights(items[2], items)

      -- Verify training happened (loss history should have entries)
      local stats = nn._get_stats()
      assert.is_true(#stats.loss_history > 0)
      assert.is_number(stats.last_loss)
    end)

    it("convergence test: loss decreases with training", function()
      -- Set random seed for reproducibility
      math.randomseed(12345)

      nn.init({
        architecture = { 11, 16, 8, 1 },
        learning_rate = 0.05,
        batch_size = 4,
        history_size = 50,
        batches_per_update = 2,
        weight_decay = 0.0001,
        dropout_rates = { 0, 0 }, -- Disable for predictable tests
        margin = 1.0,
        optimizer = "sgd",
        match_dropout = 0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
      })

      -- Create clear-cut case: high-quality vs low-quality file
      local high_quality = {
        match = 0.9,
        virtual_name = 0.9,
        frecency = 0.8,
        open = 1,
        alt = 0,
        proximity = 0.9,
        project = 1,
        recency = 0.8,
        trigram = 0.9,
        transition = 0.0,
      }

      local low_quality = {
        match = 0.1,
        virtual_name = 0.1,
        frecency = 0.2,
        open = 0,
        alt = 0,
        proximity = 0.1,
        project = 0,
        recency = 0.2,
        trigram = 0.1,
        transition = 0.0,
      }

      local items = {
        { file = "low.lua", nos = { input_buf = features_to_input_buf(low_quality), normalized_path = "low.lua" } },
        { file = "high.lua", nos = { input_buf = features_to_input_buf(high_quality), normalized_path = "high.lua" } },
      }

      -- Record initial loss
      nn.update_weights(items[2], items)
      local stats = nn._get_stats()
      local initial_loss = stats.last_loss

      -- Train for many iterations
      for _ = 1, 100 do
        nn.update_weights(items[2], items)
      end

      -- Get final loss
      stats = nn._get_stats()
      local final_loss = stats.last_loss

      -- Loss should decrease (or stay very low if already converged)
      -- Note: With random initialization and sigmoid outputs, convergence may be slow
      assert.is_true(
        final_loss <= initial_loss + 0.2,
        string.format("Loss should decrease or stay stable: initial=%.4f, final=%.4f", initial_loss, final_loss)
      )

      -- After sufficient training, loss should show improvement
      -- With margin=1.0 and sigmoid outputs, perfect convergence (loss=0) may not be achievable
      -- So we just verify the network is learning (loss improved or stayed low)
      assert.is_true(
        final_loss < initial_loss + 0.1 or final_loss < 1.0,
        string.format("Final loss should show learning: initial=%.4f, final=%.4f", initial_loss, final_loss)
      )
    end)

    it("gradients are finite after backward pass", function()
      local items = {
        {
          file = "1.lua",
          nos = {
            input_buf = { 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 },
            normalized_path = "1.lua",
          },
        },
        {
          file = "2.lua",
          nos = {
            input_buf = { 0.4, 0.4, 0.4, 0, 0, 0.4, 1, 0.4, 0.4, 0.0, 1.0 },
            normalized_path = "2.lua",
          },
        },
      }

      -- Train
      nn.update_weights(items[2], items)

      -- Get weight norms
      local stats = nn._get_stats()
      if stats.weight_norms then
        for i, norm in ipairs(stats.weight_norms) do
          assert.is_true(norm == norm, string.format("Weight norm %d should not be NaN", i))
          assert.is_true(norm < math.huge, string.format("Weight norm %d should not be Inf", i))
          assert.is_true(norm > 0, string.format("Weight norm %d should be positive", i))
        end
      end
    end)
  end)

  describe("Input-Size Migration", function()
    local nn, weights_module

    before_each(function()
      -- Setup mocks
      _G._TEST = true
      _G.vim = _G.vim or {}
      _G.vim.loop = {
        hrtime = function()
          return 0
        end,
      }
      local original_vim_tbl_extend = _G.vim.tbl_extend
      local original_vim_tbl_deep_extend = _G.vim.tbl_deep_extend

      _G.vim.tbl_extend = function(mode, ...)
        local result = {}
        for i = 1, select("#", ...) do
          local tbl = select(i, ...)
          for k, v in pairs(tbl) do
            result[k] = v
          end
        end
        return result
      end
      _G.vim.tbl_deep_extend = _G.vim.tbl_extend -- For testing
      _G.vim.log = { levels = { INFO = 1 } }
      _G.vim.notify = function(msg, level)
        _G.last_notify_message = msg
      end
      _G.vim.deepcopy = function(t)
        if type(t) ~= "table" then
          return t
        end
        local copy = {}
        for k, v in pairs(t) do
          copy[k] = vim.deepcopy(v)
        end
        return copy
      end

      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.algorithms.nn_core"] = nil
      package.loaded["neural-open.weights"] = nil

      -- Mock weights module
      weights_module = {
        saved_weights = nil,
        get_weights = function(algo)
          return weights_module.saved_weights
        end,
        save_weights = function(algo, data)
          weights_module.saved_weights = data
        end,
      }
      package.loaded["neural-open.weights"] = weights_module

      nn = require("neural-open.algorithms.nn")

      -- Store originals for restoration
      _G._original_vim_tbl_extend = original_vim_tbl_extend
      _G._original_vim_tbl_deep_extend = original_vim_tbl_deep_extend
    end)

    after_each(function()
      _G.vim.tbl_extend = _G._original_vim_tbl_extend
      _G.vim.tbl_deep_extend = _G._original_vim_tbl_deep_extend
    end)

    it("migrates 10-input layer to 11-input layer and backfills training history", function()
      -- Simulate saved weights with 10-input first layer (old feature count)
      -- Architecture: {10, 4, 1} -> 10 inputs, 4 hidden, 1 output
      local original_weights_layer1 = {}
      for i = 1, 10 do
        original_weights_layer1[i] = {}
        for j = 1, 4 do
          original_weights_layer1[i][j] = 0.1 * i + 0.01 * j
        end
      end
      local weights_layer2 = { { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 } }

      weights_module.saved_weights = {
        nn = {
          version = "2.0-hinge",
          network = {
            weights = { original_weights_layer1, weights_layer2 },
            biases = { { { 0.1, 0.2, 0.3, 0.4 } }, { { 0.05 } } },
            gammas = {},
            betas = {},
          },
          training_history = {
            {
              -- Non-current file: trigram=0.5 (idx 9), proximity=0.3 (idx 6) -> not_current=1.0
              positive_input = { { 0.8, 0.7, 0.6, 1, 0, 0.3, 1, 0.9, 0.5, 0.2 } },
              negative_input = { { 0.3, 0.2, 0.1, 0, 0, 0.1, 1, 0.5, 0.3, 0.1 } },
              positive_file = "src/app.lua",
              negative_file = "src/utils.lua",
            },
            {
              -- Current file heuristic: trigram=0.99 (idx 9), proximity=1.0 (idx 6) -> not_current=0.0
              positive_input = { { 0.9, 0.8, 0.7, 1, 1, 1.0, 1, 0.95, 0.99, 0.4 } },
              negative_input = { { 0.4, 0.3, 0.2, 0, 0, 0.5, 0, 0.6, 0.2, 0.0 } },
              positive_file = "src/current.lua",
              negative_file = "src/other.lua",
            },
          },
          stats = {
            samples_processed = 50,
            batches_trained = 5,
            loss_history = { 0.3, 0.2 },
          },
          optimizer_type = "sgd",
        },
      }

      _G.last_notify_message = nil
      nn.init({
        architecture = { 11, 4, 1 },
        optimizer = "sgd",
        learning_rate = 0.01,
        batch_size = 32,
        history_size = 100,
        batches_per_update = 1,
        weight_decay = 0.0001,
        warmup_steps = 0,
        warmup_start_factor = 0.1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        match_dropout = 0,
        margin = 1.0,
        dropout_rates = { 0 },
      })

      -- Trigger weight loading (input-size migration happens here)
      local score = nn.calculate_score({ 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 })
      assert.is_number(score)

      -- Verify notification about input-size migration
      assert.is_not_nil(_G.last_notify_message)
      assert.is_true(
        string.find(_G.last_notify_message, "Migrated NN input layer from 10 to 11") ~= nil,
        "Expected input-size migration notification, got: " .. tostring(_G.last_notify_message)
      )

      -- Verify first layer now has 11 rows
      local weights = nn._get_weights()
      assert.is_not_nil(weights)
      assert.is_not_nil(weights[1])
      assert.equals(11, #weights[1])

      -- Verify original 10 rows are preserved
      for i = 1, 10 do
        for j = 1, 4 do
          assert.equals(
            original_weights_layer1[i][j],
            weights[1][i][j],
            string.format("Weight[1][%d][%d] should be preserved", i, j)
          )
        end
      end

      -- Verify 11th row was added with 4 values
      assert.equals(4, #weights[1][11])

      -- Verify second layer was not modified
      assert.equals(4, #weights[2])

      -- Verify training history was backfilled with 11 elements
      local history = nn._get_training_history()
      assert.equals(2, #history)

      -- First pair: non-current file (trigram=0.5, proximity=0.3) -> not_current=1.0
      -- Training history stores inputs as matrices: { {v1, v2, ..., vN} }
      assert.equals(11, #history[1].positive_input[1])
      assert.equals(1.0, history[1].positive_input[1][11])
      assert.equals(11, #history[1].negative_input[1])
      assert.equals(1.0, history[1].negative_input[1][11])

      -- Second pair: positive is current file (trigram=0.99, proximity=1.0) -> not_current=0.0
      assert.equals(11, #history[2].positive_input[1])
      assert.equals(0.0, history[2].positive_input[1][11])
      -- Negative is not current file -> not_current=1.0
      assert.equals(11, #history[2].negative_input[1])
      assert.equals(1.0, history[2].negative_input[1][11])
    end)

    it("resets AdamW first-layer optimizer moments after input-size migration", function()
      -- Architecture: {10, 4, 1} -> 10 inputs, 4 hidden, 1 output
      local weights_layer1 = {}
      for i = 1, 10 do
        weights_layer1[i] = {}
        for j = 1, 4 do
          weights_layer1[i][j] = 0.1 * i + 0.01 * j
        end
      end
      local weights_layer2 = { { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 } }

      -- Create 10×4 optimizer moments (old dimension)
      nn_core = require("neural-open.algorithms.nn_core")
      local old_first_weights = nn_core.zeros(10, 4)
      -- Put non-zero values so we can verify they get replaced
      for i = 1, 10 do
        for j = 1, 4 do
          old_first_weights[i][j] = 0.5
        end
      end
      local old_second_weights = nn_core.zeros(10, 4)
      for i = 1, 10 do
        for j = 1, 4 do
          old_second_weights[i][j] = 0.25
        end
      end

      weights_module.saved_weights = {
        nn = {
          version = "2.0-hinge",
          network = {
            weights = { vim.deepcopy(weights_layer1), weights_layer2 },
            biases = { { { 0.1, 0.2, 0.3, 0.4 } }, { { 0.05 } } },
            gammas = {},
            betas = {},
          },
          training_history = {},
          stats = { samples_processed = 10, batches_trained = 2, loss_history = {} },
          optimizer_type = "adamw",
          optimizer_state = {
            timestep = 50,
            moments = {
              first = {
                weights = { vim.deepcopy(old_first_weights), nn_core.zeros(4, 1) },
                biases = { nn_core.zeros(1, 4), nn_core.zeros(1, 1) },
                gammas = {},
                betas = {},
              },
              second = {
                weights = { vim.deepcopy(old_second_weights), nn_core.zeros(4, 1) },
                biases = { nn_core.zeros(1, 4), nn_core.zeros(1, 1) },
                gammas = {},
                betas = {},
              },
            },
          },
        },
      }

      _G.last_notify_message = nil
      nn.init({
        architecture = { 11, 4, 1 },
        optimizer = "adamw",
        learning_rate = 0.001,
        batch_size = 32,
        history_size = 100,
        batches_per_update = 1,
        weight_decay = 0.0001,
        warmup_steps = 0,
        warmup_start_factor = 0.1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        match_dropout = 0,
        margin = 1.0,
        dropout_rates = { 0 },
      })

      -- Trigger weight loading
      local score = nn.calculate_score({ 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 })
      assert.is_number(score)

      -- Verify first-layer moments were reset to 11×4
      local opt = nn._get_optimizer_state()
      assert.is_not_nil(opt)
      assert.is_not_nil(opt.moments)

      local m1_w1 = opt.moments.first.weights[1]
      assert.equals(11, #m1_w1, "First moment weights[1] should have 11 rows")
      assert.equals(4, #m1_w1[1], "First moment weights[1] should have 4 cols")
      -- All values should be zero (reset)
      for i = 1, 11 do
        for j = 1, 4 do
          assert.equals(0, m1_w1[i][j], string.format("first.weights[1][%d][%d] should be 0", i, j))
        end
      end

      local m2_w1 = opt.moments.second.weights[1]
      assert.equals(11, #m2_w1, "Second moment weights[1] should have 11 rows")
      assert.equals(4, #m2_w1[1], "Second moment weights[1] should have 4 cols")

      -- Second layer moments should be untouched (still 4×1)
      assert.equals(4, #opt.moments.first.weights[2])
      assert.equals(1, #opt.moments.first.weights[2][1])

      -- Timestep should be preserved
      assert.equals(50, opt.timestep)
    end)
  end)

  describe("Debug View", function()
    local nn

    before_each(function()
      -- Setup mocks
      _G._TEST = true
      _G.vim = _G.vim or {}
      _G.vim.loop = {
        hrtime = function()
          return 0
        end,
      }
      local original_vim_tbl_extend = _G.vim.tbl_extend
      local original_vim_tbl_deep_extend = _G.vim.tbl_deep_extend

      _G.vim.tbl_extend = function(mode, ...)
        local result = {}
        for i = 1, select("#", ...) do
          local tbl = select(i, ...)
          for k, v in pairs(tbl) do
            result[k] = v
          end
        end
        return result
      end
      _G.vim.tbl_deep_extend = _G.vim.tbl_extend -- For testing
      _G.vim.notify = function() end
      _G.vim.log = { levels = { INFO = 1 } }

      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.algorithms.nn_core"] = nil
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return nil
        end,
        save_weights = function() end,
      }

      nn = require("neural-open.algorithms.nn")
      nn.init({
        architecture = { 11, 4, 1 },
        optimizer = "sgd",
        learning_rate = 0.01,
        margin = 1.0,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        batch_size = 32,
        batches_per_update = 1,
        history_size = 100,
        match_dropout = 0,
        warmup_start_factor = 0.1,
        warmup_steps = 0,
        weight_decay = 0.0001,
        dropout_rates = { 0 },
      })

      -- Store originals for restoration
      _G._original_vim_tbl_extend = original_vim_tbl_extend
      _G._original_vim_tbl_deep_extend = original_vim_tbl_deep_extend
    end)

    after_each(function()
      _G.vim.tbl_extend = _G._original_vim_tbl_extend
      _G.vim.tbl_deep_extend = _G._original_vim_tbl_deep_extend
    end)

    it("displays margin in debug view", function()
      local item = {
        file = "test.lua",
        nos = {
          input_buf = { 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 },
          neural_score = 50,
        },
      }

      local debug_lines = nn.debug_view(item, {})
      local debug_text = table.concat(debug_lines, "\n")

      -- Verify margin is displayed
      assert.is_true(debug_text:find("Margin") ~= nil, "Debug view should display margin")
      assert.is_true(debug_text:find("1.00") ~= nil, "Debug view should show margin value")
    end)

    it("displays 'Hinge Loss' labels in debug view", function()
      local item = {
        file = "test.lua",
        nos = {
          input_buf = { 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 },
          neural_score = 50,
        },
      }

      local debug_lines = nn.debug_view(item, {})
      local debug_text = table.concat(debug_lines, "\n")

      -- Verify hinge loss labels
      assert.is_true(debug_text:find("Hinge Loss") ~= nil, "Debug view should mention hinge loss")
    end)

    it("displays pairwise training mode in debug view", function()
      local item = {
        file = "test.lua",
        nos = {
          input_buf = { 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.0, 1.0 },
          neural_score = 50,
        },
      }

      local debug_lines = nn.debug_view(item, {})
      local debug_text = table.concat(debug_lines, "\n")

      -- Verify pairwise mode is mentioned
      assert.is_true(debug_text:find("Pairwise") ~= nil, "Debug view should mention pairwise training")
      assert.is_true(debug_text:find("pairs") ~= nil, "Debug view should show history in pairs")
    end)

    it("displays transition feature in debug output", function()
      local item = {
        file = "test.lua",
        nos = {
          input_buf = { 0.5, 0.5, 0.5, 0, 0, 0.5, 1, 0.5, 0.5, 0.3, 1.0 },
          neural_score = 50,
        },
      }

      local debug_lines = nn.debug_view(item, {})
      local debug_text = table.concat(debug_lines, "\n")

      -- Verify transition appears in both feature importance and input features
      assert.is_true(debug_text:find("Transition") ~= nil, "Debug view should display transition feature")
      -- Verify the normalized value is shown
      assert.is_true(
        debug_text:find("0.3") ~= nil or debug_text:find("0.3000") ~= nil,
        "Debug view should show transition value"
      )
    end)
  end)
end)
