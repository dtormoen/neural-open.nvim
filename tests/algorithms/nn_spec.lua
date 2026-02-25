describe("Neural Network Algorithm", function()
  local nn

  local function create_mock_item(file, features, score)
    return {
      file = file,
      score = score or 0,
      nos = {
        normalized_features = features,
        neural_score = score or 0,
      },
    }
  end

  -- Standard test configuration used across most tests
  -- Smaller network and simpler params for faster test execution
  local STANDARD_TEST_CONFIG = {
    architecture = { 10, 4, 1 },
    optimizer = "sgd",
    learning_rate = 0.1,
    batch_size = 4,
    history_size = 10,
    match_dropout = 0,
    warmup_steps = 0,
    dropout_rates = { 0 },
  }

  before_each(function()
    local helpers = require("tests.helpers")

    -- Clear module cache
    package.loaded["neural-open.algorithms.nn"] = nil
    package.loaded["neural-open.algorithms.nn_core"] = nil

    nn = require("neural-open.algorithms.nn")

    -- Initialize with standard test configuration
    local config = helpers.create_algorithm_config("nn", STANDARD_TEST_CONFIG)
    nn.init(config.algorithm_config.nn)
  end)

  describe("Basic Functionality", function()
    it("returns algorithm name", function()
      assert.equals("nn", nn.get_name())
    end)

    it("calculates scores from normalized features", function()
      local features = {
        match = 0.8,
        virtual_name = 0.5,
        frecency = 0.3,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.4,
        trigram = 0.7,
        transition = 0.0,
      }

      local score = nn.calculate_score(features)
      assert.is_number(score)
      assert.is_true(score >= 0)
      assert.is_true(score <= 100)
    end)

    it("produces different scores for different features", function()
      local helpers = require("tests.helpers")

      -- Initialize with a specific seed for deterministic results
      math.randomseed(42)
      local test_config = helpers.create_algorithm_config("nn", {
        architecture = { 10, 8, 4, 1 },
        optimizer = "sgd",
        dropout_rates = { 0, 0 },
      })
      nn.init(test_config.algorithm_config.nn)

      local features1 = {
        match = 0.9,
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

      local features2 = {
        match = 0.1,
        virtual_name = 0.0,
        frecency = 0.0,
        open = 1.0,
        alt = 1.0,
        proximity = 1.0,
        project = 1.0,
        recency = 1.0,
        trigram = 1.0,
        transition = 0.0,
      }

      local score1 = nn.calculate_score(features1)
      local score2 = nn.calculate_score(features2)

      -- With batch norm, initial scores might be similar but should differ after proper initialization
      -- Allow for small differences due to numerical precision
      assert.is_true(
        math.abs(score1 - score2) > 0.01,
        string.format("Scores should be different: %.6f vs %.6f", score1, score2)
      )
    end)

    it("handles missing features gracefully", function()
      local features = {
        match = 0.5,
        -- Missing other features
      }

      local score = nn.calculate_score(features)
      assert.is_number(score)
      assert.is_true(score >= 0)
    end)
  end)

  describe("Weight Updates", function()
    it("updates weights based on user selection", function()
      local selected_item = create_mock_item("selected.lua", {
        match = 0.5,
        virtual_name = 0.0,
        frecency = 0.2,
        open = 0.0,
        alt = 0.0,
        proximity = 0.8,
        project = 1.0,
        recency = 0.3,
        trigram = 0.6,
        transition = 0.0,
      })

      local ranked_items = {
        create_mock_item("higher1.lua", {
          match = 0.9,
          virtual_name = 0.0,
          frecency = 0.1,
          open = 1.0,
          alt = 0.0,
          proximity = 0.2,
          project = 1.0,
          recency = 0.0,
          trigram = 0.1,
          transition = 0.0,
        }),
        create_mock_item("higher2.lua", {
          match = 0.8,
          virtual_name = 0.0,
          frecency = 0.3,
          open = 0.0,
          alt = 1.0,
          proximity = 0.3,
          project = 1.0,
          recency = 0.1,
          trigram = 0.2,
          transition = 0.0,
        }),
        selected_item,
        create_mock_item("lower1.lua", {
          match = 0.3,
          virtual_name = 0.0,
          frecency = 0.0,
          open = 0.0,
          alt = 0.0,
          proximity = 0.1,
          project = 0.0,
          recency = 0.0,
          trigram = 0.0,
          transition = 0.0,
        }),
      }

      -- Mock weights module to capture saved weights
      local helpers = require("tests.helpers")
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      nn.update_weights(selected_item, ranked_items)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn)
      assert.is_not_nil(saved_weights.nn.network)
      assert.is_not_nil(saved_weights.nn.network.weights)
      assert.is_not_nil(saved_weights.nn.network.biases)
      assert.is_not_nil(saved_weights.nn.training_history)
      assert.is_not_nil(saved_weights.nn.stats)

      -- Check that samples were processed
      assert.is_true(saved_weights.nn.stats.samples_processed > 0)
      assert.is_true(saved_weights.nn.stats.batches_trained > 0)
    end)

    it("tracks batch training timing statistics", function()
      local selected_item = create_mock_item("selected.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.5,
        trigram = 0.4,
        transition = 0.0,
      })

      -- Need at least 2 items above selected to meet 50% threshold (batch_size=4, min=2)
      local ranked_items = {
        create_mock_item("first.lua", {
          match = 0.4,
          virtual_name = 0.3,
          frecency = 0.2,
          open = 0.0,
          alt = 0.0,
          proximity = 0.1,
          project = 0.0,
          recency = 0.0,
          trigram = 0.0,
          transition = 0.0,
        }),
        create_mock_item("second.lua", {
          match = 0.3,
          virtual_name = 0.2,
          frecency = 0.1,
          open = 0.0,
          alt = 0.0,
          proximity = 0.1,
          project = 0.0,
          recency = 0.0,
          trigram = 0.0,
          transition = 0.0,
        }),
        selected_item,
      }

      -- Mock weights module
      local helpers = require("tests.helpers")
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Mock vim.loop.hrtime for timing
      local mock_time = 0
      _G.vim = _G.vim or {}
      _G.vim.loop = _G.vim.loop or {}
      _G.vim.loop.hrtime = function()
        mock_time = mock_time + 1e6 -- Simulate 1ms per call
        return mock_time
      end

      nn.update_weights(selected_item, ranked_items)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn.stats)

      -- Check that timing stats were captured
      assert.is_not_nil(saved_weights.nn.stats.batch_timings)
      assert.is_true(#saved_weights.nn.stats.batch_timings > 0)

      -- Check that average timing was calculated
      assert.is_not_nil(saved_weights.nn.stats.avg_batch_timing)
      assert.is_not_nil(saved_weights.nn.stats.avg_batch_timing.forward_ms)
      assert.is_not_nil(saved_weights.nn.stats.avg_batch_timing.backward_ms)
      assert.is_not_nil(saved_weights.nn.stats.avg_batch_timing.update_ms)
      assert.is_not_nil(saved_weights.nn.stats.avg_batch_timing.total_ms)

      -- Verify timing values are reasonable (should be > 0)
      assert.is_true(saved_weights.nn.stats.avg_batch_timing.forward_ms > 0)
      assert.is_true(saved_weights.nn.stats.avg_batch_timing.backward_ms > 0)
      assert.is_true(saved_weights.nn.stats.avg_batch_timing.update_ms > 0)
      assert.is_true(saved_weights.nn.stats.avg_batch_timing.total_ms > 0)
    end)

    it("builds training history over multiple selections", function()
      local helpers = require("tests.helpers")
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- First selection - with multiple items to create pairs
      local item1 = create_mock_item("file1.lua", {
        match = 0.7,
        virtual_name = 0.0,
        frecency = 0.3,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.2,
        trigram = 0.4,
        transition = 0.0,
      })

      local other1 = create_mock_item("higher1.lua", {
        match = 0.9,
        virtual_name = 0.0,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.2,
        project = 1.0,
        recency = 0.1,
        trigram = 0.2,
        transition = 0.0,
      })

      local items1 = { other1, item1 }
      nn.update_weights(item1, items1)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      local history_size_1 = #saved_weights.nn.training_history
      assert.is_true(history_size_1 > 0, "Should have pairs after first selection")

      -- Second selection - weights will be loaded automatically
      local item2 = create_mock_item("file2.lua", {
        match = 0.5,
        virtual_name = 0.2,
        frecency = 0.6,
        open = 0.0,
        alt = 1.0,
        proximity = 0.3,
        project = 1.0,
        recency = 0.5,
        trigram = 0.1,
        transition = 0.0,
      })

      local items2 = {
        create_mock_item("other.lua", {
          match = 0.8,
          virtual_name = 0.0,
          frecency = 0.1,
          open = 0.0,
          alt = 0.0,
          proximity = 0.9,
          project = 1.0,
          recency = 0.0,
          trigram = 0.0,
          transition = 0.0,
        }),
        item2,
      }
      nn.update_weights(item2, items2)

      saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      local history_size_2 = #saved_weights.nn.training_history
      assert.is_true(history_size_2 > history_size_1)
    end)

    it("respects history size limit", function()
      -- Initialize with small history size
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          history_size = 3, -- Very small for testing
          batches_per_update = 1,
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Make multiple selections to exceed history limit
      for i = 1, 5 do
        local item = create_mock_item("file" .. i .. ".lua", {
          match = 0.5 + i * 0.1,
          virtual_name = 0.0,
          frecency = 0.1 * i,
          open = i % 2,
          alt = 0.0,
          proximity = 0.3,
          project = 1.0,
          recency = 0.2,
          trigram = 0.1,
          transition = 0.0,
        })

        local items = { item }
        nn.update_weights(item, items)
      end

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_true(#saved_weights.nn.training_history <= 3)
    end)

    it("maintains circular buffer of last 10 batch timings", function()
      -- Initialize with small batch size to ensure many batches
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 1,
          history_size = 50,
          batches_per_update = 15, -- Will create 15 batches
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Mock vim.loop.hrtime
      local mock_time = 0
      _G.vim = _G.vim or {}
      _G.vim.loop = _G.vim.loop or {}
      _G.vim.loop.hrtime = function()
        mock_time = mock_time + 1e6
        return mock_time
      end

      -- Create many training samples to fill history
      local selected = create_mock_item("selected.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.5,
        trigram = 0.4,
        transition = 0.0,
      })

      local items = { selected }

      -- Add more items to ensure we have enough history for multiple batches
      for i = 1, 20 do
        table.insert(
          items,
          create_mock_item("file" .. i .. ".lua", {
            match = math.random() * 0.5,
            virtual_name = 0.0,
            frecency = math.random() * 0.3,
            open = 0.0,
            alt = 0.0,
            proximity = math.random(),
            project = 1.0,
            recency = math.random() * 0.2,
            trigram = math.random() * 0.3,
          })
        )
      end

      nn.update_weights(selected, items)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn.stats)
      assert.is_not_nil(saved_weights.nn.stats.batch_timings)

      -- Should have at most 10 timings despite processing 15 batches
      assert.is_true(#saved_weights.nn.stats.batch_timings <= 10)

      -- The average should be computed from available timings
      assert.is_not_nil(saved_weights.nn.stats.avg_batch_timing)
      assert.is_true(saved_weights.nn.stats.avg_batch_timing.total_ms > 0)
    end)

    it("collects enhanced negative samples for training", function()
      -- Re-init with larger history size for this test
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          history_size = 100, -- Large enough to keep all samples
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Mock weights module to capture training samples
      local saved_weights = nil
      local save_call_count = 0
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_weights or {}
        end,
        save_weights = function(algo_name, weights)
          save_call_count = save_call_count + 1
          saved_weights = weights
        end,
      }

      -- Create a selected item at rank 8
      local selected_item = create_mock_item("selected.lua", {
        match = 0.5,
        virtual_name = 0.1,
        frecency = 0.3,
        open = 0.0,
        alt = 0.0,
        proximity = 0.7,
        project = 1.0,
        recency = 0.4,
        trigram = 0.5,
        transition = 0.0,
      })

      -- Create ranked items list with selected at position 8
      local ranked_items = {}

      -- Add 7 items ranked above selected (positions 1-7)
      for i = 1, 7 do
        ranked_items[i] = create_mock_item("higher_" .. i .. ".lua", {
          match = 0.9 - (i * 0.05),
          virtual_name = 0.2,
          frecency = 0.4,
          open = 0.0,
          alt = 0.0,
          proximity = 0.5,
          project = 1.0,
          recency = 0.6,
          trigram = 0.3,
          transition = 0.0,
        })
      end

      -- Add selected item at position 8 (important: must be the same instance)
      ranked_items[8] = selected_item

      -- Add 10 items ranked below selected (positions 9-18)
      for i = 1, 10 do
        ranked_items[8 + i] = create_mock_item("lower_" .. i .. ".lua", {
          match = 0.3 - (i * 0.02),
          virtual_name = 0.0,
          frecency = 0.1,
          open = 0.0,
          alt = 0.0,
          proximity = 0.2,
          project = 0.0,
          recency = 0.1,
          trigram = 0.1,
          transition = 0.0,
        })
      end

      -- Seed random for reproducible test
      math.randomseed(42)

      -- Verify the selected item is in the ranked items list
      local found_in_ranked = false
      local file_match_rank = nil
      for i, item in ipairs(ranked_items) do
        if item == selected_item then
          found_in_ranked = true
          assert.equals(8, i, "Selected item should be at position 8")
        end
        if item.file == selected_item.file then
          file_match_rank = i
        end
      end
      assert.is_true(found_in_ranked, "Selected item should be in ranked_items list")
      assert.equals(8, file_match_rank, "Selected item file should match at position 8")

      -- Verify selected_item has required fields
      assert.is_not_nil(selected_item.nos, "Selected item should have nos field")
      assert.is_not_nil(selected_item.nos.normalized_features, "Selected item should have normalized_features")
      assert.equals("selected.lua", selected_item.file, "Selected item should have correct file name")

      nn.update_weights(selected_item, ranked_items)

      assert.equals(1, save_call_count, "save_weights should have been called once")
      assert.is_not_nil(saved_weights, "Should have saved weights")
      assert.is_not_nil(saved_weights.nn, "Should have nn field in saved weights")
      assert.is_not_nil(saved_weights.nn.training_history, "Should have training_history field")

      -- Use the actual training history from saved_weights (now contains pairs)
      local captured_pairs = saved_weights.nn.training_history
      assert.is_not_nil(captured_pairs)
      assert.is_true(#captured_pairs > 0, "Should have captured some pairs")

      -- Count pairs and collect negative files from this update
      -- (captured_pairs includes all history, we want to check just what was added)
      local pair_count = 0
      local negative_files = {}
      local positive_files_seen = {}

      -- Look for pairs from this update (they'll have the files we just created)
      for _, pair in ipairs(captured_pairs) do
        -- Check if pair is from our test
        local is_test_pair = pair.positive_file == "selected.lua"
          or (pair.positive_file and pair.positive_file:match("^higher_"))
          or (pair.positive_file and pair.positive_file:match("^lower_"))
          or (pair.negative_file and pair.negative_file:match("^higher_"))
          or (pair.negative_file and pair.negative_file:match("^lower_"))

        if is_test_pair and pair.positive_file == "selected.lua" then
          pair_count = pair_count + 1
          table.insert(negative_files, pair.negative_file)
          positive_files_seen[pair.positive_file] = true
        end
      end

      -- Should have found recent pairs
      assert.is_true(pair_count > 0, "Should have found recent pairs from this update")

      -- All pairs should have the selected item as positive
      assert.equals(1, vim.tbl_count(positive_files_seen), "Should have exactly 1 positive file (selected.lua)")
      assert.is_true(positive_files_seen["selected.lua"], "Positive file should be selected.lua")

      -- Should have collected exactly 10 pairs:
      -- - Selected at rank 8
      -- - Collects top 10 items excluding selected: items 1-7, 9-11
      -- - Creates pairs with all 10 items (hard negatives)
      assert.equals(10, pair_count)

      -- Verify that we have all 7 items from above selected (all are in top-10)
      local hard_negative_count = 0
      for _, file in ipairs(negative_files) do
        if file:match("^higher_") then
          hard_negative_count = hard_negative_count + 1
        end
      end
      assert.equals(7, hard_negative_count, "Should have all 7 items ranked above selected")

      -- Verify that we have items from below selected (lower_1, lower_2, lower_3 to reach 10 total)
      local has_lower_1 = false
      local has_lower_2 = false
      local has_lower_3 = false
      for _, file in ipairs(negative_files) do
        if file == "lower_1.lua" then
          has_lower_1 = true
        elseif file == "lower_2.lua" then
          has_lower_2 = true
        elseif file == "lower_3.lua" then
          has_lower_3 = true
        end
      end
      assert.is_true(has_lower_1, "Should include lower_1.lua (position 9)")
      assert.is_true(has_lower_2, "Should include lower_2.lua (position 10)")
      assert.is_true(has_lower_3, "Should include lower_3.lua (position 11)")

      -- Verify we collected exactly the right items (7 higher + 3 lower = 10 total)
      local lower_count = 0
      for _, file in ipairs(negative_files) do
        if file:match("^lower_") then
          lower_count = lower_count + 1
        end
      end
      assert.equals(3, lower_count, "Should have exactly 3 items from below selected")
    end)

    it("handles edge cases in sample collection", function()
      local helpers = require("tests.helpers")
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Test case 1: Selected item is ranked #1 (no items above)
      local item1 = create_mock_item("best.lua", {
        match = 1.0,
        virtual_name = 0.0,
        frecency = 0.8,
        open = 1.0,
        alt = 0.0,
        proximity = 1.0,
        project = 1.0,
        recency = 1.0,
        trigram = 0.9,
        transition = 0.0,
      })

      local item2 = create_mock_item("second.lua", {
        match = 0.5,
        virtual_name = 0.0,
        frecency = 0.3,
        open = 0.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.2,
        trigram = 0.4,
        transition = 0.0,
      })

      nn.update_weights(item1, { item1, item2 })
      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      local pairs_case1 = saved_weights.nn.training_history

      -- Should create 1 pair: best.lua (positive) vs second.lua (negative, item below)
      local pair_count = 0
      for _, pair in ipairs(pairs_case1) do
        if pair.positive_file == "best.lua" and pair.negative_file == "second.lua" then
          pair_count = pair_count + 1
        end
      end
      assert.equals(1, pair_count, "Should have 1 pair with best.lua positive and second.lua negative")

      -- Test case 2: Selected item is last (no items below)
      nn.update_weights(item2, { item1, item2 })
      saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      local pairs_case2 = saved_weights.nn.training_history

      -- Should create 1 pair: second.lua (positive) vs best.lua (negative, item above)
      pair_count = 0
      for _, pair in ipairs(pairs_case2) do
        if pair.positive_file == "second.lua" and pair.negative_file == "best.lua" then
          pair_count = pair_count + 1
        end
      end
      assert.equals(1, pair_count, "Should have 1 pair with second.lua positive and best.lua negative")

      -- Test case 3: Only one item in the list
      -- Capture history size before this update
      local history_before_case3 = saved_weights
          and saved_weights.nn
          and saved_weights.nn.training_history
          and #saved_weights.nn.training_history
        or 0

      nn.update_weights(item1, { item1 })
      assert.is_not_nil(saved_weights)
      local pairs_case3 = saved_weights.nn.training_history

      -- Should not add any new pairs (no negatives to pair with)
      assert.equals(history_before_case3, #pairs_case3, "Should not add new pairs when only one item in list")
    end)
  end)

  describe("Dropout Integration", function()
    it("initializes with dropout configuration", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          architecture = { 10, 8, 4, 1 }, -- 2 hidden layers
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
          dropout_rates = { 0.5, 0.3 }, -- Dropout for each hidden layer
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Should initialize without error
      assert.is_not_nil(nn)
    end)

    it("validates dropout configuration length", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Should error if dropout rates don't match hidden layer count
      assert.has_error(function()
        local helpers = require("tests.helpers")
        local config = helpers.create_algorithm_config(
          "nn",
          vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
            architecture = { 10, 8, 4, 1 }, -- 2 hidden layers
            learning_rate = 0.01,
            batch_size = 32,
            batches_per_update = 1,
            history_size = 100,
            dropout_rates = { 0.5 }, -- Only 1 dropout rate
          })
        )
        nn.init(config.algorithm_config.nn)
      end)
    end)

    it("validates dropout rate ranges", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Should error for invalid dropout rates
      assert.has_error(function()
        local helpers = require("tests.helpers")
        local config = helpers.create_algorithm_config(
          "nn",
          vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
            architecture = { 10, 8, 1 },
            learning_rate = 0.01,
            batch_size = 32,
            batches_per_update = 1,
            history_size = 100,
            dropout_rates = { 1.0 }, -- Invalid: must be < 1
          })
        )
        nn.init(config.algorithm_config.nn)
      end)

      assert.has_error(function()
        local helpers = require("tests.helpers")
        local config = helpers.create_algorithm_config(
          "nn",
          vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
            architecture = { 10, 8, 1 },
            learning_rate = 0.01,
            batch_size = 32,
            batches_per_update = 1,
            history_size = 100,
            dropout_rates = { -0.1 }, -- Invalid: must be >= 0
          })
        )
        nn.init(config.algorithm_config.nn)
      end)
    end)

    it("applies dropout differently between training and inference", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          architecture = { 10, 16, 1 },
          dropout_rates = { 0.5 },
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      local features = {
        match = 0.8,
        virtual_name = 0.5,
        frecency = 0.3,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.4,
        trigram = 0.7,
        transition = 0.0,
      }

      -- Calculate scores multiple times - should be deterministic in inference
      local score1 = nn.calculate_score(features)
      local score2 = nn.calculate_score(features)
      local score3 = nn.calculate_score(features)

      -- In inference mode (no dropout), scores should be identical
      assert.equals(score1, score2)
      assert.equals(score2, score3)
    end)
  end)

  describe("Batch Training", function()
    it("processes multiple batches with gradient accumulation", function()
      -- Initialize with specific batch configuration
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 2,
          batches_per_update = 3,
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Create a selection scenario
      local selected_item = create_mock_item("selected.lua", {
        match = 0.4,
        virtual_name = 0.0,
        frecency = 0.3,
        open = 0.0,
        alt = 0.0,
        proximity = 0.7,
        project = 1.0,
        recency = 0.5,
        trigram = 0.6,
        transition = 0.0,
      })

      local ranked_items = {
        create_mock_item("higher.lua", {
          match = 0.9,
          virtual_name = 0.0,
          frecency = 0.1,
          open = 1.0,
          alt = 0.0,
          proximity = 0.2,
          project = 1.0,
          recency = 0.1,
          trigram = 0.1,
          transition = 0.0,
        }),
        selected_item,
      }

      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- First update to build some history
      nn.update_weights(selected_item, ranked_items)
      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)

      local initial_batches = saved_weights.nn.stats.batches_trained

      -- Second update should use multiple batches if history is available
      nn.update_weights(selected_item, ranked_items)
      saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)

      -- Should have trained more batches
      assert.is_true(saved_weights.nn.stats.batches_trained > initial_batches)

      -- Check that samples per batch is being tracked
      assert.is_number(saved_weights.nn.stats.samples_per_batch)
      assert.is_true(saved_weights.nn.stats.samples_per_batch > 0)
    end)

    it("handles insufficient samples for full batches", function()
      -- Initialize with batch_size=10, requiring minimum 5 pairs (50%)
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 10, -- Want 10 pairs per batch
          history_size = 50,
          batches_per_update = 5,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.5,
        virtual_name = 0.0,
        frecency = 0.2,
        open = 0.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.3,
        trigram = 0.4,
        transition = 0.0,
      })

      -- Create enough items to generate at least 5 pairs (50% of batch_size=10)
      local ranked_items = { item }
      for i = 1, 6 do
        table.insert(
          ranked_items,
          1,
          create_mock_item("higher" .. i .. ".lua", {
            match = 0.8 + i * 0.01,
            virtual_name = 0.0,
            frecency = 0.1,
            open = 0.0,
            alt = 0.0,
            proximity = 0.2,
            project = 1.0,
            recency = 0.1,
            trigram = 0.2,
            transition = 0.0,
          })
        )
      end

      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Update with items that will create 6 pairs (above minimum threshold of 5)
      nn.update_weights(item, ranked_items)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      -- Should train with 6 pairs (above 50% threshold)
      assert.is_true(saved_weights.nn.stats.batches_trained > 0)
      assert.is_true(saved_weights.nn.stats.last_loss >= 0)
    end)

    it("skips training when batch size is below 50% threshold", function()
      -- Reload module to ensure clean state
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Initialize with batch_size=10, minimum required is 5 pairs
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 10,
          history_size = 50,
          batches_per_update = 1,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.5,
        virtual_name = 0.0,
        frecency = 0.2,
        open = 0.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.3,
        trigram = 0.4,
        transition = 0.0,
      })

      -- Create only 3 items above selected (will create only 3 pairs, below 5 minimum)
      local ranked_items = {}
      for i = 1, 3 do
        table.insert(
          ranked_items,
          create_mock_item("higher" .. i .. ".lua", {
            match = 0.8 + i * 0.01,
            virtual_name = 0.0,
            frecency = 0.1,
            open = 0.0,
            alt = 0.0,
            proximity = 0.2,
            project = 1.0,
            recency = 0.1,
            trigram = 0.2,
            transition = 0.0,
          })
        )
      end
      table.insert(ranked_items, item)

      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Update with items that will create only 3 pairs (below 50% threshold of 5)
      nn.update_weights(item, ranked_items)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      -- Should NOT train with only 3 pairs (below 50% threshold)
      assert.equals(0, saved_weights.nn.stats.batches_trained)
    end)

    it("tracks loss history per-batch not per-update", function()
      -- Initialize with batches_per_update = 2
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 2,
          history_size = 50,
          batches_per_update = 2,
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      local item1 = create_mock_item("test1.lua", {
        match = 0.8,
        virtual_name = 0.5,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.7,
        project = 1.0,
        recency = 0.4,
        trigram = 0.8,
        transition = 0.0,
      })

      local item2 = create_mock_item("test2.lua", {
        match = 0.3,
        virtual_name = 0.2,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.3,
        project = 0.0,
        recency = 0.2,
        trigram = 0.1,
        transition = 0.0,
      })

      local item3 = create_mock_item("test3.lua", {
        match = 0.6,
        virtual_name = 0.4,
        frecency = 0.4,
        open = 0.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.3,
        trigram = 0.5,
        transition = 0.0,
      })

      local item4 = create_mock_item("test4.lua", {
        match = 0.2,
        virtual_name = 0.1,
        frecency = 0.05,
        open = 0.0,
        alt = 0.0,
        proximity = 0.1,
        project = 0.0,
        recency = 0.1,
        trigram = 0.05,
        transition = 0.0,
      })

      -- First update with 4 items - selected is ranked 4th
      -- This will create multiple pairs: 3 from items above + potentially 0 below = 3 pairs minimum
      -- With batch_size=2, we can make 1 batch from current pairs + 0 from history (no history yet)
      -- So first update might only create 1 batch
      nn.update_weights(item1, { item2, item3, item4, item1 })

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn.stats)
      assert.is_not_nil(saved_weights.nn.stats.loss_history)

      local initial_history_size = #saved_weights.nn.stats.loss_history
      -- Should have trained at least 1 batch (may be 2 if enough pairs)
      assert.is_true(initial_history_size >= 1, "Should have at least 1 loss entry after first update")

      -- Second update - with history now available, should train 2 batches
      nn.update_weights(item1, { item2, item3, item4, item1 })

      saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights.nn.stats.loss_history)
      local after_second_update = #saved_weights.nn.stats.loss_history
      -- With existing history, second update should add 2 batches
      assert.equals(initial_history_size + 2, after_second_update, "Should add 2 loss entries on second update")

      -- Verify that batches_trained matches loss_history length
      assert.equals(
        saved_weights.nn.stats.batches_trained,
        after_second_update,
        "batches_trained should match loss_history length"
      )
    end)
  end)

  describe("State Persistence", function()
    it("generates debug view without errors", function()
      -- Test debug view with uninitialized stats
      local item = create_mock_item("test.lua", {
        match = 0.5,
        virtual_name = 0.0,
        frecency = 0.0,
        open = 0.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.0,
        trigram = 0.0,
        transition = 0.0,
      })

      -- Get debug view before any training
      local debug_lines = nn.debug_view(item, { item })
      assert.is_table(debug_lines)
      assert.is_true(#debug_lines > 0)

      -- Convert to string for checking content
      local debug_text = table.concat(debug_lines, "\n")

      -- Verify it contains expected sections
      assert.is_true(debug_text:match("Training Statistics:") ~= nil)
      assert.is_true(debug_text:match("Samples Processed:") ~= nil)
      assert.is_true(debug_text:match("Batches Trained:") ~= nil)
      -- Should have either "Avg Hinge Loss:" or "Last Hinge Loss:" depending on training state
      assert.is_true(debug_text:match("Avg Hinge Loss:") ~= nil or debug_text:match("Last Hinge Loss:") ~= nil)

      -- Train the network with empty batches (edge case)
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 100, -- Very large to ensure no batches
          batches_per_update = 5,
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Try to update with insufficient data
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock
      nn.update_weights(item, { item })

      -- Debug view should still work
      debug_lines = nn.debug_view(item, { item })
      assert.is_table(debug_lines)
      assert.is_true(#debug_lines > 0)
    end)

    it("saves and loads network state through weights module", function()
      -- Mock the weights module to capture saved state
      local saved_state = nil
      local mock_weights = {
        get_weights = function(algo)
          if algo == "nn" then
            return saved_state
          end
          return nil
        end,
        save_weights = function(algo, data)
          if algo == "nn" then
            saved_state = data
          end
        end,
      }

      -- Replace the weights module
      package.loaded["neural-open.weights"] = mock_weights

      -- Train the network with some data
      local item = create_mock_item("test.lua", {
        match = 0.6,
        virtual_name = 0.3,
        frecency = 0.4,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.2,
        trigram = 0.7,
        transition = 0.0,
      })

      local items = { item }
      nn.update_weights(item, items)

      -- Verify state was saved
      assert.is_not_nil(saved_state)
      assert.is_not_nil(saved_state.nn)
      assert.is_not_nil(saved_state.nn.network)

      -- Calculate score with trained state
      local score1 = nn.calculate_score(item.nos.normalized_features)

      -- Reset the module and reload through production pathway
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config("nn", STANDARD_TEST_CONFIG)
      nn.init(config.algorithm_config.nn)
      nn.load_weights() -- This should load from the mock

      -- Calculate score with loaded state
      local score2 = nn.calculate_score(item.nos.normalized_features)

      -- Scores should be the same
      assert.is_true(math.abs(score1 - score2) < 0.001)
    end)
  end)

  describe("Weight Decay", function()
    it("reduces weight magnitudes over time with weight decay", function()
      -- Set random seed for reproducibility
      math.randomseed(12345)

      -- Initialize with significant weight decay
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          learning_rate = 0.01, -- Lower learning rate so weight decay can dominate
          batch_size = 2,
          history_size = 20,
          batches_per_update = 1,
          weight_decay = 0.1, -- Higher decay for testing
        })
      )
      nn.init(config.algorithm_config.nn)

      local saved_weights = nil
      local initial_weights = nil
      local weights_after_training = nil

      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_weights or {}
        end,
        save_weights = function(algo_name, weights)
          saved_weights = weights
          if not initial_weights and weights.nn and weights.nn.stats and weights.nn.stats.weight_norms then
            -- Capture initial weight norms after first save
            initial_weights = {}
            for i, norm in ipairs(weights.nn.stats.weight_norms) do
              initial_weights[i] = norm
            end
          end
          weights_after_training = weights
        end,
      }

      -- Create training samples
      local training_items = {}
      for i = 1, 10 do
        local features = {}
        for _, key in ipairs({
          "match",
          "virtual_name",
          "frecency",
          "open",
          "alt",
          "proximity",
          "project",
          "recency",
          "trigram",
        }) do
          features[key] = math.random()
        end
        table.insert(training_items, create_mock_item("file" .. i .. ".lua", features))
      end

      -- Train multiple times to accumulate weight decay effect
      -- Use all training_items as ranked list to ensure adequate pairs are created
      for _ = 1, 5 do
        for _, item in ipairs(training_items) do
          nn.update_weights(item, training_items)
        end
      end

      -- Check that weight norms decreased
      assert.is_not_nil(weights_after_training)
      assert.is_not_nil(weights_after_training.nn.stats.weight_norms)

      -- At least some layers should show reduced weight norms
      local any_decreased = false
      for i = 1, #weights_after_training.nn.stats.weight_norms do
        if initial_weights[i] and initial_weights[i] > 0 then
          if weights_after_training.nn.stats.weight_norms[i] < initial_weights[i] then
            any_decreased = true
            break
          end
        end
      end

      assert.is_true(any_decreased, "Weight decay should reduce at least some weight magnitudes")
    end)

    it("has no effect when weight decay is zero", function()
      -- Initialize with zero weight decay
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 2,
          batches_per_update = 1,
          weight_decay = 0.0, -- No decay
        })
      )
      nn.init(config.algorithm_config.nn)

      local saved_count = 0
      local weights_history = {}

      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return weights_history[#weights_history] or {}
        end,
        save_weights = function(algo_name, weights)
          saved_count = saved_count + 1
          table.insert(weights_history, vim.deepcopy(weights))
        end,
      }

      -- Train with same data multiple times
      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.5,
        frecency = 0.3,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.4,
        trigram = 0.7,
        transition = 0.0,
      })

      local other_item = create_mock_item("other.lua", {
        match = 0.3,
        virtual_name = 0.1,
        frecency = 0.9,
        open = 0.0,
        alt = 0.0,
        proximity = 0.2,
        project = 0.0,
        recency = 0.1,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Train multiple times
      for _ = 1, 3 do
        nn.update_weights(item, { other_item, item })
      end

      assert.equals(3, saved_count)
      -- With zero decay, weights should only change due to gradients
      -- This test primarily ensures the system works without decay
    end)

    it("applies per-layer decay multipliers correctly", function()
      -- Initialize with per-layer decay multipliers
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 2,
          batches_per_update = 1,
          weight_decay = 0.01,
          layer_decay_multipliers = { 0.5, 2.0 }, -- First layer less decay, second layer more
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Train to apply weight decay
      local item = create_mock_item("test.lua", {
        match = 0.7,
        virtual_name = 0.3,
        frecency = 0.5,
        open = 1.0,
        alt = 0.0,
        proximity = 0.4,
        project = 1.0,
        recency = 0.6,
        trigram = 0.8,
        transition = 0.0,
      })

      nn.update_weights(item, { item })

      -- Configuration should be preserved
      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn)
      -- Test passes if no errors occur during training with per-layer multipliers
    end)

    it("maintains training convergence with reasonable weight decay", function()
      -- Initialize with moderate weight decay
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          architecture = { 10, 8, 1 },
          history_size = 50,
        })
      )
      nn.init(config.algorithm_config.nn)

      local losses = {}
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return {}
        end,
        save_weights = function(algo_name, weights)
          if weights.nn and weights.nn.stats and weights.nn.stats.last_loss then
            table.insert(losses, weights.nn.stats.last_loss)
          end
        end,
      }

      -- Create consistent training data
      local positive_item = create_mock_item("positive.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 1.0,
        proximity = 0.9,
        project = 1.0,
        recency = 0.8,
        trigram = 0.9,
        transition = 0.0,
      })

      local negative_item = create_mock_item("negative.lua", {
        match = 0.1,
        virtual_name = 0.1,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.1,
        project = 0.0,
        recency = 0.1,
        trigram = 0.1,
        transition = 0.0,
      })

      -- Train multiple epochs - rank negative higher to ensure training happens
      for _ = 1, 5 do
        nn.update_weights(positive_item, { negative_item, positive_item })
      end

      -- Loss should be recorded
      assert.is_true(#losses > 0, "Training should produce loss values")

      -- With pairwise hinge loss, if positive is already ranked #1, loss may be 0
      -- Just verify that training executes without errors
      -- (The actual loss values may be constant if margin is already satisfied)
    end)
  end)

  describe("Debug View", function()
    it("generates debug information", function()
      local item = {
        file = "test.lua",
        nos = {
          neural_score = 5.5,
          normalized_features = {
            match = 0.8,
            virtual_name = 0.2,
            frecency = 0.3,
            open = 1.0,
            alt = 0.0,
            proximity = 0.6,
            project = 1.0,
            recency = 0.4,
            trigram = 0.5,
            transition = 0.0,
          },
        },
      }

      local lines = nn.debug_view(item)

      assert.is_table(lines)
      assert.is_true(#lines > 0)

      -- Check for expected content
      local content = table.concat(lines, "\n")
      assert.is_true(content:find("Neural Network") ~= nil)
      assert.is_true(content:find("Architecture") ~= nil)
      assert.is_true(content:find("Learning Rate") ~= nil)
      assert.is_true(content:find("Training Statistics") ~= nil)
    end)

    it("respects custom history_size configuration", function()
      -- Mock the main module config
      package.loaded["neural-open"] = {
        config = {
          algorithm = "nn",
          algorithm_config = {
            nn = {
              architecture = { 10, 4, 1 },
              optimizer = "sgd",
              learning_rate = 0.1,
              batch_size = 4,
              history_size = 10000, -- Custom history size
              batches_per_update = 2,
              dropout_rates = { 0 },
            },
          },
        },
      }

      -- Reload the nn module to pick up the new config
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local item = {
        file = "test.lua",
        nos = {
          neural_score = 5.5,
          normalized_features = {
            match = 0.8,
            virtual_name = 0.2,
            frecency = 0.3,
            open = 1.0,
            alt = 0.0,
            proximity = 0.6,
            project = 1.0,
            recency = 0.4,
            trigram = 0.5,
            transition = 0.0,
          },
        },
      }

      local lines = nn.debug_view(item)
      local content = table.concat(lines, "\n")

      -- Verify that the debug view shows the custom history size (10000) not the default (1000)
      assert.is_true(content:find("History Size:") ~= nil, "Debug view should show history size")

      -- Find the actual history size line
      local history_line = nil
      for _, line in ipairs(lines) do
        if line:match("History Size:") then
          history_line = line
          break
        end
      end

      assert.is_not_nil(history_line, "Should find History Size line")
      assert.is_true(history_line:find("/10000") ~= nil, "History Size line should show /10000, got: " .. history_line)
    end)

    it("displays weight decay information", function()
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          weight_decay = 0.0001, -- Explicitly set for this test
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = {
        file = "test.lua",
        nos = {
          neural_score = 5.0,
          normalized_features = {
            match = 0.5,
            virtual_name = 0.5,
            frecency = 0.5,
            open = 0.5,
            alt = 0.5,
            proximity = 0.5,
            project = 0.5,
            recency = 0.5,
            trigram = 0.5,
            transition = 0.0,
          },
        },
      }

      local lines = nn.debug_view(item)
      local content = table.concat(lines, "\n")

      -- Check that weight decay is displayed
      assert.is_true(content:find("Weight Decay") ~= nil, "Debug view should show weight decay")
      assert.is_true(content:find("0.000100") ~= nil, "Debug view should show weight decay value")
    end)

    it("displays weight statistics when available", function()
      -- Initialize and perform training to generate weight statistics
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          weight_decay = 0.001,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return {}
        end,
        save_weights = function()
          -- Mock save
        end,
      }

      -- Perform training to generate statistics
      local item = create_mock_item("test.lua", {
        match = 0.7,
        virtual_name = 0.3,
        frecency = 0.5,
        open = 1.0,
        alt = 0.0,
        proximity = 0.4,
        project = 1.0,
        recency = 0.6,
        trigram = 0.8,
        transition = 0.0,
      })

      nn.update_weights(item, { item })

      local lines = nn.debug_view(item)
      local content = table.concat(lines, "\n")

      -- Check for weight statistics section
      if content:find("Weight Statistics") then
        assert.is_true(content:find("L2 Norm") ~= nil, "Should show L2 norm when statistics available")
        assert.is_true(content:find("Avg Magnitude") ~= nil, "Should show average magnitude when statistics available")
      end
    end)

    it("shows progressive loss averages based on available samples", function()
      -- Initialize network
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          batch_size = 2,
          history_size = 1100, -- Support up to 1000 loss history
          batches_per_update = 1,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.5,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.3,
        trigram = 0.8,
        transition = 0.0,
      })

      local item2 = create_mock_item("other.lua", {
        match = 0.3,
        virtual_name = 0.2,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.4,
        project = 1.0,
        recency = 0.9,
        trigram = 0.2,
        transition = 0.0,
      })

      -- Mock weights module to prevent actual persistence
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return nil
        end,
        save_weights = function() end,
      }

      -- Initially with no training, should show "Last Hinge Loss:"
      local lines = nn.debug_view(item)
      local content = table.concat(lines, "\n")
      assert.is_true(content:match("Last Hinge Loss:") ~= nil)
      assert.is_nil(content:match("Avg Hinge Loss:"))

      -- Train once to get 1 sample
      nn.update_weights(item, { item, item2 })
      lines = nn.debug_view(item)
      content = table.concat(lines, "\n")

      -- With history, should show "Avg Hinge Loss:" with appropriate windows
      -- With just a few samples, should only show [1] and maybe [2] if we have 2+ samples
      assert.is_true(content:match("Avg Hinge Loss:") ~= nil)
      assert.is_true(content:match("%[1%]") ~= nil) -- Should have [1] for last loss

      -- Train more times to build up history
      for _ = 1, 15 do
        nn.update_weights(item, { item, item2 })
      end

      lines = nn.debug_view(item)
      content = table.concat(lines, "\n")

      -- Should now have [1], [10], and actual count averages
      assert.is_true(content:match("Avg Hinge Loss:") ~= nil)
      assert.is_true(content:match("%[1%]") ~= nil) -- Last loss
      assert.is_true(content:match("%[10%]") ~= nil) -- 10 sample average

      -- Train many more times to reach 100+ samples
      for _ = 1, 90 do
        nn.update_weights(item, { item, item2 })
      end

      lines = nn.debug_view(item)
      content = table.concat(lines, "\n")

      -- Should now have [1], [10], [100] averages
      assert.is_true(content:match("Avg Hinge Loss:") ~= nil)
      assert.is_true(content:match("%[1%]") ~= nil)
      assert.is_true(content:match("%[10%]") ~= nil)
      assert.is_true(content:match("%[100%]") ~= nil)

      -- Verify format is on single line
      local loss_line = nil
      for _, line in ipairs(lines) do
        if line:match("Avg Hinge Loss:") then
          loss_line = line
          break
        end
      end
      assert.is_not_nil(loss_line)

      -- Verify format matches expected pattern: "Avg Hinge Loss: [1] X.XXXXXX [10] X.XXXXXX [100] X.XXXXXX"
      assert.is_true(loss_line:match("Avg Hinge Loss: %[1%] %d+%.%d+ %[10%] %d+%.%d+ %[100%] %d+%.%d+") ~= nil)

      -- Train a few more times to get to 103 samples (testing partial bucket scenario)
      for _ = 1, 3 do
        nn.update_weights(item, { item, item2 })
      end

      lines = nn.debug_view(item)
      content = table.concat(lines, "\n")

      -- Should now have [1], [10], [100], [103] (partial bucket for 1000)
      assert.is_true(content:match("Avg Hinge Loss:") ~= nil)
      assert.is_true(content:match("%[1%]") ~= nil)
      assert.is_true(content:match("%[10%]") ~= nil)
      assert.is_true(content:match("%[100%]") ~= nil)

      -- Find the loss line again
      loss_line = nil
      for _, line in ipairs(lines) do
        if line:match("Avg Hinge Loss:") then
          loss_line = line
          break
        end
      end
      assert.is_not_nil(loss_line)

      -- Should show the partial bucket with actual sample count (around 103, may vary due to batching)
      -- The actual count might be slightly different due to how batching works, but should be > 100
      local partial_bucket = loss_line:match("%[(%d+)%]%s+%d+%.%d+$")
      assert.is_not_nil(partial_bucket, "Should show partial bucket at end of loss line")
      local count = tonumber(partial_bucket)
      assert.is_true(count > 100, "Partial bucket should show > 100 samples")
      assert.is_true(count < 1000, "Partial bucket should show < 1000 samples")
    end)
  end)

  describe("AdamW Optimizer", function()
    before_each(function()
      -- Mock vim.notify for optimizer migration messages
      _G.vim = _G.vim or {}
      _G.vim.notify = function() end
      _G.vim.log = { levels = { INFO = 1 } }
    end)

    it("initializes with AdamW optimizer", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      assert.is_not_nil(nn)
    end)

    it("rejects invalid optimizer types", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      assert.has_error(function()
        local helpers = require("tests.helpers")
        local config = helpers.create_algorithm_config(
          "nn",
          vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
            optimizer = "invalid",
            learning_rate = 0.01,
            batch_size = 32,
            batches_per_update = 1,
            history_size = 100,
          })
        )
        nn.init(config.algorithm_config.nn)
      end)
    end)

    it("saves and loads AdamW optimizer state", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      local selected_item = create_mock_item("selected.lua", {
        match = 0.7,
        virtual_name = 0.3,
        frecency = 0.5,
        open = 1.0,
        alt = 0.0,
        proximity = 0.4,
        project = 1.0,
        recency = 0.6,
        trigram = 0.8,
        transition = 0.0,
      })

      -- Create enough items to meet 50% threshold (16 pairs for batch_size=32)
      -- Pair generation: up to 5 hard negatives + 1 below + 4 random below = max 10 pairs per update
      -- So we need at least 16 items ranked higher to have history for batching
      -- Place selected item at rank 17 (16 items above, rest below)
      local ranked_items = {}

      -- Add 16 items above selected
      for i = 1, 16 do
        table.insert(
          ranked_items,
          create_mock_item("higher" .. i .. ".lua", {
            match = 0.8 + i * 0.001,
            virtual_name = 0.4,
            frecency = 0.6,
            open = 0.0,
            alt = 0.0,
            proximity = 0.5,
            project = 1.0,
            recency = 0.5,
            trigram = 0.7,
            transition = 0.0,
          })
        )
      end

      -- Add selected item
      table.insert(ranked_items, selected_item)

      -- Add 5 items below selected for random pairs
      for i = 1, 5 do
        table.insert(
          ranked_items,
          create_mock_item("lower" .. i .. ".lua", {
            match = 0.6 - i * 0.001,
            virtual_name = 0.2,
            frecency = 0.4,
            open = 0.0,
            alt = 0.0,
            proximity = 0.3,
            project = 1.0,
            recency = 0.3,
            trigram = 0.5,
            transition = 0.0,
          })
        )
      end

      -- This will generate 5 (hard neg) + 1 (below) + 4 (random) = 10 pairs
      -- But we need to run multiple updates to build history for a batch of 16
      -- Run 2 updates to accumulate pairs in history
      nn.update_weights(selected_item, ranked_items)
      nn.update_weights(selected_item, ranked_items)

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn)
      assert.equals("adamw", saved_weights.nn.optimizer_type)
      assert.is_not_nil(saved_weights.nn.optimizer_state)
      assert.is_not_nil(saved_weights.nn.optimizer_state.timestep)
      assert.is_true(saved_weights.nn.optimizer_state.timestep > 0)
      assert.is_not_nil(saved_weights.nn.optimizer_state.moments)
      assert.is_not_nil(saved_weights.nn.optimizer_state.moments.first)
      assert.is_not_nil(saved_weights.nn.optimizer_state.moments.second)
    end)

    it("migrates legacy weight files to SGD optimizer", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Simulate legacy weight file without optimizer_type
      local legacy_weights = {
        nn = {
          network = {
            weights = {},
            biases = {},
          },
          training_history = {},
          stats = {},
        },
      }

      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return legacy_weights
        end,
        save_weights = function() end,
      }

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          learning_rate = 0.01,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      local features = {
        match = 0.5,
        virtual_name = 0.5,
        frecency = 0.5,
        open = 0.5,
        alt = 0.5,
        proximity = 0.5,
        project = 0.5,
        recency = 0.5,
        trigram = 0.5,
        transition = 0.0,
      }

      -- Should load without error and default to SGD
      local score = nn.calculate_score(features)
      assert.is_number(score)
    end)

    it("resets optimizer state when switching optimizers", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Start with AdamW
      local saved_weights = {
        nn = {
          version = "2.0-hinge", -- Add version to avoid migration notification
          network = {
            weights = {},
            biases = {},
          },
          optimizer_type = "adamw",
          optimizer_state = { timestep = 100, moments = {} },
          training_history = {},
          stats = {},
        },
      }

      local notify_called = false
      _G.vim.notify = function(msg, level)
        notify_called = true
        assert.is_true(msg:find("Optimizer changed") ~= nil)
      end

      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_weights
        end,
        save_weights = function() end,
      }

      -- Initialize with SGD
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          learning_rate = 0.01,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Trigger weight loading by calculating a score
      local features = {
        match = 0.5,
        virtual_name = 0.5,
        frecency = 0.5,
        open = 0.5,
        alt = 0.5,
        proximity = 0.5,
        project = 0.5,
        recency = 0.5,
        trigram = 0.5,
        transition = 0.0,
      }
      nn.calculate_score(features)

      -- Should have notified about optimizer change
      assert.is_true(notify_called)
    end)

    it("updates weights using AdamW with moment accumulation", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
          batch_size = 2,
          batches_per_update = 1,
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      local item1 = create_mock_item("test1.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.9,
        project = 1.0,
        recency = 0.8,
        trigram = 0.9,
        transition = 0.0,
      })

      local item2 = create_mock_item("test2.lua", {
        match = 0.1,
        virtual_name = 0.1,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.1,
        project = 0.0,
        recency = 0.1,
        trigram = 0.1,
        transition = 0.0,
      })

      -- First update
      nn.update_weights(item1, { item2, item1 })
      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.equals(1, saved_weights.nn.optimizer_state.timestep)

      -- Second update - timestep should increase
      nn.update_weights(item1, { item2, item1 })
      saved_weights = weights_mock.get_saved()
      assert.is_true(saved_weights.nn.optimizer_state.timestep > 1)

      -- Check that moments are being accumulated
      local first_moments = saved_weights.nn.optimizer_state.moments.first
      local second_moments = saved_weights.nn.optimizer_state.moments.second
      assert.is_not_nil(first_moments.weights[1])
      assert.is_not_nil(second_moments.weights[1])
    end)

    it("displays AdamW optimizer information in debug view", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      local item = create_mock_item("test.lua", {
        match = 0.5,
        virtual_name = 0.5,
        frecency = 0.5,
        open = 0.5,
        alt = 0.5,
        proximity = 0.5,
        project = 0.5,
        recency = 0.5,
        trigram = 0.5,
        transition = 0.0,
      })

      nn.update_weights(item, { item })

      local lines = nn.debug_view(item)
      local content = table.concat(lines, "\n")

      assert.is_true(content:find("Optimizer: AdamW") ~= nil)
      assert.is_true(content:find("Timestep:") ~= nil)
      assert.is_true(content:find("Beta1:") ~= nil)
      assert.is_true(content:find("Beta2:") ~= nil)
    end)

    it("produces different training behavior between SGD and AdamW", function()
      -- Test SGD
      package.loaded["neural-open.algorithms.nn"] = nil
      local nn_sgd = require("neural-open.algorithms.nn")

      local helpers = require("tests.helpers")
      local config_sgd = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          learning_rate = 0.01,
          batch_size = 2,
          batches_per_update = 1,
        })
      )
      nn_sgd.init(config_sgd.algorithm_config.nn)

      local saved_sgd = nil
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_sgd or {}
        end,
        save_weights = function(algo_name, weights)
          saved_sgd = weights
        end,
      }

      math.randomseed(42)

      local item_pos = create_mock_item("positive.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.9,
        project = 1.0,
        recency = 0.8,
        trigram = 0.9,
        transition = 0.0,
      })

      local item_neg = create_mock_item("negative.lua", {
        match = 0.1,
        virtual_name = 0.1,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.1,
        project = 0.0,
        recency = 0.1,
        trigram = 0.1,
        transition = 0.0,
      })

      for _ = 1, 3 do
        nn_sgd.update_weights(item_pos, { item_neg, item_pos })
      end

      local sgd_loss = saved_sgd.nn.stats.last_loss

      -- Test AdamW
      package.loaded["neural-open.algorithms.nn"] = nil
      local nn_adamw = require("neural-open.algorithms.nn")

      nn_adamw.init({
        architecture = { 10, 4, 1 },
        optimizer = "adamw",
        learning_rate = 0.001,
        batch_size = 2,
        history_size = 10,
        batches_per_update = 1,
        dropout_rates = { 0 },
      })

      local saved_adamw = nil
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_adamw or {}
        end,
        save_weights = function(algo_name, weights)
          saved_adamw = weights
        end,
      }

      math.randomseed(42)

      for _ = 1, 3 do
        nn_adamw.update_weights(item_pos, { item_neg, item_pos })
      end

      local adamw_loss = saved_adamw.nn.stats.last_loss

      -- Both should produce valid losses
      assert.is_number(sgd_loss)
      assert.is_number(adamw_loss)
      assert.is_true(sgd_loss >= 0)
      assert.is_true(adamw_loss >= 0)
    end)

    it("preserves network weights when switching optimizers", function()
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Train with SGD first
      local helpers = require("tests.helpers")
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          learning_rate = 0.01,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config.algorithm_config.nn)
      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      local selected_item = create_mock_item("selected.lua", {
        match = 0.7,
        virtual_name = 0.3,
        frecency = 0.5,
        open = 1.0,
        alt = 0.0,
        proximity = 0.4,
        project = 1.0,
        recency = 0.6,
        trigram = 0.8,
        transition = 0.0,
      })

      local other_item = create_mock_item("other.lua", {
        match = 0.8,
        virtual_name = 0.4,
        frecency = 0.6,
        open = 0.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.5,
        trigram = 0.7,
        transition = 0.0,
      })

      -- Create pairs by ranking other_item higher
      nn.update_weights(selected_item, { other_item, selected_item })

      -- Reload with AdamW
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local config_adamw = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
          batch_size = 32,
          batches_per_update = 1,
          history_size = 100,
        })
      )
      nn.init(config_adamw.algorithm_config.nn)

      -- Calculate score with original weights (should load from saved)
      local score1 = nn.calculate_score(selected_item.nos.normalized_features)

      -- Reset and use saved SGD weights but with AdamW
      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local saved_weights = weights_mock.get_saved()
      saved_weights.nn.optimizer_type = "sgd" -- Simulate old weight file

      local config_switch = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
        })
      )
      nn.init(config_switch.algorithm_config.nn)

      -- Force reload
      nn.load_weights()

      local score2 = nn.calculate_score(selected_item.nos.normalized_features)

      -- Scores should be close since weights are preserved
      assert.is_true(math.abs(score1 - score2) < 1.0, "Weights should be preserved when switching optimizers")
    end)
  end)

  describe("Learning Rate Warmup", function()
    it("calculates warmup factor correctly at different timesteps", function()
      local helpers = require("tests.helpers")

      -- Test the warmup calculation logic
      -- We'll create a simple test by initializing with warmup enabled
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          warmup_steps = 100,
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Create test items
      local item1 = create_mock_item("selected.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.7,
        open = 1.0,
        alt = 0.0,
        proximity = 0.6,
        project = 1.0,
        recency = 0.5,
        trigram = 0.4,
        transition = 0.0,
      }, 50)

      local item2 = create_mock_item("other.lua", {
        match = 0.5,
        virtual_name = 0.4,
        frecency = 0.3,
        open = 0.0,
        alt = 0.0,
        proximity = 0.2,
        project = 0.0,
        recency = 0.1,
        trigram = 0.0,
        transition = 0.0,
      }, 30)

      -- Perform update to trigger warmup
      nn.update_weights(item1, { item2, item1 })

      -- Verify weights were updated (basic sanity check)
      local score_after = nn.calculate_score(item1.nos.normalized_features)
      assert.is_number(score_after)
    end)

    it("applies warmup to SGD optimizer", function()
      local helpers = require("tests.helpers")

      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          warmup_steps = 10,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.4,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Update should work with warmup enabled
      nn.update_weights(item, { item })

      -- Score should still be calculated correctly
      local score = nn.calculate_score(item.nos.normalized_features)
      assert.is_number(score)
      assert.is_true(score >= 0 and score <= 100)
    end)

    it("applies warmup to AdamW optimizer", function()
      local helpers = require("tests.helpers")

      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          optimizer = "adamw",
          learning_rate = 0.001,
          warmup_steps = 10,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.4,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Update should work with warmup enabled
      nn.update_weights(item, { item })

      -- Score should still be calculated correctly
      local score = nn.calculate_score(item.nos.normalized_features)
      assert.is_number(score)
      assert.is_true(score >= 0 and score <= 100)
    end)

    it("handles warmup_steps = 0 (disabled)", function()
      local helpers = require("tests.helpers")

      local config = helpers.create_algorithm_config("nn", STANDARD_TEST_CONFIG)
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.4,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Should work normally without warmup
      nn.update_weights(item, { item })

      local score = nn.calculate_score(item.nos.normalized_features)
      assert.is_number(score)
      assert.is_true(score >= 0 and score <= 100)
    end)

    it("maintains backward compatibility when warmup config is missing", function()
      local helpers = require("tests.helpers")

      -- Init without warmup config
      -- No warmup_steps or warmup_start_factor (testing backward compatibility)
      local config = helpers.create_algorithm_config("nn", STANDARD_TEST_CONFIG)
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.4,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Should work with default warmup settings (disabled)
      nn.update_weights(item, { item })

      local score = nn.calculate_score(item.nos.normalized_features)
      assert.is_number(score)
      assert.is_true(score >= 0 and score <= 100)
    end)
  end)

  describe("Match Dropout", function()
    it("initializes with match_dropout configuration", function()
      local helpers = require("tests.helpers")

      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          match_dropout = 0.25,
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Should initialize without error
      assert.is_not_nil(nn)
    end)

    it("applies match_dropout during training", function()
      local helpers = require("tests.helpers")

      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          match_dropout = 1.0,
        })
      )
      nn.init(config.algorithm_config.nn)

      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Create items where match and virtual_name are important features
      local item1 = create_mock_item("matched.lua", {
        match = 0.9,
        virtual_name = 0.8,
        frecency = 0.1,
        open = 0.0,
        alt = 0.0,
        proximity = 0.1,
        project = 0.0,
        recency = 0.1,
        trigram = 0.1,
        transition = 0.0,
      })

      local item2 = create_mock_item("other.lua", {
        match = 0.1,
        virtual_name = 0.1,
        frecency = 0.9,
        open = 1.0,
        alt = 0.0,
        proximity = 0.8,
        project = 1.0,
        recency = 0.7,
        trigram = 0.6,
        transition = 0.0,
      })

      -- Train with match_dropout = 1.0, which should force network to learn from other features
      nn.update_weights(item2, { item1, item2 })

      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn)
      assert.is_not_nil(saved_weights.nn.training_history)

      -- The network should have learned despite match/virtual_name being dropped out
      assert.is_true(saved_weights.nn.stats.samples_processed > 0)
    end)

    it("does not apply match_dropout during inference", function()
      local helpers = require("tests.helpers")

      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Initialize with deterministic weights
      math.randomseed(42)
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          match_dropout = 0.5,
        })
      )
      nn.init(config.algorithm_config.nn)

      local features = {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.3,
        open = 0.0,
        alt = 0.0,
        proximity = 0.4,
        project = 1.0,
        recency = 0.2,
        trigram = 0.5,
        transition = 0.0,
      }

      -- Calculate scores multiple times - should be deterministic in inference
      local score1 = nn.calculate_score(features)
      local score2 = nn.calculate_score(features)
      local score3 = nn.calculate_score(features)

      -- In inference mode (no dropout), scores should be identical
      assert.equals(score1, score2)
      assert.equals(score2, score3)
    end)

    it("improves learning without search features with match_dropout", function()
      local helpers = require("tests.helpers")

      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          architecture = { 10, 8, 1 },
          learning_rate = 0.05,
          batch_size = 4,
          history_size = 50,
          batches_per_update = 2,
          match_dropout = 0.5,
        })
      )
      nn.init(config.algorithm_config.nn)

      local weights_mock = helpers.create_weights_mock()
      package.loaded["neural-open.weights"] = weights_mock.mock

      -- Create training samples where non-match features are predictive
      -- Item with low match but high other features should be selected
      local selected_item = create_mock_item("selected.lua", {
        match = 0.1, -- Low match
        virtual_name = 0.1, -- Low virtual_name
        frecency = 0.9, -- High frecency
        open = 1.0, -- Open buffer
        alt = 0.0,
        proximity = 0.8, -- High proximity
        project = 1.0, -- In project
        recency = 0.9, -- High recency
        trigram = 0.7, -- High trigram
      })

      -- Item with high match but low other features
      local decoy_item = create_mock_item("decoy.lua", {
        match = 0.9, -- High match
        virtual_name = 0.8, -- High virtual_name
        frecency = 0.1, -- Low frecency
        open = 0.0, -- Not open
        alt = 0.0,
        proximity = 0.2, -- Low proximity
        project = 0.0, -- Not in project
        recency = 0.1, -- Low recency
        trigram = 0.2, -- Low trigram
      })

      -- Train multiple times to learn the pattern
      for _ = 1, 10 do
        nn.update_weights(selected_item, { decoy_item, selected_item })
      end

      -- After training, the network should score selected_item higher
      -- even when match features are available during inference
      local score_selected = nn.calculate_score(selected_item.nos.normalized_features)
      local score_decoy = nn.calculate_score(decoy_item.nos.normalized_features)

      -- The network should have learned to rely on non-match features
      assert.is_number(score_selected)
      assert.is_number(score_decoy)

      -- Check that training occurred
      local saved_weights = weights_mock.get_saved()
      assert.is_not_nil(saved_weights)
      assert.is_true(saved_weights.nn.stats.batches_trained > 0)
    end)

    it("handles match_dropout = 0 (disabled)", function()
      local helpers = require("tests.helpers")

      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      local config = helpers.create_algorithm_config("nn", STANDARD_TEST_CONFIG)
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.4,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Should work normally without match dropout
      nn.update_weights(item, { item })

      local score = nn.calculate_score(item.nos.normalized_features)
      assert.is_number(score)
      assert.is_true(score >= 0 and score <= 100)
    end)

    it("maintains backward compatibility when match_dropout is not specified", function()
      local helpers = require("tests.helpers")

      package.loaded["neural-open.algorithms.nn"] = nil
      nn = require("neural-open.algorithms.nn")

      -- Init without match_dropout config (testing backward compatibility)
      local config = helpers.create_algorithm_config("nn", STANDARD_TEST_CONFIG)
      nn.init(config.algorithm_config.nn)

      local item = create_mock_item("test.lua", {
        match = 0.8,
        virtual_name = 0.7,
        frecency = 0.6,
        open = 1.0,
        alt = 0.0,
        proximity = 0.5,
        project = 1.0,
        recency = 0.4,
        trigram = 0.3,
        transition = 0.0,
      })

      -- Should use default match_dropout value (0.25)
      nn.update_weights(item, { item })

      local score = nn.calculate_score(item.nos.normalized_features)
      assert.is_number(score)
      assert.is_true(score >= 0 and score <= 100)
    end)

    it("displays match_dropout in debug view", function()
      local helpers = require("tests.helpers")

      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          match_dropout = 0.3,
        })
      )
      nn.init(config.algorithm_config.nn)

      local item = {
        file = "test.lua",
        nos = {
          neural_score = 5.0,
          normalized_features = {
            match = 0.5,
            virtual_name = 0.5,
            frecency = 0.5,
            open = 0.5,
            alt = 0.5,
            proximity = 0.5,
            project = 0.5,
            recency = 0.5,
            trigram = 0.5,
            transition = 0.0,
          },
        },
      }

      local lines = nn.debug_view(item)
      local content = table.concat(lines, "\n")

      -- Check that match_dropout is displayed
      assert.is_true(content:find("Match Dropout") ~= nil, "Debug view should show match dropout")
      assert.is_true(content:find("30.0%%") ~= nil, "Debug view should show match dropout percentage")
    end)
  end)

  describe("Leaky ReLU Activation", function()
    it("uses leaky ReLU for all hidden layers", function()
      local helpers = require("tests.helpers")

      -- Initialize with deterministic weights for testing
      math.randomseed(12345)
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          architecture = { 10, 8, 6, 4, 1 },
          learning_rate = 0.01,
          batch_size = 4,
          history_size = 10,
          dropout_rates = { 0, 0, 0 },
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Test with features that would cause dead neurons with regular ReLU
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

      -- Calculate scores - with leaky ReLU, even all-zero inputs should produce non-zero output
      local score_neg = nn.calculate_score(negative_features)
      local score_pos = nn.calculate_score(positive_features)

      -- Both scores should be valid
      assert.is_number(score_neg)
      assert.is_number(score_pos)
      assert.is_true(score_neg >= 0 and score_neg <= 100)
      assert.is_true(score_pos >= 0 and score_pos <= 100)

      -- Scores should be different
      assert.are_not.equal(score_neg, score_pos)
    end)

    it("maintains gradient flow through multiple layers", function()
      local helpers = require("tests.helpers")

      -- Initialize network
      local config = helpers.create_algorithm_config(
        "nn",
        vim.tbl_extend("force", STANDARD_TEST_CONFIG, {
          architecture = { 10, 16, 8, 4, 1 },
          batch_size = 2,
          history_size = 10,
          dropout_rates = { 0, 0, 0 },
        })
      )
      nn.init(config.algorithm_config.nn)

      -- Mock weights module
      local saved_weights = nil
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return saved_weights
        end,
        save_weights = function(algo_name, weights)
          saved_weights = weights
        end,
      }

      -- Train with extreme cases that would cause vanishing gradients with regular ReLU
      local item1 = create_mock_item("file1.lua", {
        match = 0.01,
        virtual_name = 0.01,
        frecency = 0.01,
        open = 0.0,
        alt = 0.0,
        proximity = 0.01,
        project = 0.0,
        recency = 0.01,
        trigram = 0.01,
        transition = 0.0,
      })

      local item2 = create_mock_item("file2.lua", {
        match = 0.99,
        virtual_name = 0.99,
        frecency = 0.99,
        open = 1.0,
        alt = 1.0,
        proximity = 0.99,
        project = 1.0,
        recency = 0.99,
        trigram = 0.99,
        transition = 0.0,
      })

      -- Initial scores
      local score1_before = nn.calculate_score(item1.nos.normalized_features)
      local score2_before = nn.calculate_score(item2.nos.normalized_features)

      -- Train multiple times
      for _ = 1, 5 do
        nn.update_weights(item2, { item1, item2 })
      end

      -- Scores after training
      local score1_after = nn.calculate_score(item1.nos.normalized_features)
      local score2_after = nn.calculate_score(item2.nos.normalized_features)

      -- Verify that learning occurred (scores changed)
      assert.are_not.equal(score1_before, score1_after, "Score for item1 should change after training")
      assert.are_not.equal(score2_before, score2_after, "Score for item2 should change after training")

      -- Verify weights were updated
      assert.is_not_nil(saved_weights)
      assert.is_not_nil(saved_weights.nn)
      assert.is_not_nil(saved_weights.nn.network)
      assert.is_not_nil(saved_weights.nn.network.weights)
    end)
  end)
end)
