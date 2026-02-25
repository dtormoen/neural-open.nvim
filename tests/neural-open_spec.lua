-- local mock = require("luassert.mock")
-- local spy = require("luassert.spy")
local helpers = require("tests.helpers")

--- Convert a named normalized features table into a flat input_buf array
---@param normalized_features table<string, number>
---@return number[]
local function features_to_input_buf(normalized_features)
  local scorer = require("neural-open.scorer")
  local buf = {}
  for i, name in ipairs(scorer.FEATURE_NAMES) do
    buf[i] = normalized_features[name] or 0
  end
  return buf
end

describe("neural-open", function()
  local neural_open
  local scorer
  local weights
  local db

  before_each(function()
    helpers.clear_plugin_modules()

    neural_open = require("neural-open")
    scorer = require("neural-open.scorer")
    weights = require("neural-open.weights")
    db = require("neural-open.db")

    -- Let neural_open load its own config naturally (no mock needed)
  end)

  describe("setup", function()
    it("should initialize with default config", function()
      neural_open.setup()

      -- Get expected defaults for comparison
      local defaults = helpers.get_default_config()

      assert.is_not_nil(neural_open.config)
      assert.equals(
        defaults.algorithm_config.classic.default_weights.match,
        neural_open.config.algorithm_config.classic.default_weights.match
      )
      assert.equals(
        defaults.algorithm_config.classic.learning_rate,
        neural_open.config.algorithm_config.classic.learning_rate
      )
      assert.equals(defaults.special_files["index.js"], neural_open.config.special_files["index.js"])
    end)

    it("should merge custom config", function()
      local defaults = helpers.get_default_config()

      neural_open.setup({
        algorithm_config = {
          classic = {
            learning_rate = 0.8, -- Different from default
            default_weights = {
              match = 150, -- Different from default
            },
          },
        },
        debug = { preview = true }, -- Different from default
      })

      -- Verify custom values are applied
      assert.equals(0.8, neural_open.config.algorithm_config.classic.learning_rate)
      assert.equals(150, neural_open.config.algorithm_config.classic.default_weights.match)
      assert.equals(true, neural_open.config.debug.preview)

      -- Verify other defaults are preserved
      assert.equals(
        defaults.algorithm_config.classic.default_weights.virtual_name,
        neural_open.config.algorithm_config.classic.default_weights.virtual_name
      )
      assert.equals(
        defaults.algorithm_config.classic.default_weights.frecency,
        neural_open.config.algorithm_config.classic.default_weights.frecency
      )
    end)
  end)

  describe("scorer", function()
    describe("calculate_recency_score", function()
      it("should return 0 for nil or invalid rank", function()
        assert.equals(0, scorer.calculate_recency_score(nil, 100))
        assert.equals(0, scorer.calculate_recency_score(0, 100))
        assert.equals(0, scorer.calculate_recency_score(-1, 100))
      end)

      it("should return 0 for rank exceeding max_items", function()
        assert.equals(0, scorer.calculate_recency_score(101, 100))
        assert.equals(0, scorer.calculate_recency_score(200, 100))
      end)

      it("should calculate linear decay score based on recent rank", function()
        -- Formula is (max_items - recent_rank + 1) / max_items
        local max = 100
        assert.equals(1.0, scorer.calculate_recency_score(1, max)) -- (100-1+1)/100 = 1.0
        assert.equals(0.99, scorer.calculate_recency_score(2, max)) -- (100-2+1)/100 = 0.99
        assert.equals(0.98, scorer.calculate_recency_score(3, max)) -- (100-3+1)/100 = 0.98
        assert.equals(0.91, scorer.calculate_recency_score(10, max)) -- (100-10+1)/100 = 0.91
        assert.equals(0.51, scorer.calculate_recency_score(50, max)) -- (100-50+1)/100 = 0.51
        assert.equals(0.01, scorer.calculate_recency_score(100, max)) -- (100-100+1)/100 = 0.01
      end)

      it("should work with different max_items values", function()
        assert.equals(1.0, scorer.calculate_recency_score(1, 10)) -- (10-1+1)/10 = 1.0
        assert.equals(0.5, scorer.calculate_recency_score(6, 10)) -- (10-6+1)/10 = 0.5
        assert.equals(0.1, scorer.calculate_recency_score(10, 10)) -- (10-10+1)/10 = 0.1
        assert.equals(0, scorer.calculate_recency_score(11, 10)) -- exceeds max
      end)
    end)

    describe("compute_static_raw_features", function()
      before_each(function()
        neural_open.setup({})
      end)

      it("should calculate basic features", function()
        local normalized_path = "/path/to/file.lua"
        local context = {
          cwd = "/path",
          current_file = "/path/to/other.lua",
        }

        local raw_features = scorer.compute_static_raw_features(normalized_path, context, false, false, nil, "file.lua")

        assert.is_not_nil(raw_features)
        assert.equals(0, raw_features.open)
        assert.equals(0, raw_features.alt)
        assert.equals(1, raw_features.project) -- File is in project
        assert.is_true(raw_features.proximity > 0) -- Some proximity to current file
      end)

      it("should set buffer flags correctly", function()
        helpers.with_test_buffer("/path/to/buffer.lua", function(test_buf)
          local normalized_path = "/path/to/buffer.lua"
          local context = { cwd = "/path" }

          local raw_features =
            scorer.compute_static_raw_features(normalized_path, context, true, false, nil, "buffer.lua")

          assert.equals(1, raw_features.open)
          assert.equals(0, raw_features.alt)
        end)
      end)

      it("should calculate proximity correctly", function()
        local normalized_path = "/path/to/deep/nested/file.lua"
        local context = {
          cwd = "/path/to",
          current_file = "/path/to/current.lua",
        }

        local raw_features = scorer.compute_static_raw_features(normalized_path, context, false, false, nil, "file.lua")

        assert.is_not_nil(raw_features.proximity)
        assert.is_true(raw_features.proximity > 0)
        assert.is_true(raw_features.proximity < 1) -- Not in same directory
      end)

      it("should handle trigram similarity", function()
        local normalized_path = "/path/to/index.js"
        local tris = require("neural-open.trigrams")
        local current_file_trigrams, current_file_trigrams_size = tris.compute_trigrams("helper.js")
        local context = {
          cwd = "/path",
          current_file_trigrams = current_file_trigrams,
          current_file_trigrams_size = current_file_trigrams_size,
        }

        local raw_features =
          scorer.compute_static_raw_features(normalized_path, context, false, false, nil, "to/index.js")

        assert.is_not_nil(raw_features.trigram)
        -- Should have some similarity due to shared ".js" extension
        assert.is_true(raw_features.trigram > 0)
      end)
    end)

    describe("normalize_features", function()
      it("should normalize all features to [0,1] range", function()
        local raw_features = {
          match = 100,
          virtual_name = 50,
          frecency = 80,
          open = 1,
          alt = 0,
          proximity = 0.5,
          project = 1,
          recency = 5,
          trigram = 0.3,
          transition = 0.0,
        }

        local normalized = scorer.normalize_features(raw_features)

        -- Check all values are in [0,1] range
        for key, value in pairs(normalized) do
          assert.is_true(value >= 0 and value <= 1, string.format("%s=%f is not in [0,1]", key, value))
        end

        -- Check specific normalizations
        assert.is_true(normalized.match >= 0.5) -- 100 should map to 0.5 with our sigmoid
        assert.is_true(normalized.frecency > 0.9) -- 25 should map to high value (1 - 1/26)
        assert.equals(1, normalized.open)
        assert.equals(0, normalized.alt)
        assert.equals(0.5, normalized.proximity)
        assert.equals(1, normalized.project)
        assert.equals(0.96, normalized.recency) -- (100-5+1)/100 = 0.96
        assert.equals(0.3, normalized.trigram)
      end)
    end)

    describe("on_match_handler", function()
      it("should capture frecency from Snacks item", function()
        -- Set up the algorithm in context
        local registry = require("neural-open.algorithms.registry")
        local classic = registry.get_algorithm()

        local item = {
          file = "/test/file.lua",
          text = "/test/file.lua",
          frecency = 50, -- This simulates what Snacks.nvim sets during matching
          nos = {
            normalized_path = "/test/file.lua",
            raw_features = {},
            input_buf = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            ctx = {
              algorithm = classic,
              algorithm_name = "classic",
              algorithm_weights = {
                match = 100,
                virtual_name = 100,
                open = 100,
                alt = 100,
                proximity = 100,
                project = 100,
                frecency = 100,
                recency = 100,
                trigram = 100,
                transition = 0.0,
              },
            },
          },
        }

        -- Mock matcher with basic functionality
        local matcher = {
          match = function(_, mock_item)
            -- Return a score based on text matching
            if mock_item.text == "/test/file.lua" then
              return 100
            end
            return 50
          end,
          pattern = "test",
        }

        scorer.on_match_handler(matcher, item)

        -- Frecency should be captured and normalized
        assert.equals(50, item.nos.raw_features.frecency)
        -- Normalization: 1 - 1/(1+50/8) ≈ 0.862 (frecency is index 3 in input_buf)
        assert.near(0.862, item.nos.input_buf[3], 0.01)
      end)

      it("should handle missing frecency gracefully", function()
        -- Set up the algorithm in context
        local registry = require("neural-open.algorithms.registry")
        local classic = registry.get_algorithm()

        local item = {
          file = "/test/file.lua",
          text = "/test/file.lua",
          -- No frecency field - simulates item without frecency data
          nos = {
            normalized_path = "/test/file.lua",
            raw_features = {},
            input_buf = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            ctx = {
              algorithm = classic,
              algorithm_name = "classic",
              algorithm_weights = {
                match = 100,
                virtual_name = 100,
                open = 100,
                alt = 100,
                proximity = 100,
                project = 100,
                frecency = 100,
                recency = 100,
                trigram = 100,
                transition = 0.0,
              },
            },
          },
        }

        local matcher = {
          match = function(_, _)
            return 100
          end,
          pattern = "",
        }

        scorer.on_match_handler(matcher, item)

        -- Should remain 0 if no frecency is provided by Snacks
        assert.equals(0, item.nos.raw_features.frecency)
        -- Frecency is index 3 in input_buf
        assert.equals(0, item.nos.input_buf[3])
      end)
    end)
  end)

  describe("weight learning", function()
    local registry
    local classic_algorithm
    local mock_weights

    before_each(function()
      -- Create mock weights module for test isolation (like classic_spec.lua)
      local default_config = helpers.get_default_config()
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
        save_weights = function(algo_name, new_weights)
          mock_weights.weights[algo_name] = new_weights
        end,
        reset_weights = function(algo_name, new_weights)
          new_weights = new_weights or {}
          mock_weights.weights[algo_name] = vim.deepcopy(new_weights)
          return new_weights
        end,
      }

      -- Inject mock weights module
      package.loaded["neural-open.weights"] = mock_weights

      -- Clear algorithm modules so they pick up the mock
      package.loaded["neural-open.algorithms.classic"] = nil
      package.loaded["neural-open.algorithms.registry"] = nil
      package.loaded["neural-open"] = nil

      -- Update the outer scope's weights variable to point to our mock
      weights = mock_weights

      -- Now load fresh modules that will use the mock
      neural_open = require("neural-open")
      neural_open.setup({ algorithm = "classic" })
      registry = require("neural-open.algorithms.registry")
      classic_algorithm = registry.get_algorithm()
    end)

    after_each(function()
      package.loaded["neural-open.weights"] = nil
    end)

    it("should not update weights when item is rank 1", function()
      local selected = {
        file = "/selected",
        neural_rank = 1,
        nos = { input_buf = features_to_input_buf({ match = 1.0 }) },
      }
      local ranked = { selected }

      local weights_before = vim.deepcopy(weights.get_weights("classic"))
      classic_algorithm.update_weights(selected, ranked)
      local weights_after = weights.get_weights("classic")

      assert.are.same(weights_before, weights_after)
    end)

    it("should update weights based on feature differences", function()
      local selected = {
        file = "/selected",
        neural_rank = 3,
        nos = {
          input_buf = features_to_input_buf({
            match = 0.5,
            proximity = 0.8,
            open = 0,
            frecency = 0.3,
          }),
        },
      }

      local higher1 = {
        file = "/higher1",
        neural_rank = 1,
        nos = {
          input_buf = features_to_input_buf({
            match = 0.9,
            proximity = 0.2,
            open = 1.0,
            frecency = 0.1,
          }),
        },
      }

      local higher2 = {
        file = "/higher2",
        neural_rank = 2,
        nos = {
          input_buf = features_to_input_buf({
            match = 0.7,
            proximity = 0.3,
            open = 0,
            frecency = 0.2,
          }),
        },
      }

      local ranked = { higher1, higher2, selected }

      -- update_weights now saves internally
      classic_algorithm.update_weights(selected, ranked)

      -- Should increase proximity and frecency, decrease match and open
      local new_weights = weights.get_weights("classic")
      local default_weights = neural_open.config.algorithm_config.classic.default_weights

      -- Proximity was better in selected (0.8 vs 0.2, 0.3)
      assert.is_true(new_weights.proximity > default_weights.proximity)

      -- Open was worse in selected (0 vs 1.0, 0)
      assert.is_true(new_weights.open < default_weights.open)
    end)

    it("should apply learning rate correctly", function()
      weights.reset_weights("classic", {
        match = 100,
        proximity = 100,
        open = 100,
        frecency = 100,
      })

      -- Reload weights after reset
      classic_algorithm.load_weights()

      local selected = {
        file = "/selected",
        neural_rank = 2,
        nos = {
          input_buf = features_to_input_buf({
            match = 0,
            proximity = 1.0,
            open = 0,
            frecency = 0,
          }),
        },
      }

      local higher = {
        file = "/higher",
        neural_rank = 1,
        nos = {
          input_buf = features_to_input_buf({
            match = 1.0,
            proximity = 0,
            open = 1.0,
            frecency = 0,
          }),
        },
      }

      local ranked = { higher, selected }

      -- With learning rate 0.6:
      -- proximity: selected better (+1 * 0.6 = +0.6)
      -- match: higher better (-1 * 0.6 = -0.6)
      -- open: higher better (-1 * 0.6 = -0.6)
      -- update_weights now saves internally
      classic_algorithm.update_weights(selected, ranked)

      local new_weights = weights.get_weights("classic")
      assert.near(100.6, new_weights.proximity, 0.01)
      assert.near(99.4, new_weights.match, 0.01)
      assert.near(99.4, new_weights.open, 0.01)
    end)

    it("should clamp weights within bounds", function()
      weights.reset_weights("classic", {
        match = 199,
        proximity = 2,
        open = 100,
        frecency = 100,
      })

      -- Reload weights after reset
      classic_algorithm.load_weights()

      local selected_high = {
        file = "/selected_high",
        neural_rank = 2,
        nos = {
          input_buf = features_to_input_buf({
            match = 1.0,
            proximity = 0,
            open = 0,
            frecency = 0,
          }),
        },
      }

      local higher_high = {
        file = "/higher_high",
        neural_rank = 1,
        nos = {
          input_buf = features_to_input_buf({
            match = 0,
            proximity = 0,
            open = 0,
            frecency = 0,
          }),
        },
      }

      local selected_low = {
        file = "/selected_low",
        neural_rank = 2,
        nos = {
          input_buf = features_to_input_buf({
            match = 0,
            proximity = 0,
            open = 0,
            frecency = 0,
          }),
        },
      }

      local higher_low = {
        file = "/higher_low",
        neural_rank = 1,
        nos = {
          input_buf = features_to_input_buf({
            match = 0,
            proximity = 1.0,
            open = 0,
            frecency = 0,
          }),
        },
      }

      local ranked_high = { higher_high, selected_high }
      local ranked_low = { higher_low, selected_low }

      -- Set high learning rate to force weights to bounds
      local config = neural_open.config.algorithm_config.classic
      config.learning_rate = 10

      -- update_weights now saves internally
      classic_algorithm.update_weights(selected_high, ranked_high)
      local weights_high = weights.get_weights("classic")
      assert.equals(200, weights_high.match) -- Clamped to max

      -- Reload weights before second update
      classic_algorithm.load_weights()

      -- update_weights now saves internally
      classic_algorithm.update_weights(selected_low, ranked_low)
      local weights_low = weights.get_weights("classic")
      assert.equals(1, weights_low.proximity) -- Clamped to min
    end)
  end)

  describe("db", function()
    it("should save and load weights", function()
      helpers.with_temp_db(function(temp_db_path)
        neural_open.setup({ weights_path = temp_db_path })

        local test_weights = { classic = { match = 123, proximity = 456 } }
        db.save_weights(test_weights)

        local loaded = db.get_weights()
        assert.are.same(test_weights, loaded)
      end)
    end)
  end)

  describe("reset_weights", function()
    before_each(function()
      neural_open.setup({})
    end)

    it("should reset weights to defaults", function()
      weights.save_weights("classic", { match = 50, proximity = 25 })

      neural_open.reset_weights("classic")

      local current = weights.get_weights("classic")
      local defaults = neural_open.config.algorithm_config.classic.default_weights
      assert.are.same(defaults, current)
    end)
  end)
end)
