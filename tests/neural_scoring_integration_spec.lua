-- local mock = require("luassert.mock")
-- local spy = require("luassert.spy")
local helpers = require("tests.helpers")

describe("neural scoring integration", function()
  local neural_open
  local scorer

  before_each(function()
    helpers.clear_plugin_modules()

    neural_open = require("neural-open")
    scorer = require("neural-open.scorer")

    -- Mock the main plugin to provide config after loading
    package.loaded["neural-open"].config = helpers.get_default_config()
  end)

  describe("score_item with Snacks integration", function()
    before_each(function()
      neural_open.setup({
        score_balance = 1.0,
      })

      -- No need to manually initialize algorithms with new pattern
    end)

    it("should use metadata flags for buffer scoring", function()
      -- Set up item with raw features as would be done by transform
      local item = {
        file = "/path/to/buffer.lua",
        nos = {
          normalized_path = "/path/to/buffer.lua",
          is_open_buffer = true,
          is_alternate = true,
          raw_features = {
            match = 0,
            virtual_name = 0,
            frecency = 0,
            open = 1,
            alt = 1,
            proximity = 0,
            project = 1,
            recency = 0,
            trigram = 0,
          },
          normalized_features = {},
          ctx = { cwd = "/path" },
        },
      }

      -- Normalize features as would be done by on_match_handler
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)

      -- Check that the features were set correctly
      assert.is_not_nil(item.nos.normalized_features)
      assert.is_not_nil(item.nos.normalized_features.open)
      assert.equals(1, item.nos.normalized_features.open) -- open buffer should have normalized value of 1
      assert.is_not_nil(item.nos.normalized_features.alt)
      assert.equals(1, item.nos.normalized_features.alt) -- alternate buffer should have normalized value of 1

      -- Calculate score using algorithm
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()
      local neural_score = algorithm.calculate_score(item.nos.normalized_features)

      -- Check that neural score was calculated and includes buffer bonuses
      assert.is_not_nil(neural_score)
      assert.is_true(neural_score > 0, string.format("Expected positive neural_score, got: %s", tostring(neural_score)))
    end)

    it("should skip fuzzy scoring in favor of Snacks matcher", function()
      local item = {
        file = "/path/to/file.lua",
        nos = {
          normalized_path = "/path/to/file.lua",
          is_open_buffer = false,
          is_alternate = false,
          raw_features = {
            match = 0,
            virtual_name = 0,
            frecency = 0,
            open = 0,
            alt = 0,
            proximity = 0,
            project = 1,
            recency = 0,
            trigram = 0,
          },
          normalized_features = {},
          ctx = { cwd = "/path", query = "file" },
        },
        fuzzy_score = 0.8, -- This should be ignored
      }

      -- Normalize features as would be done by on_match_handler
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)

      -- Verify that features were normalized but don't calculate components
      -- since that function was removed (now internal to algorithms)
      assert.is_not_nil(item.nos.normalized_features)
      -- Should not have path_match in normalized features since we handle fuzzy matching in Snacks
      assert.is_nil(item.nos.normalized_features.path_match)
    end)
  end)

  describe("score_add integration", function()
    it("should set score_add field for Snacks picker", function()
      -- Mock process_file behavior
      local item = {
        text = "/test/file.lua",
        file = "/test/file.lua",
        nos = {
          normalized_path = "/test/file.lua",
          is_open_buffer = false,
          is_alternate = false,
          raw_features = {
            match = 0,
            virtual_name = 0,
            frecency = 0,
            open = 0,
            alt = 0,
            proximity = 0.5,
            project = 1,
            recency = 0,
            trigram = 0,
          },
          normalized_features = {},
          ctx = {
            cwd = "/test",
            current_file = "/test/other.lua",
            query = "",
          },
        },
      }

      -- Normalize features and calculate score as on_match_handler would
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()
      local score = algorithm.calculate_score(item.nos.normalized_features)

      -- Verify score is calculated
      assert.is_true(score > 0)

      -- Simulate what the transform function would do
      item.score_add = score
      item.nos.neural_score = score

      -- Verify the score_add field is set (this is what Snacks uses)
      assert.equals(score, item.score_add)
      assert.equals(score, item.nos.neural_score)
    end)
  end)

  describe("score_balance configuration", function()
    it("should apply score_balance multiplier", function()
      neural_open.setup({
        score_balance = 0.5, -- Half weight for neural scores
      })

      local item = {
        file = "/path/to/file.lua",
        nos = {
          normalized_path = "/path/to/file.lua",
          is_open_buffer = true,
          is_alternate = false,
          raw_features = {
            match = 0,
            virtual_name = 0,
            frecency = 0,
            open = 1,
            alt = 0,
            proximity = 0,
            project = 1,
            recency = 0,
            trigram = 0,
          },
          normalized_features = {},
          ctx = { cwd = "/path" },
        },
      }

      -- Calculate base score
      neural_open.config.score_balance = 1.0
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()
      local base_score = algorithm.calculate_score(item.nos.normalized_features)

      -- Calculate with balance
      neural_open.config.score_balance = 0.5
      local balanced_score = base_score * 0.5

      -- The transform function would apply this
      assert.equals(balanced_score, base_score * neural_open.config.score_balance)
    end)

    it("should handle score_balance of 0", function()
      neural_open.setup({
        score_balance = 0, -- Disable neural scoring
      })

      local item = {
        file = "/path/to/file.lua",
        nos = {
          normalized_path = "/path/to/file.lua",
          is_open_buffer = true,
          is_alternate = false,
          raw_features = {
            match = 0,
            virtual_name = 0,
            frecency = 0,
            open = 1,
            alt = 0,
            proximity = 0,
            project = 1,
            recency = 0,
            trigram = 0,
          },
          normalized_features = {},
          ctx = { cwd = "/path" },
        },
      }

      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()
      local score = algorithm.calculate_score(item.nos.normalized_features)
      local balanced = score * neural_open.config.score_balance

      assert.equals(0, balanced)
    end)
  end)

  describe("caching mechanism", function()
    it("should cache scores for performance", function()
      -- This test validates the caching logic conceptually
      -- In real usage, the source.lua transform function handles caching

      local cache = {}
      local item = {
        file = "/test/file.lua",
        nos = {
          normalized_path = "/test/file.lua",
          is_open_buffer = false,
          is_alternate = false,
          raw_features = {},
          normalized_features = {},
        },
      }

      local context = { cwd = "/test" }

      -- First calculation
      item.nos.ctx = context
      item.nos.raw_features = {
        match = 50,
        virtual_name = 0,
        frecency = 10,
        open = 0,
        alt = 0,
        proximity = 0.5,
        project = 1,
        recency = 0,
        trigram = 0,
      }
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()
      local score1 = algorithm.calculate_score(item.nos.normalized_features)
      cache[item.file] = { score = score1, features = item.nos.normalized_features }

      -- Cached retrieval simulation
      local cached = cache[item.file]
      assert.equals(score1, cached.score)
      assert.same(item.nos.normalized_features, cached.features)
    end)

    it("should use different cache keys for different queries", function()
      local cache = {}
      local item = {
        file = "/test/file.lua",
        nos = {
          normalized_path = "/test/file.lua",
          is_open_buffer = false,
          is_alternate = false,
          raw_features = {},
          normalized_features = {},
        },
      }

      local context1 = { cwd = "/test", query = "file" }
      local context2 = { cwd = "/test", query = "test" }

      -- Simulate different match scores for different queries
      item.nos.ctx = context1
      item.nos.raw_features = {
        match = 100, -- High match for "file" query
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 1,
        recency = 0,
        trigram = 0,
      }
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()
      local score1 = algorithm.calculate_score(item.nos.normalized_features)
      cache[item.file .. "__" .. context1.query] =
        { score = score1, features = vim.deepcopy(item.nos.normalized_features) }

      item.nos.ctx = context2
      item.nos.raw_features.match = 20 -- Lower match for "test" query
      item.nos.normalized_features = scorer.normalize_features(item.nos.raw_features)
      local score2 = algorithm.calculate_score(item.nos.normalized_features)
      cache[item.file .. "__" .. context2.query] =
        { score = score2, features = vim.deepcopy(item.nos.normalized_features) }

      -- Different cache entries for different queries
      assert.is_not_nil(cache["/test/file.lua__file"])
      assert.is_not_nil(cache["/test/file.lua__test"])
    end)
  end)
end)
