local helpers = require("tests.helpers")

describe("trigram scoring integration", function()
  local scorer
  local neural_open
  local source

  before_each(function()
    helpers.clear_plugin_modules()

    neural_open = require("neural-open")
    scorer = require("neural-open.scorer")
    source = require("neural-open.source")

    -- Mock the main plugin to provide config after loading
    package.loaded["neural-open"].config = helpers.get_default_config()

    -- Mock recent module so source.capture_context does not rely on vim state
    package.loaded["neural-open.recent"] = {
      get_recency_map = function()
        return {}
      end,
    }

    -- Initialize with default config
    neural_open.setup({
      algorithm = "classic",
    })
  end)

  describe("context capture", function()
    it("should compute trigrams for current file", function()
      -- Mock current buffer
      vim.api.nvim_get_current_buf = function()
        return 1
      end
      vim.api.nvim_buf_is_valid = function(buf)
        return buf == 1
      end
      vim.api.nvim_buf_get_name = function(buf)
        if buf == 1 then
          return "/path/to/test_helper.js"
        end
        return ""
      end

      local ctx = { meta = {} }
      source.capture_context(ctx)

      assert.is_not_nil(ctx.meta.nos_ctx.current_file_trigrams)
      -- Check for some expected trigrams from "test_helper.js"
      assert.is_true(ctx.meta.nos_ctx.current_file_trigrams["tes"] or false)
      assert.is_true(ctx.meta.nos_ctx.current_file_trigrams["hel"] or false)
      assert.is_true(ctx.meta.nos_ctx.current_file_trigrams[".js"] or false)
    end)

    it("should handle no current file", function()
      vim.api.nvim_get_current_buf = function()
        return 1
      end
      vim.api.nvim_buf_is_valid = function()
        return true
      end
      vim.api.nvim_buf_get_name = function()
        return ""
      end

      local ctx = { meta = {} }
      source.capture_context(ctx)

      assert.is_nil(ctx.meta.nos_ctx.current_file_trigrams)
    end)
  end)

  describe("trigram feature computation", function()
    local context

    before_each(function()
      context = {
        current_file = "/path/to/user_controller.rb",
        current_file_trigrams = require("neural-open.trigrams").compute_trigrams("user_controller.rb"),
        cwd = "/path",
      }
    end)

    it("should compute trigram similarity for similar files", function()
      local normalized_path = "/path/to/user_service.rb"
      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "user_service.rb",
      }

      local raw_features = scorer.compute_static_raw_features(normalized_path, context, item_data)

      -- Should have non-zero trigram score due to shared "user" and ".rb"
      assert.is_not_nil(raw_features.trigram)
      assert.is_true(raw_features.trigram > 0.2)
      assert.is_true(raw_features.trigram < 0.8)

      -- Normalize and check it matches raw (already 0-1)
      local normalized_features = scorer.normalize_features(raw_features)
      assert.equals(raw_features.trigram, normalized_features.trigram)
    end)

    it("should compute low similarity for different files", function()
      local normalized_path = "/path/to/database.yml"
      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "database.yml",
      }

      local raw_features = scorer.compute_static_raw_features(normalized_path, context, item_data)

      assert.is_not_nil(raw_features.trigram)
      assert.is_true(raw_features.trigram < 0.3)
    end)

    it("should use virtual name for index files", function()
      local normalized_path = "/path/to/components/index.js"
      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "components/index.js",
      }

      context.current_file = "/path/to/helpers/index.js"
      context.current_file_trigrams = require("neural-open.trigrams").compute_trigrams("helpers/index.js")

      local raw_features = scorer.compute_static_raw_features(normalized_path, context, item_data)

      -- Should have moderate similarity due to shared "index.js"
      assert.is_not_nil(raw_features.trigram)
      assert.is_true(raw_features.trigram > 0.3)
      assert.is_true(raw_features.trigram < 0.7)
    end)

    it("should handle missing current file trigrams", function()
      local normalized_path = "/path/to/test.js"
      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "test.js",
      }

      context.current_file_trigrams = nil

      local raw_features = scorer.compute_static_raw_features(normalized_path, context, item_data)

      -- Should default to 0 when no current file trigrams
      assert.equals(0, raw_features.trigram)

      local normalized_features = scorer.normalize_features(raw_features)
      assert.equals(0, normalized_features.trigram)
    end)
  end)

  describe("trigram weight application", function()
    it("should apply trigram weight in classic algorithm", function()
      local weights_module = require("neural-open.weights")
      local registry = require("neural-open.algorithms.registry")
      local algorithm = registry.get_algorithm()

      -- Set specific weights for testing (isolate trigram and match)
      local test_weights = vim.deepcopy(helpers.get_default_config().algorithm_config.classic.default_weights)
      test_weights.open = 0
      test_weights.alt = 0
      test_weights.proximity = 0
      test_weights.project = 0
      test_weights.frecency = 0
      test_weights.recency = 0
      test_weights.trigram = 50
      weights_module.save_weights("classic", test_weights)

      local context = {
        current_file = "/path/to/test_helper.js",
        current_file_trigrams = require("neural-open.trigrams").compute_trigrams("test_helper.js"),
        cwd = "/path",
      }

      -- Compute features for first item
      local raw_features1 = scorer.compute_static_raw_features(
        "/path/to/test_helpers.js",
        context,
        { is_open_buffer = false, is_alternate = false, virtual_name = "test_helpers.js" }
      )
      local normalized_features1 = scorer.normalize_features(raw_features1)
      local score1 = algorithm.calculate_score(normalized_features1)

      -- Compute features for second item
      local raw_features2 = scorer.compute_static_raw_features(
        "/path/to/database.yml",
        context,
        { is_open_buffer = false, is_alternate = false, virtual_name = "database.yml" }
      )
      local normalized_features2 = scorer.normalize_features(raw_features2)
      local score2 = algorithm.calculate_score(normalized_features2)

      -- Item1 should have higher score due to trigram similarity
      assert.is_true(score1 > score2)

      -- Verify trigram contributed to the score
      local trigram_contribution1 = normalized_features1.trigram * 50
      local trigram_contribution2 = normalized_features2.trigram * 50

      assert.is_true(trigram_contribution1 > 0)
      assert.is_true(trigram_contribution1 > trigram_contribution2)
    end)
  end)
end)
