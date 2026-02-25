describe("scorer with transitions", function()
  local helpers = require("tests.helpers")
  local scorer

  before_each(function()
    helpers.setup()
    package.loaded["neural-open.scorer"] = nil
    scorer = require("neural-open.scorer")
  end)

  after_each(function()
    package.loaded["neural-open.scorer"] = nil
  end)

  describe("compute_static_raw_features", function()
    it("should initialize transition to 0 when no context transition_scores", function()
      local context = {
        cwd = "/test/project",
        current_file = "/test/project/current.lua",
        recent_files = {},
        -- No transition_scores
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "file.lua",
      }

      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)

      assert.equals(0, raw_features.transition)
    end)

    it("should set transition from context.transition_scores", function()
      local context = {
        cwd = "/test/project",
        current_file = "/test/project/current.lua",
        recent_files = {},
        transition_scores = {
          ["/test/project/file.lua"] = 0.5,
          ["/test/project/other.lua"] = 0.833,
        },
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "file.lua",
      }

      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)

      assert.equals(0.5, raw_features.transition)
    end)

    it("should default to 0 when path not in transition_scores", function()
      local context = {
        cwd = "/test/project",
        current_file = "/test/project/current.lua",
        recent_files = {},
        transition_scores = {
          ["/test/project/other.lua"] = 0.833,
        },
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "file.lua",
      }

      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)

      assert.equals(0, raw_features.transition)
    end)

    it("should handle empty transition_scores map", function()
      local context = {
        cwd = "/test/project",
        current_file = "/test/project/current.lua",
        recent_files = {},
        transition_scores = {},
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "file.lua",
      }

      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)

      assert.equals(0, raw_features.transition)
    end)

    it("should compute all features including transition", function()
      local context = {
        cwd = "/test/project",
        current_file = "/test/project/current.lua",
        recent_files = {},
        transition_scores = {
          ["/test/project/file.lua"] = 0.75,
        },
      }

      local item_data = {
        is_open_buffer = true,
        is_alternate = false,
        recent_rank = 2,
        virtual_name = "file.lua",
      }

      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)

      -- Verify all features are computed
      assert.equals(0, raw_features.match) -- Will be set in on_match_handler
      assert.equals(0, raw_features.virtual_name) -- Will be set in on_match_handler
      assert.equals(0, raw_features.frecency) -- Will be set in on_match_handler
      assert.equals(1, raw_features.open)
      assert.equals(0, raw_features.alt)
      assert.is_number(raw_features.proximity)
      assert.is_number(raw_features.project)
      assert.equals(2, raw_features.recency)
      assert.is_number(raw_features.trigram)
      assert.equals(0.75, raw_features.transition)
    end)
  end)

  describe("normalize_features", function()
    it("should pass through transition unchanged (already normalized)", function()
      local raw_features = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0.5,
      }

      local normalized = scorer.normalize_features(raw_features)

      assert.equals(0.5, normalized.transition)
    end)

    it("should normalize transition score of 0", function()
      local raw_features = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0,
      }

      local normalized = scorer.normalize_features(raw_features)

      assert.equals(0, normalized.transition)
    end)

    it("should normalize transition score of 0.833", function()
      local raw_features = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0.833,
      }

      local normalized = scorer.normalize_features(raw_features)

      assert.equals(0.833, normalized.transition)
    end)

    it("should normalize transition score of 1.0", function()
      local raw_features = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 1.0,
      }

      local normalized = scorer.normalize_features(raw_features)

      assert.equals(1.0, normalized.transition)
    end)

    it("should handle missing transition field", function()
      local raw_features = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        -- No transition field
      }

      local normalized = scorer.normalize_features(raw_features)

      assert.equals(0, normalized.transition)
    end)

    it("should normalize all features including transition", function()
      local raw_features = {
        match = 100,
        virtual_name = 50,
        frecency = 16,
        open = 1,
        alt = 0,
        proximity = 0.8,
        project = 1,
        recency = 3,
        trigram = 0.6,
        transition = 0.75,
      }

      local normalized = scorer.normalize_features(raw_features)

      -- All features should be normalized
      assert.is_number(normalized.match)
      assert.is_number(normalized.virtual_name)
      assert.is_number(normalized.frecency)
      assert.equals(1, normalized.open)
      assert.equals(0, normalized.alt)
      assert.equals(0.8, normalized.proximity)
      assert.equals(1, normalized.project)
      assert.is_number(normalized.recency)
      assert.equals(0.6, normalized.trigram)
      assert.equals(0.75, normalized.transition)
    end)
  end)

  describe("transition as static feature", function()
    it("should remain unchanged during on_match_handler calls", function()
      -- Create a context with transition scores
      local context = {
        cwd = "/test/project",
        current_file = "/test/project/current.lua",
        recent_files = {},
        transition_scores = {
          ["/test/project/file.lua"] = 0.5,
        },
        algorithm = {
          calculate_score = function(normalized_features)
            return normalized_features.transition or 0
          end,
        },
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "file.lua",
      }

      -- Compute static features (including transition)
      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)
      assert.equals(0.5, raw_features.transition)

      -- Create a mock item
      local item = {
        file = "/test/project/file.lua",
        text = "/test/project/file.lua",
        nos = {
          normalized_path = "/test/project/file.lua",
          raw_features = raw_features,
          normalized_features = {},
          ctx = context,
        },
      }

      -- Create a mock matcher
      local mock_matcher = {
        filter = { search = "test" },
        match = function(self, mock_item)
          return 100
        end,
      }

      -- Call on_match_handler
      scorer.on_match_handler(mock_matcher, item)

      -- Transition should still be 0.5
      assert.equals(0.5, item.nos.raw_features.transition)
      assert.equals(0.5, item.nos.normalized_features.transition)
    end)
  end)
end)
