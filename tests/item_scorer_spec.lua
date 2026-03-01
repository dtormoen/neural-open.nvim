local helpers = require("tests.helpers")

describe("item_scorer module", function()
  local item_scorer

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    -- Provide config for scorer dependency
    package.loaded["neural-open"] = {
      config = helpers.get_default_config(),
    }

    item_scorer = require("neural-open.item_scorer")
  end)

  after_each(function()
    helpers.clear_plugin_modules()
  end)

  describe("ITEM_FEATURE_NAMES", function()
    it("has exactly 8 features", function()
      assert.equals(8, #item_scorer.ITEM_FEATURE_NAMES)
    end)

    it("has the correct feature ordering", function()
      local expected = {
        "match",
        "frecency",
        "cwd_frecency",
        "recency",
        "cwd_recency",
        "text_length_inv",
        "not_last_selected",
        "transition",
      }
      assert.same(expected, item_scorer.ITEM_FEATURE_NAMES)
    end)
  end)

  describe("input_buf_to_features", function()
    it("converts flat array to named features", function()
      local input_buf = { 0.5, 0.3, 0.2, 0.8, 0.6, 0.4, 1.0, 0.15 }
      local features = item_scorer.input_buf_to_features(input_buf)

      assert.equals(0.5, features.match)
      assert.equals(0.3, features.frecency)
      assert.equals(0.2, features.cwd_frecency)
      assert.equals(0.8, features.recency)
      assert.equals(0.6, features.cwd_recency)
      assert.equals(0.4, features.text_length_inv)
      assert.equals(1.0, features.not_last_selected)
      assert.equals(0.15, features.transition)
    end)

    it("roundtrips with ITEM_FEATURE_NAMES", function()
      local input_buf = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 }
      local features = item_scorer.input_buf_to_features(input_buf)

      -- Reconstruct from features
      local reconstructed = {}
      for i, name in ipairs(item_scorer.ITEM_FEATURE_NAMES) do
        reconstructed[i] = features[name]
      end
      assert.same(input_buf, reconstructed)
    end)
  end)

  describe("normalize_match_score", function()
    it("returns 0 for 0 or negative input", function()
      assert.equals(0, item_scorer.normalize_match_score(0))
      assert.equals(0, item_scorer.normalize_match_score(-5))
    end)

    it("returns 0.5 for raw score of 100", function()
      assert.are.near(0.5, item_scorer.normalize_match_score(100), 1e-10)
    end)

    it("uses same sigmoid as file scorer", function()
      local scorer = require("neural-open.scorer")
      for _, raw in ipairs({ 1, 50, 100, 150, 200, 300 }) do
        assert.are.near(
          scorer.normalize_match_score(raw),
          item_scorer.normalize_match_score(raw),
          1e-15,
          "mismatch at raw=" .. raw
        )
      end
    end)
  end)

  describe("normalize_item_frecency", function()
    it("returns 0 for 0 or negative input", function()
      assert.equals(0, item_scorer.normalize_item_frecency(0))
      assert.equals(0, item_scorer.normalize_item_frecency(-1))
    end)

    it("returns 0.5 for score of 8", function()
      assert.are.near(0.5, item_scorer.normalize_item_frecency(8), 1e-10)
    end)

    it("approaches 1 for large values", function()
      assert.is_true(item_scorer.normalize_item_frecency(1000) > 0.99)
    end)

    it("gives good spread for typical item tracking scores", function()
      -- After 1 selection: score ~1
      local s1 = item_scorer.normalize_item_frecency(1)
      -- After 3 selections: score ~3
      local s3 = item_scorer.normalize_item_frecency(3)
      -- After 10 selections: score ~10
      local s10 = item_scorer.normalize_item_frecency(10)

      assert.is_true(s1 > 0)
      assert.is_true(s3 > s1)
      assert.is_true(s10 > s3)
      assert.is_true(s10 < 1)
    end)
  end)

  describe("calculate_recency_score", function()
    it("returns 0 for nil or 0 rank", function()
      assert.equals(0, item_scorer.calculate_recency_score(nil, 100))
      assert.equals(0, item_scorer.calculate_recency_score(0, 100))
    end)

    it("returns 1.0 for rank 1 with max 100", function()
      assert.are.near(1.0, item_scorer.calculate_recency_score(1, 100), 1e-10)
    end)

    it("returns 0.51 for rank 50 with max 100", function()
      assert.are.near(0.51, item_scorer.calculate_recency_score(50, 100), 1e-10)
    end)

    it("returns 0.01 for rank 100 with max 100", function()
      assert.are.near(0.01, item_scorer.calculate_recency_score(100, 100), 1e-10)
    end)
  end)

  describe("normalize_text_length", function()
    it("returns 1.0 for zero-length text", function()
      assert.are.near(1.0, item_scorer.normalize_text_length(0), 1e-10)
    end)

    it("returns 0.5 for length 10", function()
      assert.are.near(0.5, item_scorer.normalize_text_length(10), 1e-10)
    end)

    it("decreases for longer text", function()
      local short = item_scorer.normalize_text_length(5)
      local long = item_scorer.normalize_text_length(20)
      assert.is_true(short > long)
    end)

    it("stays in [0,1] range", function()
      for _, len in ipairs({ 0, 1, 5, 10, 50, 100, 1000 }) do
        local val = item_scorer.normalize_text_length(len)
        assert.is_true(val >= 0 and val <= 1, "out of range for len=" .. len)
      end
    end)
  end)

  describe("on_match_handler", function()
    it("skips items without nos field", function()
      local item = { text = "test" }
      -- Should not error
      item_scorer.on_match_handler({}, item)
      assert.is_nil(item.score)
    end)

    it("skips items without algorithm in context", function()
      local item = { text = "test", nos = { ctx = {}, input_buf = { 0, 0, 0, 0, 0, 0, 0, 0 }, raw_features = {} } }
      item_scorer.on_match_handler({}, item)
      assert.is_nil(item.score)
    end)

    it("updates input_buf[1] and item.score when algorithm is present", function()
      local mock_algorithm = {
        calculate_score = function(_input_buf)
          return 42.5
        end,
      }

      local input_buf = { 0, 0.3, 0.2, 0.8, 0.6, 0.4, 1.0, 0.5 }
      local item = {
        text = "build",
        nos = {
          ctx = { algorithm = mock_algorithm },
          input_buf = input_buf,
          raw_features = { match = 0 },
          neural_score = 0,
        },
      }

      -- Mock matcher with empty query
      local mock_matcher = {}
      item_scorer.on_match_handler(mock_matcher, item)

      assert.equals(42.5, item.score)
      assert.equals(42.5, item.nos.neural_score)
      -- Match should be 0 (no query)
      assert.equals(0, input_buf[1])
      -- Static features unchanged
      assert.equals(0.3, input_buf[2])
    end)

    it("computes match score when query is present", function()
      local mock_algorithm = {
        calculate_score = function(input_buf)
          return input_buf[1] * 100
        end,
      }

      local input_buf = { 0, 0.3, 0.2, 0.8, 0.6, 0.4, 1.0, 0.5 }
      local item = {
        text = "build",
        nos = {
          ctx = { algorithm = mock_algorithm },
          input_buf = input_buf,
          raw_features = { match = 0 },
          neural_score = 0,
        },
      }

      local mock_matcher = {
        pattern = "bui",
        match = function(_, match_item)
          if match_item.text == "build" then
            return 150
          end
          return 0
        end,
      }

      item_scorer.on_match_handler(mock_matcher, item)

      -- Match should be normalized sigmoid of 150
      local expected_match = item_scorer.normalize_match_score(150)
      assert.are.near(expected_match, input_buf[1], 1e-10)
      assert.equals(150, item.nos.raw_features.match)
    end)
  end)
end)
