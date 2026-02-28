local helpers = require("tests.helpers")

describe("item_source module", function()
  local item_source
  local item_scorer
  local mock_db
  local original_new_timer
  local original_os_time
  local mock_time

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    mock_time = 1000000000
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
    end

    -- Mock vim.loop.new_timer for item_tracking
    original_new_timer = vim.loop.new_timer
    vim.loop.new_timer = function()
      return {
        start = function() end,
        stop = function() end,
        close = function() end,
      }
    end

    -- Mock db module
    mock_db = {
      weights_data = {},
      get_weights = function(_picker_name, _latency_ctx)
        return vim.deepcopy(mock_db.weights_data)
      end,
      save_weights = function(_picker_name, data, _latency_ctx)
        mock_db.weights_data = vim.deepcopy(data)
      end,
    }
    package.loaded["neural-open.db"] = mock_db

    -- Mock weights module
    package.loaded["neural-open.weights"] = {
      get_weights = function(_algo, _picker_name)
        return {}
      end,
      save_weights = function() end,
      get_default_weights = function(_algo)
        return {}
      end,
    }

    -- Provide config
    local config = helpers.get_default_config()
    package.loaded["neural-open"] = { config = config }

    item_scorer = require("neural-open.item_scorer")
    item_source = require("neural-open.item_source")
  end)

  after_each(function()
    os.time = original_os_time -- luacheck: ignore 122
    vim.loop.new_timer = original_new_timer
    helpers.clear_plugin_modules()
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open.weights"] = nil
  end)

  describe("capture_context", function()
    it("stores nos_ctx in ctx.meta", function()
      local ctx = { meta = {} }
      local config = helpers.get_default_config()

      item_source.capture_context("test_picker", ctx, config)

      assert.is_not_nil(ctx.meta.nos_ctx)
      assert.equals("test_picker", ctx.meta.nos_ctx.picker_name)
      assert.is_not_nil(ctx.meta.nos_ctx.cwd)
      assert.is_not_nil(ctx.meta.nos_ctx.algorithm)
      assert.is_not_nil(ctx.meta.nos_ctx.tracking_data)
    end)

    it("loads tracking data for the picker", function()
      -- Seed tracking data
      local item_tracking = require("neural-open.item_tracking")
      item_tracking.record_selection("test_picker", "build", vim.fn.getcwd())

      local ctx = { meta = {} }
      local config = helpers.get_default_config()

      item_source.capture_context("test_picker", ctx, config)

      local tracking = ctx.meta.nos_ctx.tracking_data
      assert.is_true(tracking.frecency["build"] > 0)
      assert.equals("build", tracking.last_selected)
    end)

    it("populates transition_scores when last_cwd_selected exists", function()
      local item_tracking = require("neural-open.item_tracking")
      local cwd = vim.fn.getcwd()

      -- Record two selections to establish a transition: build -> test
      item_tracking.record_selection("test_picker", "build", cwd)
      item_tracking.record_selection("test_picker", "test", cwd)

      local ctx = { meta = {} }
      local config = helpers.get_default_config()

      item_source.capture_context("test_picker", ctx, config)

      -- last_cwd_selected is "test", which has no outgoing transitions
      -- But "build" -> "test" was recorded, so transition_scores from "test" is empty
      -- Let's check that transition_scores field exists
      -- The context should have transition_scores (possibly empty table or nil)
      assert.is_not_nil(ctx.meta.nos_ctx)

      -- Now record another selection so "test" has outgoing transitions
      item_tracking.record_selection("test_picker", "build", cwd)

      local ctx2 = { meta = {} }
      item_source.capture_context("test_picker", ctx2, config)

      -- last_cwd_selected is "build", which has transition to "test"
      assert.is_not_nil(ctx2.meta.nos_ctx.transition_scores)
      assert.is_true(ctx2.meta.nos_ctx.transition_scores["test"] > 0)
    end)
  end)

  describe("create_item_transform", function()
    local transform
    local ctx

    before_each(function()
      local mock_algorithm = {
        calculate_score = function(input_buf)
          local score = 0
          for i = 1, #input_buf do
            score = score + input_buf[i]
          end
          return score
        end,
        init = function() end,
        load_weights = function() end,
        get_name = function()
          return "naive"
        end,
        update_weights = function() end,
        debug_view = function()
          return {}
        end,
      }

      ctx = {
        meta = {
          nos_ctx = {
            cwd = "/test/proj",
            algorithm = mock_algorithm,
            tracking_data = {
              frecency = { build = 2.5, test = 1.0 },
              cwd_frecency = { build = 1.5 },
              recency_rank = { build = 1, test = 2 },
              cwd_recency_rank = { build = 1 },
              last_selected = "build",
            },
            picker_name = "test_picker",
          },
        },
      }

      local config = helpers.get_default_config()
      transform = item_source.create_item_transform("test_picker", config, item_scorer)
    end)

    it("attaches nos field with correct structure", function()
      local item = { text = "test", value = "test" }
      local result = transform(item, ctx)

      assert.is_not_nil(result)
      assert.is_not_nil(item.nos)
      assert.equals("test", item.nos.item_id)
      assert.is_not_nil(item.nos.raw_features)
      assert.is_not_nil(item.nos.input_buf)
      assert.equals(8, #item.nos.input_buf)
      assert.equals(0, item.nos.neural_score)
      assert.equals(ctx.meta.nos_ctx, item.nos.ctx)
    end)

    it("uses item.value as identity when present", function()
      local item = { text = "Build Project", value = "build" }
      transform(item, ctx)

      assert.equals("build", item.nos.item_id)
    end)

    it("falls back to item.text when value is nil", function()
      local item = { text = "test" }
      transform(item, ctx)

      assert.equals("test", item.nos.item_id)
    end)

    it("deduplicates by item identity", function()
      local item1 = { text = "build", value = "build" }
      local item2 = { text = "Build (dup)", value = "build" }

      local result1 = transform(item1, ctx)
      local result2 = transform(item2, ctx)

      assert.is_not_nil(result1)
      assert.equals(false, result2)
    end)

    it("returns item for items without text", function()
      local item = { value = "something" }
      local result = transform(item, ctx)
      -- No text means no transform
      assert.equals(item, result)
      assert.is_nil(item.nos)
    end)

    it("computes frecency features from tracking data", function()
      local item = { text = "build", value = "build" }
      transform(item, ctx)

      -- build has frecency=2.5, cwd_frecency=1.5
      assert.equals(2.5, item.nos.raw_features.frecency)
      assert.equals(1.5, item.nos.raw_features.cwd_frecency)
      -- Normalized values in input_buf
      assert.are.near(item_scorer.normalize_item_frecency(2.5), item.nos.input_buf[2], 1e-10)
      assert.are.near(item_scorer.normalize_item_frecency(1.5), item.nos.input_buf[3], 1e-10)
    end)

    it("computes recency features from tracking data", function()
      local item = { text = "build", value = "build" }
      transform(item, ctx)

      -- build has recency_rank=1, cwd_recency_rank=1
      assert.equals(1, item.nos.raw_features.recency)
      assert.equals(1, item.nos.raw_features.cwd_recency)
      -- Normalized values
      assert.are.near(item_scorer.calculate_recency_score(1, 100), item.nos.input_buf[4], 1e-10)
      assert.are.near(item_scorer.calculate_recency_score(1, 100), item.nos.input_buf[5], 1e-10)
    end)

    it("computes text_length_inv feature", function()
      local item = { text = "build", value = "build" }
      transform(item, ctx)

      assert.equals(5, item.nos.raw_features.text_length_inv)
      assert.are.near(item_scorer.normalize_text_length(5), item.nos.input_buf[6], 1e-10)
    end)

    it("sets not_last_selected to 0 for the last selected item", function()
      local item = { text = "build", value = "build" }
      transform(item, ctx)

      -- build IS the last selected, so not_last_selected = 0
      assert.equals(0, item.nos.raw_features.not_last_selected)
      assert.equals(0, item.nos.input_buf[7])
    end)

    it("sets not_last_selected to 1 for non-last-selected items", function()
      local item = { text = "test", value = "test" }
      transform(item, ctx)

      -- test is NOT the last selected, so not_last_selected = 1
      assert.equals(1, item.nos.raw_features.not_last_selected)
      assert.equals(1, item.nos.input_buf[7])
    end)

    it("computes transition feature from transition_scores", function()
      -- Add transition_scores to context
      ctx.meta.nos_ctx.transition_scores = { build = 0.7, test = 0.3 }

      local item = { text = "build", value = "build" }
      transform(item, ctx)

      assert.equals(0.7, item.nos.raw_features.transition)
      assert.are.near(0.7, item.nos.input_buf[8], 1e-10)
    end)

    it("sets transition to 0 when no transition_scores in context", function()
      ctx.meta.nos_ctx.transition_scores = nil

      local item = { text = "build", value = "build" }
      transform(item, ctx)

      assert.equals(0, item.nos.raw_features.transition)
      assert.equals(0, item.nos.input_buf[8])
    end)

    it("handles items with no tracking history", function()
      local item = { text = "new_item", value = "new_item" }
      transform(item, ctx)

      assert.equals(0, item.nos.raw_features.frecency)
      assert.equals(0, item.nos.raw_features.cwd_frecency)
      assert.equals(0, item.nos.raw_features.recency)
      assert.equals(0, item.nos.raw_features.cwd_recency)
      assert.equals(1, item.nos.raw_features.not_last_selected)
      assert.equals(0, item.nos.raw_features.transition)

      -- Normalized: frecency=0, recency=0, transition=0
      assert.equals(0, item.nos.input_buf[2])
      assert.equals(0, item.nos.input_buf[3])
      assert.equals(0, item.nos.input_buf[4])
      assert.equals(0, item.nos.input_buf[5])
      assert.equals(0, item.nos.input_buf[8])
    end)

    it("leaves match slot at 0 in input_buf", function()
      local item = { text = "build", value = "build" }
      transform(item, ctx)

      assert.equals(0, item.nos.input_buf[1])
      assert.equals(0, item.nos.raw_features.match)
    end)
  end)

  describe("integration: transform + on_match", function()
    it("produces valid scores through the full pipeline", function()
      local score_values = {}
      local mock_algorithm = {
        calculate_score = function(input_buf)
          local score = 0
          for i = 1, #input_buf do
            score = score + input_buf[i]
          end
          score_values[#score_values + 1] = score
          return score
        end,
        init = function() end,
        load_weights = function() end,
        get_name = function()
          return "naive"
        end,
        update_weights = function() end,
        debug_view = function()
          return {}
        end,
      }

      local ctx = {
        meta = {
          nos_ctx = {
            cwd = "/test/proj",
            algorithm = mock_algorithm,
            tracking_data = {
              frecency = { build = 3.0 },
              cwd_frecency = { build = 2.0 },
              recency_rank = { build = 1 },
              cwd_recency_rank = { build = 1 },
              last_selected = "other",
            },
            picker_name = "test_picker",
          },
        },
      }

      local config = helpers.get_default_config()
      local transform = item_source.create_item_transform("test_picker", config, item_scorer)

      local item = { text = "build", value = "build" }
      transform(item, ctx)

      -- Simulate on_match with a query
      local mock_matcher = {
        pattern = "bui",
        match = function(_, match_item)
          if match_item.text == "build" then
            return 120
          end
          return 0
        end,
      }

      item_scorer.on_match_handler(mock_matcher, item)

      assert.is_true(item.score > 0)
      assert.is_true(item.nos.input_buf[1] > 0) -- match is non-zero
      assert.is_true(item.nos.input_buf[2] > 0) -- frecency is non-zero
    end)
  end)
end)
