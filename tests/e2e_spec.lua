--- End-to-end tests for multi-picker support.
--- Verifies full pipelines: registration, transforms, scoring, learning, and isolation.
local helpers = require("tests.helpers")

describe("end-to-end multi-picker", function()
  local neural_open
  local mock_db
  local original_os_time
  local mock_time

  -- Per-picker storage simulation
  local db_store
  local tracking_store

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    mock_time = 1000000000
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
    end

    -- Per-picker db store (weights and tracking stored separately)
    db_store = {}
    tracking_store = {}
    mock_db = {
      get_weights = function(picker_name, _latency_ctx)
        return vim.deepcopy(db_store[picker_name] or {})
      end,
      save_weights = function(picker_name, data, _latency_ctx)
        db_store[picker_name] = vim.deepcopy(data)
        return true
      end,
      get_tracking = function(picker_name, _latency_ctx)
        return vim.deepcopy(tracking_store[picker_name] or {})
      end,
      save_tracking = function(picker_name, data, _latency_ctx)
        tracking_store[picker_name] = vim.deepcopy(data)
        return true
      end,
      reset_cache = function() end,
    }
    package.loaded["neural-open.db"] = mock_db

    -- Mock weights module using real logic but backed by mock_db
    package.loaded["neural-open.weights"] = {
      get_weights = function(algo_name, picker_name)
        picker_name = picker_name or "files"
        local stored = mock_db.get_weights(picker_name)
        return stored[algo_name] or {}
      end,
      save_weights = function(algo_name, weights, latency_ctx, picker_name)
        picker_name = picker_name or "files"
        local stored = mock_db.get_weights(picker_name)
        stored[algo_name] = weights
        mock_db.save_weights(picker_name, stored, latency_ctx)
      end,
      get_default_weights = function(_algo, _picker_name)
        return {}
      end,
      reset_weights = function() end,
    }

    neural_open = require("neural-open")
    neural_open.setup({ algorithm = "classic" })
  end)

  after_each(function()
    os.time = original_os_time -- luacheck: ignore 122
    helpers.clear_plugin_modules()
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open.weights"] = nil
  end)

  describe("item picker nos field", function()
    it("attaches nos with 8 features via transform pipeline", function()
      local item_source = require("neural-open.item_source")
      local item_scorer = require("neural-open.item_scorer")
      local config = helpers.get_default_config()

      local ctx = { meta = {} }
      item_source.capture_context("recipes", ctx, config)

      local transform = item_source.create_item_transform("recipes", config, item_scorer)

      local item = { text = "build", value = "build" }
      local result = transform(item, ctx)

      assert.is_not_nil(result)
      assert.is_not_nil(item.nos)
      assert.equals(8, #item.nos.input_buf)
      assert.equals("build", item.nos.item_id)
      assert.is_not_nil(item.nos.raw_features)
      assert.is_not_nil(item.nos.ctx)

      -- Verify all 8 feature names map to raw_features
      for _, name in ipairs(item_scorer.ITEM_FEATURE_NAMES) do
        assert.is_not_nil(item.nos.raw_features[name], "missing raw feature: " .. name)
      end
    end)
  end)

  describe("file picker nos field", function()
    it("attaches nos with 11 features via transform pipeline", function()
      local source_mod = require("neural-open.source")
      local scorer = require("neural-open.scorer")
      local config = helpers.get_default_config()

      local ctx = { meta = {} }
      source_mod.capture_context(ctx)

      local transform = source_mod.create_neural_transform(config, scorer, {})

      local item = { file = "/test/project/file.lua" }
      local result = transform(item, ctx)

      assert.is_not_nil(result)
      assert.is_not_nil(item.nos)
      assert.equals(11, #item.nos.input_buf)
      assert.is_not_nil(item.nos.normalized_path)
      assert.is_not_nil(item.nos.raw_features)
      assert.is_not_nil(item.nos.ctx)

      -- Verify all 11 feature names map to raw_features
      for _, name in ipairs(scorer.FEATURE_NAMES) do
        assert.is_not_nil(item.nos.raw_features[name], "missing raw feature: " .. name)
      end
    end)
  end)

  describe("scoring correctness", function()
    it("items with higher frecency score higher when other features equal", function()
      local item_source = require("neural-open.item_source")
      local item_scorer = require("neural-open.item_scorer")
      local item_tracking = require("neural-open.item_tracking")
      local config = helpers.get_default_config()

      -- Record selections to build frecency
      local cwd = vim.fn.getcwd()
      item_tracking.record_selection("score_test", "popular", cwd)
      item_tracking.record_selection("score_test", "popular", cwd)
      item_tracking.record_selection("score_test", "popular", cwd)
      item_tracking.record_selection("score_test", "rare", cwd)

      -- Capture context with frecency data
      local ctx = { meta = {} }
      item_source.capture_context("score_test", ctx, config)

      local transform = item_source.create_item_transform("score_test", config, item_scorer)

      local popular = { text = "popular", value = "popular" }
      local rare = { text = "rare", value = "rare" }
      transform(popular, ctx)
      transform(rare, ctx)

      -- With no search query, frecency should dominate
      local mock_matcher = {
        pattern = "",
        match = function()
          return 0
        end,
      }
      item_scorer.on_match_handler(mock_matcher, popular)
      item_scorer.on_match_handler(mock_matcher, rare)

      assert.is_true(popular.score > rare.score, "popular item should score higher than rare item")
    end)
  end)

  describe("weight learning", function()
    it("records tracking data in the correct per-picker file", function()
      local item_tracking = require("neural-open.item_tracking")
      local cwd = vim.fn.getcwd()

      item_tracking.record_selection("learn_test", "itemB", cwd)

      local stored = tracking_store["learn_test"]
      assert.is_not_nil(stored, "learn_test picker should have data in tracking store")
      assert.is_not_nil(stored.item_tracking, "item_tracking key should exist")
      assert.is_true(stored.item_tracking.frecency["itemB"] > 0)
    end)

    it("algorithm update_weights persists to the correct per-picker file", function()
      local item_source = require("neural-open.item_source")
      local item_scorer = require("neural-open.item_scorer")
      local item_tracking = require("neural-open.item_tracking")
      local config = helpers.get_default_config()
      local cwd = vim.fn.getcwd()

      -- Pre-seed frecency so itemA scores higher
      item_tracking.record_selection("weight_learn", "itemA", cwd)
      item_tracking.record_selection("weight_learn", "itemA", cwd)
      item_tracking.record_selection("weight_learn", "itemA", cwd)

      -- Setup the item picker pipeline
      local ctx = { meta = {} }
      item_source.capture_context("weight_learn", ctx, config)
      local transform = item_source.create_item_transform("weight_learn", config, item_scorer)

      local itemA = { text = "itemA", value = "itemA" }
      local itemB = { text = "itemB", value = "itemB" }
      transform(itemA, ctx)
      transform(itemB, ctx)

      -- Score both items
      local mock_matcher = {
        pattern = "",
        match = function()
          return 0
        end,
      }
      item_scorer.on_match_handler(mock_matcher, itemA)
      item_scorer.on_match_handler(mock_matcher, itemB)

      -- Simulate: user selects itemB when itemA was ranked higher (rank 2 selection)
      itemB.neural_rank = 2
      local ranked_items = { itemA, itemB }

      -- Call algorithm.update_weights (what the confirm handler does)
      local algorithm = ctx.meta.nos_ctx.algorithm
      algorithm.update_weights(itemB, ranked_items)

      -- Verify algorithm weights were persisted to the correct picker file
      local stored = db_store["weight_learn"]
      assert.is_not_nil(stored, "weight_learn picker should have persisted data")
      assert.is_not_nil(stored.classic, "classic algorithm weights should be persisted")
    end)
  end)

  describe("per-picker isolation", function()
    it("two pickers do not share tracking data", function()
      local item_tracking = require("neural-open.item_tracking")
      local cwd = vim.fn.getcwd()

      -- Record selections in picker A
      item_tracking.record_selection("picker_a", "alpha", cwd)

      -- Record selections in picker B
      item_tracking.record_selection("picker_b", "beta", cwd)

      -- Verify each picker has its own data
      local data_a = tracking_store["picker_a"]
      local data_b = tracking_store["picker_b"]

      assert.is_not_nil(data_a)
      assert.is_not_nil(data_b)

      -- picker_a has alpha, not beta
      assert.is_not_nil(data_a.item_tracking.frecency["alpha"])
      assert.is_nil(data_a.item_tracking.frecency["beta"])

      -- picker_b has beta, not alpha
      assert.is_not_nil(data_b.item_tracking.frecency["beta"])
      assert.is_nil(data_b.item_tracking.frecency["alpha"])
    end)

    it("weight files are separate per picker", function()
      local weights = require("neural-open.weights")

      -- Save weights for two different pickers
      weights.save_weights("classic", { test_weight = 42 }, nil, "picker_x")
      weights.save_weights("classic", { test_weight = 99 }, nil, "picker_y")

      -- Each picker should have its own weights
      local x_weights = weights.get_weights("classic", "picker_x")
      local y_weights = weights.get_weights("classic", "picker_y")

      assert.equals(42, x_weights.test_weight)
      assert.equals(99, y_weights.test_weight)
    end)
  end)

  describe("migration", function()
    it("auto-migrates weights.json to files.json", function()
      helpers.with_temp_db(function(temp_dir)
        -- Create old-format weights.json
        local old_path = temp_dir .. "/weights.json"
        local old_data = vim.json.encode({ classic = { match = 100 } })
        local f = io.open(old_path, "w")
        f:write(old_data)
        f:close()

        helpers.clear_plugin_modules()
        package.loaded["neural-open.db"] = nil

        -- Configure with this temp directory
        local init = require("neural-open")
        init.config.weights_path = temp_dir

        local db = require("neural-open.db")
        db.reset_cache()

        -- Access should trigger migration
        local data = db.get_weights("files")
        assert.is_not_nil(data)
        assert.is_not_nil(data.classic)
        assert.equals(100, data.classic.match)

        -- Verify old file was backed up
        assert.equals(1, vim.fn.filereadable(old_path .. ".bak"))

        -- Verify old file was renamed (no longer exists)
        assert.equals(0, vim.fn.filereadable(old_path))

        -- Verify new file exists
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.json"))
      end)
    end)
  end)

  describe("file_sources config", function()
    it("defaults to standard file sources", function()
      local defaults = helpers.get_default_config()
      assert.are.same({ "buffers", "neural_recent", "files", "git_files" }, defaults.file_sources)
    end)

    it("is respected when overridden via setup", function()
      neural_open.setup({ file_sources = { "files" } })
      assert.are.same({ "files" }, neural_open.config.file_sources)
    end)

    it("is stored in config and accessible", function()
      local custom_sources = { "buffers", "files" }
      neural_open.setup({ file_sources = custom_sources })
      assert.are.same(custom_sources, neural_open.config.file_sources)
    end)
  end)

  describe("item_algorithm_config", function()
    it("has defaults with 8-input NN architecture", function()
      local config = helpers.get_default_config()
      assert.is_not_nil(config.item_algorithm_config)
      assert.is_not_nil(config.item_algorithm_config.nn)
      assert.equals(8, config.item_algorithm_config.nn.architecture[1])
      assert.equals(1, config.item_algorithm_config.nn.architecture[#config.item_algorithm_config.nn.architecture])
    end)

    it("has classic defaults with 8 feature weights", function()
      local config = helpers.get_default_config()
      local classic = config.item_algorithm_config.classic
      assert.is_not_nil(classic.default_weights)
      assert.is_not_nil(classic.default_weights.match)
      assert.is_not_nil(classic.default_weights.frecency)
      assert.is_not_nil(classic.default_weights.cwd_frecency)
      assert.is_not_nil(classic.default_weights.recency)
      assert.is_not_nil(classic.default_weights.cwd_recency)
      assert.is_not_nil(classic.default_weights.text_length_inv)
      assert.is_not_nil(classic.default_weights.not_last_selected)
      assert.is_not_nil(classic.default_weights.transition)
    end)
  end)
end)
