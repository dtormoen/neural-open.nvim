describe("item_tracking module", function()
  local helpers = require("tests.helpers")
  local item_tracking
  local mock_db
  local original_new_timer
  local original_os_time
  local mock_time

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    -- Mock vim.loop.new_timer
    original_new_timer = vim.loop.new_timer
    vim.loop.new_timer = function()
      return {
        start = function() end,
        stop = function() end,
        close = function() end,
      }
    end

    -- Control time for deterministic tests
    mock_time = 1000000000
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
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
    package.loaded["neural-open"] = {
      config = { recency_list_size = 5 },
    }

    item_tracking = require("neural-open.item_tracking")
  end)

  after_each(function()
    if item_tracking then
      item_tracking.reset()
    end
    vim.loop.new_timer = original_new_timer
    os.time = original_os_time -- luacheck: ignore 122
    helpers.clear_plugin_modules()
    package.loaded["neural-open.db"] = nil
  end)

  describe("init", function()
    it("loads empty tracking data when no persisted data exists", function()
      item_tracking.init("test_picker")

      local data = item_tracking.get_tracking_data("test_picker", "/cwd")
      assert.same({}, data.frecency)
      assert.same({}, data.cwd_frecency)
      assert.same({}, data.recency_rank)
      assert.same({}, data.cwd_recency_rank)
      assert.is_nil(data.last_selected)
    end)

    it("loads persisted tracking data from disk", function()
      mock_db.weights_data = {
        item_tracking = {
          frecency = { build = mock_time + 1000 },
          cwd_frecency = { ["/proj"] = { build = mock_time + 1000 } },
          recency_list = { "build", "test" },
          cwd_recency = { ["/proj"] = { "build" } },
        },
      }

      item_tracking.init("test_picker")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.is_true(data.frecency["build"] > 0)
      assert.is_true(data.cwd_frecency["build"] > 0)
      assert.equals(1, data.recency_rank["build"])
      assert.equals(2, data.recency_rank["test"])
      assert.equals(1, data.cwd_recency_rank["build"])
      assert.equals("build", data.last_selected)
    end)
  end)

  describe("record_selection", function()
    it("updates all four tracking stores", function()
      item_tracking.record_selection("test_picker", "build", "/proj")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.is_true(data.frecency["build"] > 0)
      assert.is_true(data.cwd_frecency["build"] > 0)
      assert.equals(1, data.recency_rank["build"])
      assert.equals(1, data.cwd_recency_rank["build"])
      assert.equals("build", data.last_selected)
    end)

    it("increases frecency score with repeated selections", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      local data1 = item_tracking.get_tracking_data("test_picker", "/proj")
      local score1 = data1.frecency["build"]

      item_tracking.record_selection("test_picker", "build", "/proj")
      local data2 = item_tracking.get_tracking_data("test_picker", "/proj")
      local score2 = data2.frecency["build"]

      assert.is_true(score2 > score1)
    end)

    it("maintains correct recency ordering", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")
      item_tracking.record_selection("test_picker", "lint", "/proj")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.equals(1, data.recency_rank["lint"])
      assert.equals(2, data.recency_rank["test"])
      assert.equals(3, data.recency_rank["build"])
      assert.equals("lint", data.last_selected)
    end)

    it("moves re-selected item to front of recency list", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")
      item_tracking.record_selection("test_picker", "lint", "/proj")

      -- Re-select build
      item_tracking.record_selection("test_picker", "build", "/proj")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.equals(1, data.recency_rank["build"])
      assert.equals(2, data.recency_rank["lint"])
      assert.equals(3, data.recency_rank["test"])
    end)

    it("trims recency lists to max size", function()
      for i = 1, 7 do
        item_tracking.record_selection("test_picker", "item" .. i, "/proj")
      end

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      -- Max size is 5, so only items 3-7 should remain
      assert.equals(1, data.recency_rank["item7"])
      assert.equals(5, data.recency_rank["item3"])
      assert.is_nil(data.recency_rank["item2"])
      assert.is_nil(data.recency_rank["item1"])
    end)
  end)

  describe("CWD scoping", function()
    it("tracks frecency separately per CWD", function()
      item_tracking.record_selection("test_picker", "build", "/proj_a")
      item_tracking.record_selection("test_picker", "build", "/proj_a")
      item_tracking.record_selection("test_picker", "build", "/proj_b")

      local data_a = item_tracking.get_tracking_data("test_picker", "/proj_a")
      local data_b = item_tracking.get_tracking_data("test_picker", "/proj_b")

      -- proj_a has 2 selections, proj_b has 1
      assert.is_true(data_a.cwd_frecency["build"] > data_b.cwd_frecency["build"])
    end)

    it("tracks recency separately per CWD", function()
      item_tracking.record_selection("test_picker", "build", "/proj_a")
      item_tracking.record_selection("test_picker", "test", "/proj_a")
      item_tracking.record_selection("test_picker", "lint", "/proj_b")

      local data_a = item_tracking.get_tracking_data("test_picker", "/proj_a")
      local data_b = item_tracking.get_tracking_data("test_picker", "/proj_b")

      assert.equals(1, data_a.cwd_recency_rank["test"])
      assert.equals(2, data_a.cwd_recency_rank["build"])
      assert.is_nil(data_a.cwd_recency_rank["lint"])

      assert.equals(1, data_b.cwd_recency_rank["lint"])
      assert.is_nil(data_b.cwd_recency_rank["build"])
    end)

    it("shares global tracking across CWDs", function()
      item_tracking.record_selection("test_picker", "build", "/proj_a")
      item_tracking.record_selection("test_picker", "test", "/proj_b")

      local data = item_tracking.get_tracking_data("test_picker", "/proj_a")

      -- Global frecency has both items
      assert.is_true(data.frecency["build"] > 0)
      assert.is_true(data.frecency["test"] > 0)

      -- Global recency has both items
      assert.equals(1, data.recency_rank["test"])
      assert.equals(2, data.recency_rank["build"])
    end)

    it("returns empty CWD data for unknown CWD", function()
      item_tracking.record_selection("test_picker", "build", "/proj_a")

      local data = item_tracking.get_tracking_data("test_picker", "/unknown")
      assert.same({}, data.cwd_frecency)
      assert.same({}, data.cwd_recency_rank)
    end)
  end)

  describe("frecency decay", function()
    it("decays scores over time", function()
      item_tracking.record_selection("test_picker", "build", "/proj")

      local data_now = item_tracking.get_tracking_data("test_picker", "/proj")
      local score_now = data_now.frecency["build"]

      -- Advance 30 days (one half-life)
      mock_time = mock_time + 30 * 24 * 3600

      local data_later = item_tracking.get_tracking_data("test_picker", "/proj")
      local score_later = data_later.frecency["build"]

      assert.is_true(score_later < score_now)
    end)

    it("halves raw score after one half-life", function()
      item_tracking.record_selection("test_picker", "build", "/proj")

      local data_now = item_tracking.get_tracking_data("test_picker", "/proj")
      local score_now = data_now.frecency["build"]

      -- Advance exactly one half-life
      mock_time = mock_time + 30 * 24 * 3600

      local data_later = item_tracking.get_tracking_data("test_picker", "/proj")
      local score_later = data_later.frecency["build"]

      assert.is_near(score_now / 2, score_later, 0.01)
    end)

    it("decays CWD frecency the same way", function()
      item_tracking.record_selection("test_picker", "build", "/proj")

      local data_now = item_tracking.get_tracking_data("test_picker", "/proj")
      local cwd_score_now = data_now.cwd_frecency["build"]

      mock_time = mock_time + 30 * 24 * 3600

      local data_later = item_tracking.get_tracking_data("test_picker", "/proj")
      local cwd_score_later = data_later.cwd_frecency["build"]

      assert.is_near(cwd_score_now / 2, cwd_score_later, 0.01)
    end)
  end)

  describe("flush", function()
    it("persists tracking data to the picker's weight file", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")

      item_tracking.flush("test_picker")

      local data = mock_db.weights_data
      assert.is_not_nil(data.item_tracking)
      assert.is_not_nil(data.item_tracking.frecency["build"])
      assert.is_not_nil(data.item_tracking.frecency["test"])
      assert.same({ "test", "build" }, data.item_tracking.recency_list)
    end)

    it("is a no-op when no changes have been made", function()
      local save_count = 0
      local original_save = mock_db.save_weights
      mock_db.save_weights = function(picker_name, data, latency_ctx)
        save_count = save_count + 1
        original_save(picker_name, data, latency_ctx)
      end

      item_tracking.init("test_picker")
      item_tracking.flush("test_picker")

      assert.equals(0, save_count)
    end)

    it("resets dirty flag so subsequent flush is a no-op", function()
      local save_count = 0
      local original_save = mock_db.save_weights
      mock_db.save_weights = function(picker_name, data, latency_ctx)
        save_count = save_count + 1
        original_save(picker_name, data, latency_ctx)
      end

      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.flush("test_picker")
      assert.equals(1, save_count)

      item_tracking.flush("test_picker")
      assert.equals(1, save_count)
    end)

    it("preserves existing weight data in the picker file", function()
      mock_db.weights_data = {
        nn = { weights = { { { 1, 2 } } } },
        classic = { match = 100 },
      }

      -- Re-load to pick up existing data
      package.loaded["neural-open.item_tracking"] = nil
      item_tracking = require("neural-open.item_tracking")

      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.flush("test_picker")

      local data = mock_db.weights_data
      assert.equals(100, data.classic.match)
      assert.is_not_nil(data.nn.weights)
      assert.is_not_nil(data.item_tracking)
    end)

    it("persisted data can be loaded after module reload", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")
      item_tracking.flush("test_picker")

      -- Reload module
      item_tracking.reset()
      package.loaded["neural-open.item_tracking"] = nil
      item_tracking = require("neural-open.item_tracking")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.equals(1, data.recency_rank["test"])
      assert.equals(2, data.recency_rank["build"])
      assert.is_true(data.frecency["build"] > 0)
      assert.is_true(data.frecency["test"] > 0)
    end)

    it("is a no-op for unknown picker name", function()
      -- Should not error
      item_tracking.flush("nonexistent")
    end)
  end)

  describe("picker isolation", function()
    it("tracks data independently per picker", function()
      item_tracking.record_selection("picker_a", "build", "/proj")
      item_tracking.record_selection("picker_b", "test", "/proj")

      local data_a = item_tracking.get_tracking_data("picker_a", "/proj")
      local data_b = item_tracking.get_tracking_data("picker_b", "/proj")

      assert.is_true(data_a.frecency["build"] > 0)
      assert.is_nil(data_a.frecency["test"])

      assert.is_true(data_b.frecency["test"] > 0)
      assert.is_nil(data_b.frecency["build"])
    end)

    it("flushes pickers independently", function()
      local save_picker_names = {}
      mock_db.save_weights = function(picker_name, data, _latency_ctx)
        save_picker_names[#save_picker_names + 1] = picker_name
      end

      item_tracking.record_selection("picker_a", "build", "/proj")
      item_tracking.record_selection("picker_b", "test", "/proj")

      item_tracking.flush("picker_a")
      item_tracking.flush("picker_b")

      assert.equals(2, #save_picker_names)
      assert.equals("picker_a", save_picker_names[1])
      assert.equals("picker_b", save_picker_names[2])
    end)
  end)

  describe("pruning", function()
    it("prunes global frecency entries beyond limit", function()
      local orig = item_tracking.MAX_FRECENCY_ITEMS
      item_tracking.MAX_FRECENCY_ITEMS = 3

      for i = 1, 5 do
        item_tracking.record_selection("test_picker", "item" .. i, "/proj")
      end

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      local count = 0
      for _ in pairs(data.frecency) do
        count = count + 1
      end
      assert.equals(3, count)

      item_tracking.MAX_FRECENCY_ITEMS = orig
    end)

    it("keeps highest-scoring frecency entries when pruning", function()
      local orig = item_tracking.MAX_FRECENCY_ITEMS
      item_tracking.MAX_FRECENCY_ITEMS = 2

      -- Give item1 a high score (3 selections)
      for _ = 1, 3 do
        item_tracking.record_selection("test_picker", "item1", "/proj")
      end

      -- item2 and item3 get 1 selection each
      item_tracking.record_selection("test_picker", "item2", "/proj")
      item_tracking.record_selection("test_picker", "item3", "/proj")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")

      -- item1 (highest score) should survive
      assert.is_true(data.frecency["item1"] ~= nil)

      item_tracking.MAX_FRECENCY_ITEMS = orig
    end)

    it("prunes CWD frecency entries beyond limit", function()
      local orig = item_tracking.MAX_CWD_FRECENCY_ITEMS
      item_tracking.MAX_CWD_FRECENCY_ITEMS = 2

      for i = 1, 4 do
        item_tracking.record_selection("test_picker", "item" .. i, "/proj")
      end

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      local count = 0
      for _ in pairs(data.cwd_frecency) do
        count = count + 1
      end
      assert.equals(2, count)

      item_tracking.MAX_CWD_FRECENCY_ITEMS = orig
    end)
  end)

  describe("transition tracking", function()
    it("records transition from previous CWD selection", function()
      -- First selection: no transition (no previous item)
      item_tracking.record_selection("test_picker", "build", "/proj")
      -- Second selection: should record build -> test transition
      item_tracking.record_selection("test_picker", "test", "/proj")

      local scores = item_tracking.compute_transition_scores("test_picker", "build")
      assert.is_true(scores["test"] > 0)
    end)

    it("records self-transitions", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "build", "/proj")

      local scores = item_tracking.compute_transition_scores("test_picker", "build")
      assert.is_true(scores["build"] > 0)
    end)

    it("increases score with repeated transitions", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")

      local scores1 = item_tracking.compute_transition_scores("test_picker", "build")
      local score1 = scores1["test"]

      -- Go back and forth to record another build -> test
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")

      local scores2 = item_tracking.compute_transition_scores("test_picker", "build")
      local score2 = scores2["test"]

      assert.is_true(score2 > score1)
    end)

    it("tracks multiple destinations from same source", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")

      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "lint", "/proj")

      local scores = item_tracking.compute_transition_scores("test_picker", "build")
      assert.is_true(scores["test"] > 0)
      assert.is_true(scores["lint"] > 0)
    end)

    it("scopes 'from' item by CWD", function()
      -- Select in proj_a
      item_tracking.record_selection("test_picker", "build", "/proj_a")
      -- Select in proj_b (different CWD, no previous item in this CWD)
      item_tracking.record_selection("test_picker", "test", "/proj_b")

      -- No transition should be recorded from "build" because it was in a different CWD
      local scores = item_tracking.compute_transition_scores("test_picker", "build")
      assert.is_nil(scores["test"])
    end)

    it("returns empty scores for unknown source", function()
      local scores = item_tracking.compute_transition_scores("test_picker", "nonexistent")
      assert.same({}, scores)
    end)

    it("persists transitions through flush/reload", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")
      item_tracking.flush("test_picker")

      -- Reload module
      item_tracking.reset()
      package.loaded["neural-open.item_tracking"] = nil
      item_tracking = require("neural-open.item_tracking")

      local scores = item_tracking.compute_transition_scores("test_picker", "build")
      assert.is_true(scores["test"] > 0)
    end)

    it("prunes destinations beyond limit", function()
      local orig = item_tracking.MAX_TRANSITION_DESTINATIONS
      item_tracking.MAX_TRANSITION_DESTINATIONS = 3

      -- Create transitions: build -> item1, build -> item2, ..., build -> item5
      for i = 1, 5 do
        item_tracking.record_selection("test_picker", "build", "/proj")
        item_tracking.record_selection("test_picker", "item" .. i, "/proj")
      end

      local scores = item_tracking.compute_transition_scores("test_picker", "build")
      local count = 0
      for _ in pairs(scores) do
        count = count + 1
      end
      assert.equals(3, count)

      item_tracking.MAX_TRANSITION_DESTINATIONS = orig
    end)

    it("prunes sources beyond limit", function()
      local orig = item_tracking.MAX_TRANSITION_SOURCES
      item_tracking.MAX_TRANSITION_SOURCES = 3

      -- Create 5 different sources, each with one destination
      for i = 1, 5 do
        local source = "source" .. i
        local dest = "dest" .. i
        item_tracking.record_selection("test_picker", source, "/proj")
        item_tracking.record_selection("test_picker", dest, "/proj")
      end

      local frecency = item_tracking.get_transition_frecency("test_picker")
      local source_count = 0
      for _ in pairs(frecency) do
        source_count = source_count + 1
      end
      assert.is_true(source_count <= 3)

      item_tracking.MAX_TRANSITION_SOURCES = orig
    end)

    it("returns get_transition_frecency table for debug views", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")

      local frecency = item_tracking.get_transition_frecency("test_picker")
      assert.is_not_nil(frecency["build"])
      assert.is_not_nil(frecency["build"]["test"])
    end)

    it("returns last_cwd_selected in tracking data", function()
      item_tracking.record_selection("test_picker", "build", "/proj")
      item_tracking.record_selection("test_picker", "test", "/proj")

      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.equals("test", data.last_cwd_selected)
    end)

    it("returns nil last_cwd_selected for unknown CWD", function()
      item_tracking.record_selection("test_picker", "build", "/proj")

      local data = item_tracking.get_tracking_data("test_picker", "/unknown")
      assert.is_nil(data.last_cwd_selected)
    end)
  end)

  describe("reset", function()
    it("clears all cached data", function()
      item_tracking.record_selection("test_picker", "build", "/proj")

      item_tracking.reset()

      -- After reset, data should be loaded fresh from disk (which is empty since we didn't flush)
      local data = item_tracking.get_tracking_data("test_picker", "/proj")
      assert.same({}, data.frecency)
      assert.same({}, data.recency_rank)
    end)
  end)
end)
