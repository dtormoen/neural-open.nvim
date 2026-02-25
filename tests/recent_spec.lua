describe("recent module", function()
  local helpers = require("tests.helpers")
  local recent
  local mock_db
  local original_new_timer

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    -- Mock vim.loop.new_timer to return a stub (timer won't fire in busted)
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
      get_weights = function()
        return vim.deepcopy(mock_db.weights_data)
      end,
      save_weights = function(data)
        mock_db.weights_data = vim.deepcopy(data)
      end,
    }

    -- Mock init module for config
    package.loaded["neural-open.db"] = mock_db
    package.loaded["neural-open"] = {
      config = { recency_list_size = 5 },
    }

    recent = require("neural-open.recent")
  end)

  after_each(function()
    vim.loop.new_timer = original_new_timer
    helpers.clear_plugin_modules()
    package.loaded["neural-open.db"] = nil
  end)

  describe("record_buffer_focus", function()
    it("adds a new file to an empty list at position 1", function()
      recent.record_buffer_focus("/path/to/file.lua")

      local list = recent.get_recency_list()
      assert.equals(1, #list)
      assert.equals("/path/to/file.lua", list[1])
    end)

    it("adds a new file to a seeded list at position 1", function()
      mock_db.weights_data = {
        recency_list = { "/path/old1.lua", "/path/old2.lua" },
      }

      -- Re-load module to pick up persisted data
      package.loaded["neural-open.recent"] = nil
      recent = require("neural-open.recent")

      recent.record_buffer_focus("/path/new.lua")

      local list = recent.get_recency_list()
      assert.equals("/path/new.lua", list[1])
      assert.equals("/path/old1.lua", list[2])
      assert.equals("/path/old2.lua", list[3])
    end)

    it("moves an already-present file to position 1", function()
      recent.record_buffer_focus("/path/a.lua")
      recent.record_buffer_focus("/path/b.lua")
      recent.record_buffer_focus("/path/c.lua")

      -- Now re-focus a.lua
      recent.record_buffer_focus("/path/a.lua")

      local list = recent.get_recency_list()
      assert.equals("/path/a.lua", list[1])
      assert.equals("/path/c.lua", list[2])
      assert.equals("/path/b.lua", list[3])
      assert.equals(3, #list)
    end)

    it("trims the list to recency_list_size", function()
      for i = 1, 7 do
        recent.record_buffer_focus("/path/file" .. i .. ".lua")
      end

      local list = recent.get_recency_list()
      assert.equals(5, #list)

      -- Most recent should be file7, oldest kept should be file3
      assert.equals("/path/file7.lua", list[1])
      assert.equals("/path/file3.lua", list[5])
    end)
  end)

  describe("get_recency_map", function()
    it("returns correct path-to-rank structure", function()
      recent.record_buffer_focus("/path/a.lua")
      recent.record_buffer_focus("/path/b.lua")
      recent.record_buffer_focus("/path/c.lua")

      local map = recent.get_recency_map()

      assert.equals(1, map["/path/c.lua"].recent_rank)
      assert.equals(2, map["/path/b.lua"].recent_rank)
      assert.equals(3, map["/path/a.lua"].recent_rank)
    end)

    it("respects limit parameter", function()
      recent.record_buffer_focus("/path/a.lua")
      recent.record_buffer_focus("/path/b.lua")
      recent.record_buffer_focus("/path/c.lua")

      local map = recent.get_recency_map(2)

      assert.equals(1, map["/path/c.lua"].recent_rank)
      assert.equals(2, map["/path/b.lua"].recent_rank)
      assert.is_nil(map["/path/a.lua"])
    end)

    it("returns empty map for empty list", function()
      local map = recent.get_recency_map()
      assert.same({}, map)
    end)
  end)

  describe("flush", function()
    it("writes the recency_list to the weights file", function()
      recent.record_buffer_focus("/path/a.lua")
      recent.record_buffer_focus("/path/b.lua")

      recent.flush()

      local data = mock_db.get_weights()
      assert.is_not_nil(data.recency_list)
      assert.equals(2, #data.recency_list)
      assert.equals("/path/b.lua", data.recency_list[1])
      assert.equals("/path/a.lua", data.recency_list[2])
    end)

    it("is a no-op when no changes have been made", function()
      local save_count = 0
      local original_save = mock_db.save_weights
      mock_db.save_weights = function(data)
        save_count = save_count + 1
        original_save(data)
      end

      -- Trigger ensure_loaded via get_recency_list (read-only, no dirty flag)
      recent.get_recency_list()
      recent.flush()

      assert.equals(0, save_count)
    end)

    it("preserves existing weights data", function()
      mock_db.weights_data = {
        classic = { match = 100 },
        nn = { weights = { { { 1, 2 } } } },
        transition_history = { { from = "/a", to = "/b", timestamp = 1 } },
      }

      -- Re-load to pick up existing data
      package.loaded["neural-open.recent"] = nil
      recent = require("neural-open.recent")

      recent.record_buffer_focus("/path/new.lua")
      recent.flush()

      local data = mock_db.get_weights()
      assert.equals(100, data.classic.match)
      assert.is_not_nil(data.nn.weights)
      assert.equals(1, #data.transition_history)
      assert.is_not_nil(data.recency_list)
    end)

    it("resets dirty flag so subsequent flush is a no-op", function()
      local save_count = 0
      local original_save = mock_db.save_weights
      mock_db.save_weights = function(data)
        save_count = save_count + 1
        original_save(data)
      end

      recent.record_buffer_focus("/path/a.lua")
      recent.flush()
      assert.equals(1, save_count)

      -- Second flush should be a no-op
      recent.flush()
      assert.equals(1, save_count)
    end)

    it("persisted list can be loaded from disk", function()
      recent.record_buffer_focus("/path/x.lua")
      recent.record_buffer_focus("/path/y.lua")
      recent.flush()

      -- Re-load the module to simulate a restart
      package.loaded["neural-open.recent"] = nil
      recent = require("neural-open.recent")

      local map = recent.get_recency_map()
      assert.equals(1, map["/path/y.lua"].recent_rank)
      assert.equals(2, map["/path/x.lua"].recent_rank)
    end)
  end)
end)
