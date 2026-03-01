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
      tracking_data = {},
      get_tracking = function(_picker_name, _latency_ctx)
        return vim.deepcopy(mock_db.tracking_data)
      end,
      save_tracking = function(_picker_name, data, _latency_ctx)
        mock_db.tracking_data = vim.deepcopy(data)
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
    it("adds a new file to pending touches (no disk I/O)", function()
      local read_count = 0
      local orig_get = mock_db.get_tracking
      mock_db.get_tracking = function(...)
        read_count = read_count + 1
        return orig_get(...)
      end

      recent.record_buffer_focus("/path/to/file.lua")

      -- No disk reads should have occurred
      assert.equals(0, read_count)
    end)

    it("pending touches appear in get_recency_map at position 1", function()
      recent.record_buffer_focus("/path/to/file.lua")

      local map = recent.get_recency_map()
      assert.equals(1, map["/path/to/file.lua"].recent_rank)
    end)

    it("merges pending touches with existing on-disk data", function()
      mock_db.tracking_data = {
        recency_list = { "/path/old1.lua", "/path/old2.lua" },
      }

      recent.record_buffer_focus("/path/new.lua")

      local map = recent.get_recency_map()
      assert.equals(1, map["/path/new.lua"].recent_rank)
      assert.equals(2, map["/path/old1.lua"].recent_rank)
      assert.equals(3, map["/path/old2.lua"].recent_rank)
    end)

    it("moves an already-pending file to position 1", function()
      recent.record_buffer_focus("/path/a.lua")
      recent.record_buffer_focus("/path/b.lua")
      recent.record_buffer_focus("/path/c.lua")

      -- Now re-focus a.lua
      recent.record_buffer_focus("/path/a.lua")

      local map = recent.get_recency_map()
      assert.equals(1, map["/path/a.lua"].recent_rank)
      assert.equals(2, map["/path/c.lua"].recent_rank)
      assert.equals(3, map["/path/b.lua"].recent_rank)
    end)

    it("deduplicates pending touches with on-disk entries", function()
      mock_db.tracking_data = {
        recency_list = { "/path/old1.lua", "/path/old2.lua" },
      }

      -- Focus a file that's already on disk
      recent.record_buffer_focus("/path/old2.lua")

      local map = recent.get_recency_map()
      assert.equals(1, map["/path/old2.lua"].recent_rank)
      assert.equals(2, map["/path/old1.lua"].recent_rank)
      -- Should only appear once
      local count = 0
      for _ in pairs(map) do
        count = count + 1
      end
      assert.equals(2, count)
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

    it("returns empty map when no pending touches and empty disk", function()
      local map = recent.get_recency_map()
      assert.same({}, map)
    end)

    it("reads from disk on every call (no caching)", function()
      mock_db.tracking_data = {
        recency_list = { "/path/a.lua" },
      }

      local map1 = recent.get_recency_map()
      assert.equals(1, map1["/path/a.lua"].recent_rank)

      -- Externally modify disk data
      mock_db.tracking_data = {
        recency_list = { "/path/b.lua", "/path/a.lua" },
      }

      local map2 = recent.get_recency_map()
      assert.equals(1, map2["/path/b.lua"].recent_rank)
      assert.equals(2, map2["/path/a.lua"].recent_rank)
    end)

    it("pending touches consume limit slots first", function()
      mock_db.tracking_data = {
        recency_list = { "/path/disk1.lua", "/path/disk2.lua", "/path/disk3.lua" },
      }

      recent.record_buffer_focus("/path/pending1.lua")
      recent.record_buffer_focus("/path/pending2.lua")
      recent.record_buffer_focus("/path/pending3.lua")

      -- Limit to 4: all 3 pending + 1 disk
      local map = recent.get_recency_map(4)
      assert.equals(1, map["/path/pending3.lua"].recent_rank)
      assert.equals(2, map["/path/pending2.lua"].recent_rank)
      assert.equals(3, map["/path/pending1.lua"].recent_rank)
      assert.equals(4, map["/path/disk1.lua"].recent_rank)
      assert.is_nil(map["/path/disk2.lua"])
    end)
  end)

  describe("flush", function()
    it("writes merged list to the tracking file", function()
      recent.record_buffer_focus("/path/a.lua")
      recent.record_buffer_focus("/path/b.lua")

      recent.flush()

      local data = mock_db.get_tracking()
      assert.is_not_nil(data.recency_list)
      assert.equals(2, #data.recency_list)
      assert.equals("/path/b.lua", data.recency_list[1])
      assert.equals("/path/a.lua", data.recency_list[2])
    end)

    it("is a no-op when no pending touches exist", function()
      local save_count = 0
      local original_save = mock_db.save_tracking
      mock_db.save_tracking = function(picker_name, data, latency_ctx)
        save_count = save_count + 1
        original_save(picker_name, data, latency_ctx)
      end

      recent.flush()

      assert.equals(0, save_count)
    end)

    it("clears pending state so subsequent flush is a no-op", function()
      local save_count = 0
      local original_save = mock_db.save_tracking
      mock_db.save_tracking = function(picker_name, data, latency_ctx)
        save_count = save_count + 1
        original_save(picker_name, data, latency_ctx)
      end

      recent.record_buffer_focus("/path/a.lua")
      recent.flush()
      assert.equals(1, save_count)

      -- Second flush should be a no-op
      recent.flush()
      assert.equals(1, save_count)
    end)

    it("persisted list can be loaded from disk after flush", function()
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

    it("preserves external entries when merging with disk", function()
      -- Simulate data written by another instance
      mock_db.tracking_data = {
        recency_list = { "/path/external1.lua", "/path/external2.lua" },
      }

      recent.record_buffer_focus("/path/local.lua")
      recent.flush()

      local data = mock_db.get_tracking()
      assert.equals("/path/local.lua", data.recency_list[1])
      assert.equals("/path/external1.lua", data.recency_list[2])
      assert.equals("/path/external2.lua", data.recency_list[3])
    end)

    it("trims merged list to recency_list_size", function()
      mock_db.tracking_data = {
        recency_list = { "/path/d1.lua", "/path/d2.lua", "/path/d3.lua" },
      }

      recent.record_buffer_focus("/path/p1.lua")
      recent.record_buffer_focus("/path/p2.lua")
      recent.record_buffer_focus("/path/p3.lua")

      -- recency_list_size is 5, so 3 pending + 2 disk entries
      recent.flush()

      local data = mock_db.get_tracking()
      assert.equals(5, #data.recency_list)
      assert.equals("/path/p3.lua", data.recency_list[1])
      assert.equals("/path/p2.lua", data.recency_list[2])
      assert.equals("/path/p1.lua", data.recency_list[3])
      assert.equals("/path/d1.lua", data.recency_list[4])
      assert.equals("/path/d2.lua", data.recency_list[5])
    end)
  end)
end)
