local helpers = require("tests.helpers")

describe("db module", function()
  local db
  local neural_open

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    neural_open = require("neural-open")
    db = require("neural-open.db")
  end)

  after_each(function()
    db.reset_cache()
    helpers.clear_plugin_modules()
  end)

  describe("per-picker storage", function()
    it("should store data in separate files per picker name", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        local files_data = { classic = { match = 100 } }
        local commands_data = { nn = { weights = { 1, 2, 3 } } }

        db.save_weights("files", files_data)
        db.save_weights("commands", commands_data)

        local loaded_files = db.get_weights("files")
        local loaded_commands = db.get_weights("commands")

        assert.are.same(files_data, loaded_files)
        assert.are.same(commands_data, loaded_commands)
      end)
    end)

    it("should not leak data between picker names", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        db.save_weights("files", { key = "files_value" })
        db.save_weights("other", { key = "other_value" })

        local files_result = db.get_weights("files")
        local other_result = db.get_weights("other")

        assert.equals("files_value", files_result.key)
        assert.equals("other_value", other_result.key)
      end)
    end)

    it("should return empty table for non-existent picker", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        local result = db.get_weights("nonexistent")
        assert.are.same({}, result)
      end)
    end)
  end)

  describe("migration", function()
    it("should migrate weights.json to files.json on first access", function()
      helpers.with_temp_db(function(temp_dir)
        -- Create a legacy weights.json file
        local legacy_data = { classic = { match = 42 }, recency_list = { "/a.lua" } }
        local legacy_path = temp_dir .. "/weights.json"
        local f = io.open(legacy_path, "w")
        f:write(vim.json.encode(legacy_data))
        f:close()

        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        -- Trigger migration by accessing db
        local loaded = db.get_weights("files")
        assert.are.same(legacy_data, loaded)

        -- Verify weights.json no longer exists as a regular file
        assert.equals(0, vim.fn.filereadable(legacy_path))

        -- Verify backup was created
        assert.equals(1, vim.fn.filereadable(legacy_path .. ".bak"))

        -- Verify files.json was created
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.json"))
      end)
    end)

    it("should not migrate if files.json already exists", function()
      helpers.with_temp_db(function(temp_dir)
        -- Create both weights.json and files.json
        local legacy_data = { old = true }
        local new_data = { new = true }

        local legacy_f = io.open(temp_dir .. "/weights.json", "w")
        legacy_f:write(vim.json.encode(legacy_data))
        legacy_f:close()

        local new_f = io.open(temp_dir .. "/files.json", "w")
        new_f:write(vim.json.encode(new_data))
        new_f:close()

        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        -- Should read from files.json, not migrate
        local loaded = db.get_weights("files")
        assert.is_true(loaded.new)

        -- weights.json should still exist (not migrated)
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/weights.json"))
      end)
    end)
  end)

  describe("weights_dir resolution", function()
    it("should derive directory from weights_path .json file", function()
      helpers.with_temp_db(function(temp_dir)
        local json_path = temp_dir .. "/files.json"
        neural_open.setup({ weights_path = json_path })
        db.reset_cache()

        local test_data = { classic = { match = 99 } }
        db.save_weights("files", test_data)

        local loaded = db.get_weights("files")
        assert.are.same(test_data, loaded)

        -- Verify the file was created as files.json in the parent dir
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.json"))
      end)
    end)

    it("should use weights_dir when both weights_dir and weights_path are set", function()
      helpers.with_temp_db(function(dir_path)
        helpers.with_temp_db(function(other_dir)
          -- weights_dir should take priority over dirname(weights_path)
          neural_open.setup({
            weights_path = other_dir .. "/files.json",
            weights_dir = dir_path,
          })
          db.reset_cache()

          local test_data = { classic = { match = 55 } }
          db.save_weights("files", test_data)

          -- File should be in weights_dir, not dirname(weights_path)
          assert.equals(1, vim.fn.filereadable(dir_path .. "/files.json"))
          assert.equals(0, vim.fn.filereadable(other_dir .. "/files.json"))

          local loaded = db.get_weights("files")
          assert.are.same(test_data, loaded)
        end)
      end)
    end)

    it("should accept directory path as weights_path for backward compat", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        local test_data = { classic = { match = 77 } }
        db.save_weights("files", test_data)

        local loaded = db.get_weights("files")
        assert.are.same(test_data, loaded)
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.json"))
      end)
    end)
  end)

  describe("reset_cache", function()
    it("should allow re-initialization with different path", function()
      helpers.with_temp_db(function(temp_dir1)
        helpers.with_temp_db(function(temp_dir2)
          -- First, use temp_dir1
          neural_open.setup({ weights_path = temp_dir1 })
          db.reset_cache()
          db.save_weights("files", { location = "dir1" })

          -- Switch to temp_dir2
          neural_open.setup({ weights_path = temp_dir2 })
          db.reset_cache()
          db.save_weights("files", { location = "dir2" })

          -- Verify each dir has its own data
          db.reset_cache()
          neural_open.setup({ weights_path = temp_dir1 })
          db.reset_cache()
          local data1 = db.get_weights("files")
          assert.equals("dir1", data1.location)

          neural_open.setup({ weights_path = temp_dir2 })
          db.reset_cache()
          local data2 = db.get_weights("files")
          assert.equals("dir2", data2.location)
        end)
      end)
    end)
  end)

  describe("tracking file separation", function()
    it("should create separate tracking file on save_tracking", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        db.save_tracking("files", { recency_list = { "/a.lua", "/b.lua" } })

        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.tracking.json"))
        assert.equals(0, vim.fn.filereadable(temp_dir .. "/files.json"))
      end)
    end)

    it("should migrate tracking keys from weight file on first get_tracking", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        -- Create a files.json with both weight and tracking keys
        local combined = {
          nn = { weights = { { 1, 2, 3 } } },
          recency_list = { "/a.lua", "/b.lua" },
          transition_frecency = { ["/a.lua"] = { ["/b.lua"] = 1000000000 } },
        }
        local f = io.open(temp_dir .. "/files.json", "w")
        f:write(vim.json.encode(combined))
        f:close()

        local tracking = db.get_tracking("files")

        -- Tracking data should contain migrated keys
        assert.is_not_nil(tracking.recency_list)
        assert.equals(2, #tracking.recency_list)
        assert.is_not_nil(tracking.transition_frecency)

        -- Weight file should only have nn key
        local weights = db.get_weights("files")
        assert.is_not_nil(weights.nn)
        assert.is_nil(weights.recency_list)
        assert.is_nil(weights.transition_frecency)

        -- Tracking file should exist
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.tracking.json"))
      end)
    end)

    it("should migrate item_tracking key from weight file", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        local combined = {
          nn = { weights = { { 1, 2 } } },
          item_tracking = { frecency = { build = 1000000000 } },
        }
        local f = io.open(temp_dir .. "/picker.json", "w")
        f:write(vim.json.encode(combined))
        f:close()

        local tracking = db.get_tracking("picker")

        assert.is_not_nil(tracking.item_tracking)
        assert.is_not_nil(tracking.item_tracking.frecency)

        local weights = db.get_weights("picker")
        assert.is_not_nil(weights.nn)
        assert.is_nil(weights.item_tracking)
      end)
    end)

    it("should handle already-migrated state (idempotent)", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        -- Create both files already separated
        local weights_data = { nn = { weights = { 1, 2 } } }
        local tracking_data = { recency_list = { "/a.lua" } }

        local wf = io.open(temp_dir .. "/files.json", "w")
        wf:write(vim.json.encode(weights_data))
        wf:close()

        local tf = io.open(temp_dir .. "/files.tracking.json", "w")
        tf:write(vim.json.encode(tracking_data))
        tf:close()

        local tracking = db.get_tracking("files")
        assert.same({ "/a.lua" }, tracking.recency_list)

        -- Weight file should be unchanged
        local weights = db.get_weights("files")
        assert.is_not_nil(weights.nn)
        assert.is_nil(weights.recency_list)
      end)
    end)

    it("should handle fresh install (no files exist)", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        local tracking = db.get_tracking("newpicker")
        assert.same({}, tracking)
      end)
    end)

    it("should migrate legacy transition_history key", function()
      helpers.with_temp_db(function(temp_dir)
        neural_open.setup({ weights_path = temp_dir })
        db.reset_cache()

        local combined = {
          transition_history = { { from = "/a.lua", to = "/b.lua", timestamp = 123 } },
          nn = { weights = { 1 } },
        }
        local f = io.open(temp_dir .. "/files.json", "w")
        f:write(vim.json.encode(combined))
        f:close()

        local tracking = db.get_tracking("files")

        -- transition_history is just deleted, not migrated to tracking
        assert.is_nil(tracking.transition_history)

        -- Weight file should no longer have transition_history
        local weights = db.get_weights("files")
        assert.is_nil(weights.transition_history)
        assert.is_not_nil(weights.nn)

        -- Tracking file should exist (migration wrote it, even if empty)
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.tracking.json"))
      end)
    end)
  end)
end)
