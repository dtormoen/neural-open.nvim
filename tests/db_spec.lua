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

  describe("backward compatibility", function()
    it("should use parent directory when weights_path is a .json file", function()
      helpers.with_temp_db(function(temp_dir)
        -- Configure with a legacy .json path
        local json_path = temp_dir .. "/weights.json"
        neural_open.setup({ weights_path = json_path })
        db.reset_cache()

        -- Save and load should work using the parent directory
        local test_data = { classic = { match = 99 } }
        db.save_weights("files", test_data)

        local loaded = db.get_weights("files")
        assert.are.same(test_data, loaded)

        -- Verify the file was created as files.json in the parent dir
        assert.equals(1, vim.fn.filereadable(temp_dir .. "/files.json"))
      end)
    end)

    it("should use parent directory when weights_path is an existing file", function()
      helpers.with_temp_db(function(temp_dir)
        -- Create an existing file at the configured path (non-.json extension)
        local existing_path = temp_dir .. "/my_weights"
        local f = io.open(existing_path, "w")
        f:write("{}")
        f:close()

        neural_open.setup({ weights_path = existing_path })
        db.reset_cache()

        local test_data = { classic = { match = 77 } }
        db.save_weights("files", test_data)

        local loaded = db.get_weights("files")
        assert.are.same(test_data, loaded)
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
end)
