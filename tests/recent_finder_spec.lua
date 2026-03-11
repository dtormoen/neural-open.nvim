describe("recent_finder module", function()
  local helpers = require("tests.helpers")
  local recent_finder
  local mock_recent
  local temp_dir

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    temp_dir = helpers.create_temp_dir()

    -- Mock recent module
    mock_recent = {
      recency_map = {},
      get_recency_map = function()
        return vim.deepcopy(mock_recent.recency_map)
      end,
    }
    package.loaded["neural-open.recent"] = mock_recent

    -- Remove snacks frecency so it does not interfere by default
    package.loaded["snacks.picker.core.frecency"] = nil

    recent_finder = require("neural-open.recent_finder")
  end)

  after_each(function()
    helpers.clear_plugin_modules()
    package.loaded["neural-open.recent"] = nil
    package.loaded["snacks.picker.core.frecency"] = nil
    helpers.cleanup_temp_dir(temp_dir)
  end)

  --- Create a real file in the temp directory and return its path
  local function create_temp_file(name)
    local path = temp_dir .. "/" .. name
    local f = io.open(path, "w")
    f:write("")
    f:close()
    return path
  end

  describe("finder", function()
    it("returns items with correct structure from recent files", function()
      local file_a = create_temp_file("a.lua")
      local file_b = create_temp_file("b.lua")

      mock_recent.recency_map = {
        [file_a] = { recent_rank = 2 },
        [file_b] = { recent_rank = 1 },
      }

      local items = recent_finder.finder({}, {})

      assert.equals(2, #items)
      -- Most recent first (rank 1 before rank 2)
      assert.equals(file_b, items[1].file)
      assert.equals(file_b, items[1].text)
      assert.equals(file_a, items[2].file)
      assert.equals(file_a, items[2].text)
    end)

    it("filters out non-existent files", function()
      local file_a = create_temp_file("exists.lua")
      local file_missing = temp_dir .. "/missing.lua"

      mock_recent.recency_map = {
        [file_a] = { recent_rank = 1 },
        [file_missing] = { recent_rank = 2 },
      }

      local items = recent_finder.finder({}, {})

      assert.equals(1, #items)
      assert.equals(file_a, items[1].file)
    end)

    it("returns empty table when no recent files exist", function()
      mock_recent.recency_map = {}

      local items = recent_finder.finder({}, {})

      assert.same({}, items)
    end)

    it("handles snacks frecency not being available", function()
      -- snacks.picker.core.frecency is not loaded (pcall will fail)
      local file_a = create_temp_file("a.lua")

      mock_recent.recency_map = {
        [file_a] = { recent_rank = 1 },
      }

      local items = recent_finder.finder({}, {})

      assert.equals(1, #items)
      assert.equals(file_a, items[1].file)
    end)

    it("deduplicates between recent and frecent sources", function()
      local file_shared = create_temp_file("shared.lua")
      local file_only_recent = create_temp_file("only_recent.lua")
      local file_only_frecent = create_temp_file("only_frecent.lua")

      mock_recent.recency_map = {
        [file_shared] = { recent_rank = 1 },
        [file_only_recent] = { recent_rank = 2 },
      }

      -- Mock snacks frecency
      package.loaded["snacks.picker.core.frecency"] = {
        new = function()
          return {
            cache = {
              [file_shared] = 1000, -- same file as recent
              [file_only_frecent] = 900,
            },
            to_score = function(_, deadline)
              return deadline -- use deadline directly as score for simplicity
            end,
          }
        end,
      }

      -- Re-require to pick up the frecency mock
      package.loaded["neural-open.recent_finder"] = nil
      recent_finder = require("neural-open.recent_finder")

      local items = recent_finder.finder({}, {})

      -- Should have 3 unique files, not 4
      assert.equals(3, #items)

      -- Recent files come first
      assert.equals(file_shared, items[1].file)
      assert.equals(file_only_recent, items[2].file)
      -- Frecent-only file comes last
      assert.equals(file_only_frecent, items[3].file)
    end)

    it("orders frecent files by score descending", function()
      local file_low = create_temp_file("low.lua")
      local file_high = create_temp_file("high.lua")

      mock_recent.recency_map = {}

      package.loaded["snacks.picker.core.frecency"] = {
        new = function()
          return {
            cache = {
              [file_low] = 100,
              [file_high] = 900,
            },
            to_score = function(_, deadline)
              return deadline
            end,
          }
        end,
      }

      package.loaded["neural-open.recent_finder"] = nil
      recent_finder = require("neural-open.recent_finder")

      local items = recent_finder.finder({}, {})

      assert.equals(2, #items)
      assert.equals(file_high, items[1].file)
      assert.equals(file_low, items[2].file)
    end)

    it("skips frecent files with zero score", function()
      local file_active = create_temp_file("active.lua")
      local file_zero = create_temp_file("zero.lua")

      mock_recent.recency_map = {}

      package.loaded["snacks.picker.core.frecency"] = {
        new = function()
          return {
            cache = {
              [file_active] = 500,
              [file_zero] = 0,
            },
            to_score = function(_, deadline)
              return deadline
            end,
          }
        end,
      }

      package.loaded["neural-open.recent_finder"] = nil
      recent_finder = require("neural-open.recent_finder")

      local items = recent_finder.finder({}, {})

      assert.equals(1, #items)
      assert.equals(file_active, items[1].file)
    end)
  end)
end)
