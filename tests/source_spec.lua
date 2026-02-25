describe("neural-open.source", function()
  local source
  local scorer

  before_each(function()
    -- Reset modules
    package.loaded["neural-open.source"] = nil
    package.loaded["neural-open.scorer"] = nil
    package.loaded["neural-open.algorithms.registry"] = nil

    source = require("neural-open.source")
    scorer = require("neural-open.scorer")

    -- Mock config for registry
    local helpers = require("tests.helpers")
    package.loaded["neural-open"] = {
      config = helpers.get_default_config(),
    }
  end)

  describe("unique transform", function()
    it("should deduplicate files with the same normalized path", function()
      local config = {}
      local transform = source.create_neural_transform(config, scorer, {})

      local ctx = { meta = {} }

      -- Create a temp file to test with
      local temp_dir = vim.fn.tempname()
      vim.fn.mkdir(temp_dir, "p")
      local test_file = temp_dir .. "/test.lua"
      vim.fn.writefile({ "-- test" }, test_file)

      -- First item with absolute path
      local item1 = { file = test_file }
      local result1 = transform(item1, ctx)
      assert.is_not_false(result1)
      assert.is_not_nil(result1)
      assert.is.truthy(result1.nos) -- Should have nos field
      assert.is.truthy(result1.nos.normalized_path) -- Should have normalized_path set

      -- Second item with the same absolute path should be filtered
      local item2 = { file = test_file }
      local result2 = transform(item2, ctx)
      assert.is_false(result2) -- Should be filtered out as duplicate

      -- Test that same file with different representations gets deduplicated
      -- Since test_file is in temp dir, not relative to cwd, let's test differently
      -- We'll just verify that the same absolute path gets deduplicated
      local item3 = { file = test_file }
      local result3 = transform(item3, ctx)
      assert.is_false(result3) -- Should be filtered out as duplicate

      -- Test with a file that has cwd but is already absolute
      local item4 = { file = test_file, cwd = "/some/other/path" }
      local result4 = transform(item4, ctx)
      assert.is_false(result4) -- Should still be filtered as it's the same absolute file

      -- Cleanup
      vim.fn.delete(temp_dir, "rf")
    end)

    it("should not deduplicate different files", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})

      local ctx = { meta = {} }

      -- First file
      local item1 = { file = "lua/test1.lua" }
      local result1 = transform(item1, ctx)
      assert.is_not_false(result1)
      assert.is_not_nil(result1)

      -- Different file
      local item2 = { file = "lua/test2.lua" }
      local result2 = transform(item2, ctx)
      assert.is_not_false(result2) -- Should NOT be filtered out
      assert.is_not_nil(result2)

      -- Another different file
      local item3 = { file = "tests/test.lua" }
      local result3 = transform(item3, ctx)
      assert.is_not_false(result3) -- Should NOT be filtered out
      assert.is_not_nil(result3)
    end)

    it("should handle files with cwd prefix correctly", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})

      local ctx = { meta = {} }
      local cwd = vim.fn.getcwd()

      -- Item with cwd
      local item1 = { file = "test.lua", cwd = cwd }
      local result1 = transform(item1, ctx)
      assert.is_not_false(result1)
      assert.is_not_nil(result1)

      -- Same file without cwd but with absolute path
      local abs_path = cwd .. "/test.lua"
      local item2 = { file = abs_path }
      local result2 = transform(item2, ctx)
      assert.is_false(result2) -- Should be filtered out as duplicate
    end)

    it("should skip items without file field", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})

      local ctx = { meta = {} }

      -- Item without file field
      local item = { text = "some text" }
      local result = transform(item, ctx)
      assert.are.equal(item, result) -- Should return the item unchanged
      assert.is_nil(result.nos) -- Should not add nos field
    end)

    it("should maintain separate done tracking per context", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})

      -- First context
      local ctx1 = { meta = {} }
      local item1 = { file = "test.lua" }
      local result1 = transform(item1, ctx1)
      assert.is_not_false(result1)

      -- Same file in first context should be filtered
      local result2 = transform(item1, ctx1)
      assert.is_false(result2)

      -- New context
      local ctx2 = { meta = {} }
      -- Same file in new context should NOT be filtered
      local result3 = transform(item1, ctx2)
      assert.is_not_false(result3)
    end)

    it("should set item._path to normalized absolute path when relative with cwd", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})
      local ctx = { meta = {} }

      -- Simulate git_files source providing relative path with different cwd (git root)
      -- This is the bug scenario: CWD is /my_project/src, git root is /my_project
      local item = { file = "readme.md", cwd = "/my_project" }
      local result = transform(item, ctx)

      assert.is_not_false(result)
      -- item._path should be the normalized absolute path (this is what Snacks.picker.util.path() uses)
      assert.equals("/my_project/readme.md", result._path)
      -- item.file and item.cwd should be preserved (for display formatting)
      assert.equals("readme.md", result.file)
      assert.equals("/my_project", result.cwd)
      -- Should match the normalized_path in nos
      assert.equals(result._path, result.nos.normalized_path)
    end)

    it("should set item._path for absolute paths", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})
      local ctx = { meta = {} }

      -- Item already has absolute path
      local item = { file = "/absolute/path/to/file.lua" }
      local result = transform(item, ctx)

      assert.is_not_false(result)
      -- item._path should be the absolute path
      assert.equals("/absolute/path/to/file.lua", result._path)
      assert.equals(result._path, result.nos.normalized_path)
    end)
  end)

  describe("proximity scoring with normalized paths", function()
    it("should calculate proximity correctly using normalized paths", function()
      local config = { special_files = {} }
      local transform = source.create_neural_transform(config, scorer, {})

      -- Create context with current file
      local ctx = {
        meta = {
          nos_ctx = {
            cwd = "/home/user/project",
            current_file = "/home/user/project/src/main.lua",
          },
        },
      }

      -- Test file in same directory (should have high proximity)
      local item1 = { file = "src/helper.lua", cwd = "/home/user/project" }
      local result1 = transform(item1, ctx)
      assert.is_not_false(result1)
      assert.is.truthy(result1.nos)
      assert.is.truthy(result1.nos.raw_features)
      -- Should have proximity score since it's in the same directory
      assert.is.truthy(result1.nos.raw_features.proximity)
      assert.is_true(result1.nos.raw_features.proximity > 0.9) -- Same directory should be 1.0

      -- Test file with absolute path in same directory
      local item2 = { file = "/home/user/project/src/utils.lua" }
      local result2 = transform(item2, ctx)
      assert.is_not_false(result2)
      assert.is.truthy(result2.nos)
      assert.is.truthy(result2.nos.raw_features)
      -- Should also have proximity score
      assert.is.truthy(result2.nos.raw_features.proximity)
      assert.is_true(result2.nos.raw_features.proximity > 0.9) -- Same directory should be 1.0
    end)
  end)

  describe("context capture", function()
    it("should capture buffer context", function()
      local ctx = { meta = {} }

      -- Mock recent module
      package.loaded["neural-open.recent"] = {
        get_recency_map = function(limit)
          return {
            ["/path/to/file1.lua"] = { recent_rank = 1 },
            ["/path/to/file2.lua"] = { recent_rank = 2 },
          }
        end,
      }

      source.capture_context(ctx)

      assert.is_not_nil(ctx.meta.nos_ctx)
      assert.is_not_nil(ctx.meta.nos_ctx.recent_files)
      assert.is_not_nil(ctx.meta.nos_ctx.cwd)
      assert.is_not_nil(ctx.meta.nos_ctx.current_file)
      assert.is_not_nil(ctx.meta.nos_ctx.alternate_buf)
    end)
  end)
end)
