-- Test spec for config helper functions
describe("config helpers", function()
  local helpers = require("tests.helpers")

  -- WORKAROUND: Some other tests (nn_spec.lua, nn_hinge_spec.lua) replace vim.tbl_deep_extend
  -- with vim.tbl_extend globally, which breaks deep merging for all subsequent tests.
  -- We detect and fix this pollution here.
  if _G.vim.tbl_deep_extend == _G.vim.tbl_extend then
    -- Load the real implementation from vim.shared
    _G.vim.tbl_deep_extend = require("vim.shared").tbl_deep_extend
  end

  describe("get_default_config", function()
    it("should return a deep copy of default config", function()
      local config1 = helpers.get_default_config()
      local config2 = helpers.get_default_config()

      -- Verify they have the same values
      assert.equals(config1.algorithm, config2.algorithm)
      assert.equals(config1.weights_path, config2.weights_path)

      -- Verify they are separate copies (not the same table)
      assert.is_not.equal(config1, config2)

      -- Verify modifying one doesn't affect the other
      config1.algorithm = "different"
      assert.is_not.equals(config1.algorithm, config2.algorithm)
    end)

    it("should return config with expected structure", function()
      local config = helpers.get_default_config()

      -- Verify top-level fields
      assert.is_not_nil(config.algorithm)
      assert.is_not_nil(config.algorithm_config)
      assert.is_not_nil(config.weights_path)
      assert.is_not_nil(config.debug)
      assert.is_not_nil(config.special_files)

      -- Verify algorithm configs exist
      assert.is_not_nil(config.algorithm_config.classic)
      assert.is_not_nil(config.algorithm_config.naive)
      assert.is_not_nil(config.algorithm_config.nn)

      -- Verify classic config has expected fields
      assert.is_not_nil(config.algorithm_config.classic.learning_rate)
      assert.is_not_nil(config.algorithm_config.classic.default_weights)
      assert.is_not_nil(config.algorithm_config.classic.default_weights.match)
    end)
  end)

  describe("create_test_config", function()
    it("should merge top-level overrides correctly", function()
      local config = helpers.create_test_config({
        algorithm = "nn",
        debug = { preview = true },
      })

      -- Verify overrides are applied
      assert.equals("nn", config.algorithm)
      assert.equals(true, config.debug.preview)

      -- Verify other defaults are preserved
      local defaults = helpers.get_default_config()
      assert.equals(defaults.weights_path, config.weights_path)
      assert.equals(defaults.algorithm_config.classic.learning_rate, config.algorithm_config.classic.learning_rate)
    end)

    it("should work with nil overrides", function()
      local config = helpers.create_test_config(nil)

      -- Should return a config identical to defaults
      local defaults = helpers.get_default_config()
      assert.equals(defaults.algorithm, config.algorithm)
      assert.equals(defaults.weights_path, config.weights_path)
    end)

    it("should work with empty overrides", function()
      local config = helpers.create_test_config({})

      -- Should return a config identical to defaults
      local defaults = helpers.get_default_config()
      assert.equals(defaults.algorithm, config.algorithm)
      assert.equals(defaults.weights_path, config.weights_path)
    end)

    it("should deep merge nested config", function()
      -- Get defaults first to capture the original values
      local defaults = helpers.get_default_config()
      local expected_match = defaults.algorithm_config.classic.default_weights.match

      local config = helpers.create_test_config({
        algorithm_config = {
          classic = {
            learning_rate = 0.8,
          },
        },
      })

      -- Verify override is applied
      assert.equals(0.8, config.algorithm_config.classic.learning_rate)

      -- Verify other classic config is preserved (default_weights should still be there)
      assert.is_not_nil(config.algorithm_config.classic.default_weights)
      assert.equals(expected_match, config.algorithm_config.classic.default_weights.match)
    end)
  end)

  describe("create_algorithm_config", function()
    it("should merge algorithm-specific overrides for classic", function()
      local config = helpers.create_algorithm_config("classic", {
        learning_rate = 0.8,
      })

      -- Verify override is applied
      assert.equals(0.8, config.algorithm_config.classic.learning_rate)

      -- Verify other defaults are preserved
      local defaults = helpers.get_default_config()
      assert.equals(defaults.algorithm, config.algorithm)
      assert.equals(
        defaults.algorithm_config.classic.default_weights.match,
        config.algorithm_config.classic.default_weights.match
      )
    end)

    it("should merge algorithm-specific overrides for nn", function()
      local config = helpers.create_algorithm_config("nn", {
        architecture = { 10, 4, 1 },
        batch_size = 8,
      })

      -- Verify overrides are applied
      assert.same({ 10, 4, 1 }, config.algorithm_config.nn.architecture)
      assert.equals(8, config.algorithm_config.nn.batch_size)

      -- Verify other nn defaults are preserved
      local defaults = helpers.get_default_config()
      assert.equals(defaults.algorithm_config.nn.learning_rate, config.algorithm_config.nn.learning_rate)
      assert.equals(defaults.algorithm_config.nn.optimizer, config.algorithm_config.nn.optimizer)
    end)

    it("should work with nil overrides", function()
      local config = helpers.create_algorithm_config("classic", nil)

      -- Should return a config identical to defaults
      local defaults = helpers.get_default_config()
      assert.equals(defaults.algorithm_config.classic.learning_rate, config.algorithm_config.classic.learning_rate)
      assert.equals(
        defaults.algorithm_config.classic.default_weights.match,
        config.algorithm_config.classic.default_weights.match
      )
    end)

    it("should deep merge weights", function()
      local defaults = helpers.get_default_config()
      local original_frecency = defaults.algorithm_config.classic.default_weights.frecency
      local original_proximity = defaults.algorithm_config.classic.default_weights.proximity

      local config = helpers.create_algorithm_config("classic", {
        default_weights = {
          match = 200,
        },
      })

      -- Verify override is applied
      assert.equals(200, config.algorithm_config.classic.default_weights.match)

      -- Verify other weights are preserved
      assert.is_not_nil(config.algorithm_config.classic.default_weights.frecency)
      assert.is_not_nil(config.algorithm_config.classic.default_weights.proximity)
      assert.equals(original_frecency, config.algorithm_config.classic.default_weights.frecency)
      assert.equals(original_proximity, config.algorithm_config.classic.default_weights.proximity)
    end)

    it("should not affect other algorithms", function()
      local defaults = helpers.get_default_config()
      local original_nn_lr = defaults.algorithm_config.nn.learning_rate

      local config = helpers.create_algorithm_config("classic", {
        learning_rate = 0.9,
      })

      -- Verify classic was modified
      assert.equals(0.9, config.algorithm_config.classic.learning_rate)

      -- Verify nn was not modified
      assert.equals(original_nn_lr, config.algorithm_config.nn.learning_rate)

      -- Verify naive config exists (it's an empty table)
      assert.is_not_nil(config.algorithm_config.naive)
    end)
  end)

  describe("isolation between calls", function()
    it("should return independent copies", function()
      local defaults = helpers.get_default_config()
      local original_match = defaults.algorithm_config.classic.default_weights.match

      local config1 = helpers.create_algorithm_config("classic", {
        learning_rate = 0.7,
      })

      local config2 = helpers.create_algorithm_config("classic", {
        learning_rate = 0.9,
      })

      -- Verify they have different values
      assert.equals(0.7, config1.algorithm_config.classic.learning_rate)
      assert.equals(0.9, config2.algorithm_config.classic.learning_rate)

      -- Verify they are separate objects
      assert.is_not.equal(config1, config2)

      -- Modify one and verify the other is unaffected
      config1.algorithm_config.classic.default_weights.match = 999
      assert.equals(original_match, config2.algorithm_config.classic.default_weights.match)
      assert.is_not.equals(999, config2.algorithm_config.classic.default_weights.match)
    end)
  end)
end)
