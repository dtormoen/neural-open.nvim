describe("algorithm registry", function()
  local registry

  before_each(function()
    package.loaded["neural-open.algorithms.registry"] = nil
    -- Mock the main plugin to provide config
    package.loaded["neural-open"] = {
      config = {
        algorithm = "classic",
        algorithm_config = {
          classic = {},
          naive = {},
          nn = {},
        },
      },
    }
    -- Mock the algorithm modules to avoid loading real algorithms in tests
    package.loaded["neural-open.algorithms.classic"] = {
      calculate_score = function()
        return 0
      end,
      update_weights = function()
        return nil
      end,
      debug_view = function()
        return {}
      end,
      get_name = function()
        return "Classic"
      end,
      init = function() end,
      load_weights = function() end,
    }
    package.loaded["neural-open.algorithms.naive"] = {
      calculate_score = function()
        return 0
      end,
      update_weights = function()
        return nil
      end,
      debug_view = function()
        return {}
      end,
      get_name = function()
        return "Naive"
      end,
      init = function() end,
      load_weights = function() end,
    }
    package.loaded["neural-open.algorithms.nn"] = {
      calculate_score = function()
        return 0
      end,
      update_weights = function()
        return nil
      end,
      debug_view = function()
        return {}
      end,
      get_name = function()
        return "Neural Network"
      end,
      init = function() end,
      load_weights = function() end,
    }
    registry = require("neural-open.algorithms.registry")
  end)

  describe("get_algorithm", function()
    it("should return classic algorithm by default", function()
      local algorithm = registry.get_algorithm()

      assert.is_not_nil(algorithm)
      assert.is_function(algorithm.calculate_score)
      assert.is_function(algorithm.update_weights)
      assert.is_function(algorithm.debug_view)
      assert.is_function(algorithm.get_name)
      assert.is_function(algorithm.load_weights)
      assert.are.equal("Classic", algorithm.get_name())
    end)

    it("should return specified algorithm from config", function()
      package.loaded["neural-open"].config.algorithm = "naive"
      local algorithm = registry.get_algorithm()

      assert.is_not_nil(algorithm)
      assert.are.equal("Naive", algorithm.get_name())
    end)

    it("should return nn algorithm when specified", function()
      package.loaded["neural-open"].config.algorithm = "nn"
      local algorithm = registry.get_algorithm()

      assert.is_not_nil(algorithm)
      assert.are.equal("Neural Network", algorithm.get_name())
    end)

    it("should fallback to classic for invalid algorithm name", function()
      package.loaded["neural-open"].config.algorithm = "invalid_algo"
      local algorithm = registry.get_algorithm()

      assert.is_not_nil(algorithm)
      assert.are.equal("Classic", algorithm.get_name())
    end)

    it("should initialize algorithm with its config", function()
      local init_called = false
      local received_config = nil

      package.loaded["neural-open.algorithms.classic"].init = function(cfg)
        init_called = true
        received_config = cfg
      end

      package.loaded["neural-open"].config.algorithm = "classic"
      package.loaded["neural-open"].config.algorithm_config = {
        classic = {
          learning_rate = 0.5,
        },
      }

      registry.get_algorithm()

      assert.is_true(init_called)
      assert.are.same({ learning_rate = 0.5 }, received_config)
    end)

    it("should initialize with empty config if algorithm config not provided", function()
      local init_called = false
      local received_config = nil

      package.loaded["neural-open.algorithms.classic"].init = function(cfg)
        init_called = true
        received_config = cfg
      end

      package.loaded["neural-open"].config.algorithm = "classic"
      package.loaded["neural-open"].config.algorithm_config = nil

      registry.get_algorithm()

      assert.is_true(init_called)
      assert.are.same({}, received_config)
    end)

    it("should not fail when loading core plugin algorithms", function()
      -- Since we removed pcalls, the algorithms should always load
      -- This test verifies that the normal flow works
      package.loaded["neural-open"].config.algorithm = "classic"
      local algorithm = registry.get_algorithm()

      assert.is_not_nil(algorithm)
      assert.is_function(algorithm.calculate_score)
      assert.is_function(algorithm.update_weights)
      assert.is_function(algorithm.debug_view)
      assert.is_function(algorithm.get_name)
      assert.is_function(algorithm.load_weights)
      assert.are.equal("Classic", algorithm.get_name())
    end)
  end)

  describe("get_algorithm_for_picker", function()
    it("injects picker_name into the config passed to init", function()
      local received_config = nil
      package.loaded["neural-open.algorithms.nn"].init = function(cfg)
        received_config = cfg
      end

      local algo_config = {
        nn = { architecture = { 7, 16, 8, 1 }, optimizer = "adamw" },
      }
      registry.get_algorithm_for_picker("nn", algo_config, "just_recipes")

      assert.is_not_nil(received_config)
      assert.equals("just_recipes", received_config.picker_name)
      assert.same({ 7, 16, 8, 1 }, received_config.architecture)
    end)

    it("deep copies config to prevent mutation of the original", function()
      package.loaded["neural-open.algorithms.classic"].init = function(cfg)
        cfg.mutated = true
      end

      local algo_config = {
        classic = { learning_rate = 0.5 },
      }
      registry.get_algorithm_for_picker("classic", algo_config, "test")

      assert.is_nil(algo_config.classic.mutated)
    end)

    it("falls back to classic for invalid algorithm name", function()
      registry.get_algorithm_for_picker("invalid", { invalid = {} }, "test")
      -- Should not error; falls back to classic
    end)

    it("merges extra_config into the algorithm config", function()
      local received_config = nil
      package.loaded["neural-open.algorithms.classic"].init = function(cfg)
        received_config = cfg
      end

      local algo_config = {
        classic = { learning_rate = 0.6 },
      }
      local extra = { feature_names = { "match", "frecency" } }
      registry.get_algorithm_for_picker("classic", algo_config, "test", extra)

      assert.same({ "match", "frecency" }, received_config.feature_names)
      assert.equals("test", received_config.picker_name)
      assert.equals(0.6, received_config.learning_rate)
    end)
  end)
end)
