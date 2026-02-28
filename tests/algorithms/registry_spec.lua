describe("algorithm registry", function()
  local registry

  --- Helper: returns a create_instance function for nn/classic mocks.
  --- Each call returns a fresh instance table with the Algorithm interface.
  local function make_create_instance_mock(algo_name)
    return function(config)
      return {
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
          return algo_name
        end,
        init = function() end,
        load_weights = function() end,
        _config = config,
      }
    end
  end

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
    -- Mock classic and nn with create_instance (production path).
    -- No module-level Algorithm methods — only create_instance.
    package.loaded["neural-open.algorithms.classic"] = {
      create_instance = make_create_instance_mock("Classic"),
    }
    package.loaded["neural-open.algorithms.nn"] = {
      create_instance = make_create_instance_mock("Neural Network"),
    }
    -- Naive is stateless: module-level methods, init, no create_instance
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

    it("should pass config to create_instance", function()
      local create_called = false
      local received_config = nil

      package.loaded["neural-open.algorithms.classic"].create_instance = function(cfg)
        create_called = true
        received_config = cfg
        return make_create_instance_mock("Classic")(cfg)
      end

      package.loaded["neural-open"].config.algorithm = "classic"
      package.loaded["neural-open"].config.algorithm_config = {
        classic = {
          learning_rate = 0.5,
        },
      }

      registry.get_algorithm()

      assert.is_true(create_called)
      assert.are.same({ learning_rate = 0.5 }, received_config)
    end)

    it("should pass empty config to create_instance when algorithm config not provided", function()
      local create_called = false
      local received_config = nil

      package.loaded["neural-open.algorithms.classic"].create_instance = function(cfg)
        create_called = true
        received_config = cfg
        return make_create_instance_mock("Classic")(cfg)
      end

      package.loaded["neural-open"].config.algorithm = "classic"
      package.loaded["neural-open"].config.algorithm_config = nil

      registry.get_algorithm()

      assert.is_true(create_called)
      assert.are.same({}, received_config)
    end)

    it("should use create_instance when available", function()
      local classic_module = package.loaded["neural-open.algorithms.classic"]
      local algorithm = registry.get_algorithm()

      -- Returned algorithm is the instance, not the module itself
      assert.is_not_equal(classic_module, algorithm)
      assert.are.equal("Classic", algorithm.get_name())
    end)

    it("should return the module itself for naive (no create_instance)", function()
      package.loaded["neural-open"].config.algorithm = "naive"
      local naive_module = package.loaded["neural-open.algorithms.naive"]
      local algorithm = registry.get_algorithm()

      -- Naive has no create_instance, so the module itself is returned
      assert.are.equal(naive_module, algorithm)
    end)
  end)

  describe("get_algorithm_for_picker", function()
    it("injects picker_name into the config passed to create_instance", function()
      local received_config = nil
      package.loaded["neural-open.algorithms.nn"].create_instance = function(cfg)
        received_config = cfg
        return make_create_instance_mock("Neural Network")(cfg)
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
      package.loaded["neural-open.algorithms.classic"].create_instance = function(cfg)
        cfg.mutated = true
        return make_create_instance_mock("Classic")(cfg)
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
      package.loaded["neural-open.algorithms.classic"].create_instance = function(cfg)
        received_config = cfg
        return make_create_instance_mock("Classic")(cfg)
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

    it("returns independent instances for different picker names", function()
      local algo_config = {
        nn = { architecture = { 7, 16, 8, 1 } },
      }
      local instance_a = registry.get_algorithm_for_picker("nn", algo_config, "picker_a")
      local instance_b = registry.get_algorithm_for_picker("nn", algo_config, "picker_b")

      -- Different table references
      assert.is_not_equal(instance_a, instance_b)
      -- Both have correct interface
      assert.are.equal("Neural Network", instance_a.get_name())
      assert.are.equal("Neural Network", instance_b.get_name())
      -- Each received its own picker_name
      assert.equals("picker_a", instance_a._config.picker_name)
      assert.equals("picker_b", instance_b._config.picker_name)
    end)

    it("uses create_instance when available, returning instance not module", function()
      local nn_module = package.loaded["neural-open.algorithms.nn"]
      local algo_config = {
        nn = { architecture = { 7, 16, 8, 1 } },
      }
      local instance = registry.get_algorithm_for_picker("nn", algo_config, "test")

      assert.is_not_equal(nn_module, instance)
      assert.are.equal("Neural Network", instance.get_name())
    end)
  end)
end)
