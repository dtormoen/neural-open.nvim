describe("Per-picker state isolation", function()
  local helpers = require("tests.helpers")

  -- Build NN config for a given architecture with test-friendly defaults.
  -- Automatically sets dropout_rates to match the number of hidden layers.
  local function nn_config(overrides)
    overrides = overrides or {}
    local arch = overrides.architecture or { 11, 4, 1 }
    local num_hidden = #arch - 2
    local dropout = {}
    for _ = 1, num_hidden do
      dropout[#dropout + 1] = 0
    end
    local config = helpers.create_algorithm_config(
      "nn",
      vim.tbl_extend("force", {
        architecture = arch,
        optimizer = "sgd",
        learning_rate = 0.1,
        batch_size = 4,
        history_size = 10,
        match_dropout = 0,
        warmup_steps = 0,
        dropout_rates = dropout,
      }, overrides)
    )
    return config.algorithm_config.nn
  end

  describe("NN architecture isolation", function()
    local nn

    before_each(function()
      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.algorithms.nn_core"] = nil

      -- Mock weights module to prevent disk I/O
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return {}
        end,
        save_weights = function() end,
      }

      nn = require("neural-open.algorithms.nn")
    end)

    it("two instances with different input sizes coexist without corruption", function()
      -- File picker: 11 features
      local files_cfg = nn_config({ architecture = { 11, 4, 1 } })
      files_cfg.picker_name = "files"
      local files_instance = nn.create_instance(files_cfg)
      files_instance.load_weights()

      -- Item picker: 8 features
      local recipes_cfg = nn_config({ architecture = { 8, 4, 1 } })
      recipes_cfg.picker_name = "recipes"
      local recipes_instance = nn.create_instance(recipes_cfg)
      recipes_instance.load_weights()

      -- Score with 11-element input on files instance
      local files_buf = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }
      local files_score_1 = files_instance.calculate_score(files_buf)
      assert.is_number(files_score_1)

      -- Score with 8-element input on recipes instance
      local recipes_buf = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }
      local recipes_score = recipes_instance.calculate_score(recipes_buf)
      assert.is_number(recipes_score)

      -- Score files instance again — must not be corrupted by recipes instance
      local files_score_2 = files_instance.calculate_score(files_buf)
      assert.is_number(files_score_2)
      assert.are.equal(files_score_1, files_score_2)
    end)

    it("training one instance does not alter scores of the other", function()
      -- Create two instances with different architectures
      local files_cfg = nn_config({ architecture = { 11, 4, 1 } })
      files_cfg.picker_name = "files_train"
      local files_instance = nn.create_instance(files_cfg)
      files_instance.load_weights()

      local recipes_cfg = nn_config({ architecture = { 8, 4, 1 } })
      recipes_cfg.picker_name = "recipes_train"
      local recipes_instance = nn.create_instance(recipes_cfg)
      recipes_instance.load_weights()

      -- Capture baseline score for recipes
      local recipes_buf = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 }
      local recipes_score_before = recipes_instance.calculate_score(recipes_buf)

      -- Train the files instance
      local selected = {
        file = "selected.lua",
        nos = {
          input_buf = { 0.9, 0.0, 0.3, 1.0, 0.0, 0.5, 1.0, 0.4, 0.2, 0.0, 1.0 },
          normalized_path = "selected.lua",
        },
        neural_rank = 3,
      }
      local ranked = {
        {
          file = "rank1.lua",
          nos = {
            input_buf = { 0.5, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 1.0 },
            normalized_path = "rank1.lua",
          },
        },
        {
          file = "rank2.lua",
          nos = {
            input_buf = { 0.6, 0.0, 0.2, 0.0, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 1.0 },
            normalized_path = "rank2.lua",
          },
        },
        selected,
      }
      files_instance.update_weights(selected, ranked)

      -- Recipes score must be unchanged
      local recipes_score_after = recipes_instance.calculate_score(recipes_buf)
      assert.are.equal(recipes_score_before, recipes_score_after)
    end)
  end)

  describe("Classic weight isolation", function()
    local classic

    before_each(function()
      package.loaded["neural-open.algorithms.classic"] = nil

      local default_config = helpers.get_default_config()
      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return vim.deepcopy(default_config.algorithm_config.classic.default_weights)
        end,
        get_default_weights = function()
          return vim.deepcopy(default_config.algorithm_config.classic.default_weights)
        end,
        save_weights = function() end,
      }

      classic = require("neural-open.algorithms.classic")
    end)

    it("two instances with different picker names score independently", function()
      local cfg_a = helpers.create_algorithm_config("classic")
      cfg_a.algorithm_config.classic.picker_name = "picker_a"
      local instance_a = classic.create_instance(cfg_a.algorithm_config.classic)
      instance_a.load_weights()

      local cfg_b = helpers.create_algorithm_config("classic")
      cfg_b.algorithm_config.classic.picker_name = "picker_b"
      local instance_b = classic.create_instance(cfg_b.algorithm_config.classic)
      instance_b.load_weights()

      -- Both should produce identical scores from identical defaults
      local input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0.2, 0, 0, 1 }
      local score_a = instance_a.calculate_score(input_buf)
      local score_b = instance_b.calculate_score(input_buf)
      assert.are.equal(score_a, score_b)
    end)

    it("modifying one instance's weights does not affect the other", function()
      local cfg_a = helpers.create_algorithm_config("classic")
      cfg_a.algorithm_config.classic.picker_name = "modify_a"
      local instance_a = classic.create_instance(cfg_a.algorithm_config.classic)
      instance_a.load_weights()

      local cfg_b = helpers.create_algorithm_config("classic")
      cfg_b.algorithm_config.classic.picker_name = "modify_b"
      local instance_b = classic.create_instance(cfg_b.algorithm_config.classic)
      instance_b.load_weights()

      -- Capture baseline score for instance B
      local input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0.2, 0, 0, 1 }
      local score_b_before = instance_b.calculate_score(input_buf)

      -- Trigger weight learning on instance A (rank 2 selection)
      local selected = {
        neural_rank = 2,
        nos = {
          input_buf = { 0.2, 0, 0.8, 0, 0, 0.7, 0, 0.6, 0, 0, 1 },
        },
      }
      local higher_ranked = {
        {
          neural_rank = 1,
          nos = {
            input_buf = { 0.9, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 1 },
          },
        },
        selected,
      }
      instance_a.update_weights(selected, higher_ranked)

      -- Instance B score must be unchanged
      local score_b_after = instance_b.calculate_score(input_buf)
      assert.are.equal(score_b_before, score_b_after)
    end)
  end)

  describe("Weight save isolation", function()
    local save_calls

    before_each(function()
      save_calls = {}

      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.algorithms.nn_core"] = nil
      package.loaded["neural-open.algorithms.classic"] = nil

      package.loaded["neural-open.weights"] = {
        get_weights = function(algo_name)
          if algo_name == "classic" then
            local default_config = helpers.get_default_config()
            return vim.deepcopy(default_config.algorithm_config.classic.default_weights)
          end
          return {}
        end,
        get_default_weights = function(algo_name)
          if algo_name == "classic" then
            local default_config = helpers.get_default_config()
            return vim.deepcopy(default_config.algorithm_config.classic.default_weights)
          end
          return {}
        end,
        save_weights = function(algo_name, weights, latency_ctx, picker_name)
          table.insert(save_calls, {
            algo_name = algo_name,
            picker_name = picker_name,
          })
        end,
      }
    end)

    it("NN update_weights saves to the correct picker_name", function()
      local nn = require("neural-open.algorithms.nn")

      local files_cfg = nn_config({ architecture = { 11, 4, 1 } })
      files_cfg.picker_name = "files"
      local files_instance = nn.create_instance(files_cfg)
      files_instance.load_weights()

      local recipes_cfg = nn_config({ architecture = { 8, 4, 1 } })
      recipes_cfg.picker_name = "recipes"
      local recipes_instance = nn.create_instance(recipes_cfg)
      recipes_instance.load_weights()

      -- Train files instance
      local selected_11 = {
        file = "selected.lua",
        nos = {
          input_buf = { 0.9, 0.0, 0.3, 1.0, 0.0, 0.5, 1.0, 0.4, 0.2, 0.0, 1.0 },
          normalized_path = "selected.lua",
        },
        neural_rank = 2,
      }
      local ranked_11 = {
        {
          file = "other.lua",
          nos = {
            input_buf = { 0.5, 0.0, 0.1, 0.0, 0.0, 0.3, 0.0, 0.1, 0.0, 0.0, 1.0 },
            normalized_path = "other.lua",
          },
        },
        selected_11,
      }
      files_instance.update_weights(selected_11, ranked_11)

      -- Train recipes instance
      local selected_8 = {
        file = "recipe_a",
        nos = {
          input_buf = { 0.9, 0.3, 0.2, 0.4, 0.3, 0.5, 1.0, 0.2 },
          normalized_path = "recipe_a",
        },
        neural_rank = 2,
      }
      local ranked_8 = {
        {
          file = "recipe_b",
          nos = {
            input_buf = { 0.5, 0.1, 0.1, 0.1, 0.1, 0.3, 1.0, 0.1 },
            normalized_path = "recipe_b",
          },
        },
        selected_8,
      }
      recipes_instance.update_weights(selected_8, ranked_8)

      -- Verify each save targeted the correct picker
      local files_saves = vim.tbl_filter(function(c)
        return c.picker_name == "files"
      end, save_calls)
      local recipes_saves = vim.tbl_filter(function(c)
        return c.picker_name == "recipes"
      end, save_calls)

      assert.is_true(#files_saves >= 1, "Expected at least one save for 'files' picker")
      assert.is_true(#recipes_saves >= 1, "Expected at least one save for 'recipes' picker")
      assert.equals(#files_saves + #recipes_saves, #save_calls)

      -- All saves should be for "nn" algorithm
      for _, call in ipairs(save_calls) do
        assert.are.equal("nn", call.algo_name)
      end
    end)

    it("Classic update_weights saves to the correct picker_name", function()
      local classic = require("neural-open.algorithms.classic")

      local cfg_a = helpers.create_algorithm_config("classic")
      cfg_a.algorithm_config.classic.picker_name = "picker_alpha"
      local instance_a = classic.create_instance(cfg_a.algorithm_config.classic)
      instance_a.load_weights()

      local cfg_b = helpers.create_algorithm_config("classic")
      cfg_b.algorithm_config.classic.picker_name = "picker_beta"
      local instance_b = classic.create_instance(cfg_b.algorithm_config.classic)
      instance_b.load_weights()

      -- Trigger learning on instance A
      local selected = {
        neural_rank = 2,
        nos = {
          input_buf = { 0.2, 0, 0.8, 0, 0, 0.7, 0, 0.6, 0, 0, 1 },
        },
      }
      local ranked = {
        {
          nos = {
            input_buf = { 0.9, 0, 0.1, 0, 0, 0.1, 0, 0.1, 0, 0, 1 },
          },
        },
        selected,
      }
      instance_a.update_weights(selected, ranked)

      -- Trigger learning on instance B
      instance_b.update_weights(selected, ranked)

      -- Verify saves targeted the correct pickers
      local alpha_saves = vim.tbl_filter(function(c)
        return c.picker_name == "picker_alpha"
      end, save_calls)
      local beta_saves = vim.tbl_filter(function(c)
        return c.picker_name == "picker_beta"
      end, save_calls)

      assert.is_true(#alpha_saves >= 1, "Expected at least one save for 'picker_alpha'")
      assert.is_true(#beta_saves >= 1, "Expected at least one save for 'picker_beta'")
      assert.equals(#alpha_saves + #beta_saves, #save_calls)

      -- All saves should be for "classic" algorithm
      for _, call in ipairs(save_calls) do
        assert.are.equal("classic", call.algo_name)
      end
    end)
  end)
end)
