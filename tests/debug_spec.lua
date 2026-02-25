local helper = require("tests.helpers")
helper.setup()

describe("debug module", function()
  local debug
  local item_with_full_data
  local item_with_partial_data
  local item_without_nos
  local ctx

  before_each(function()
    local helpers = require("tests.helpers")

    -- Clear algorithm cache
    package.loaded["neural-open.algorithms.registry"] = nil
    package.loaded["neural-open.algorithms.classic"] = nil

    debug = require("neural-open.debug")

    -- Mock weights module FIRST (before loading algorithm)
    local default_weights = helpers.get_default_config().algorithm_config.classic.default_weights

    package.loaded["neural-open.weights"] = {
      get_weights = function()
        return vim.deepcopy(default_weights)
      end,
      get_default_weights = function()
        return vim.deepcopy(default_weights)
      end,
      calculate_components = function(normalized_features)
        local components = {}
        for feature_name, normalized_value in pairs(normalized_features) do
          if default_weights[feature_name] then
            components[feature_name] = normalized_value * default_weights[feature_name]
          end
        end
        return components
      end,
      simulate_weight_adjustments = function()
        return {
          compared_with = 1,
          changes = {
            match = { old = default_weights.match, new = default_weights.match + 5, delta = 5 },
            proximity = { old = default_weights.proximity, new = default_weights.proximity - 2, delta = -2 },
          },
        }
      end,
      save_weights = function() end, -- No-op for tests
    }

    -- Load classic algorithm with mocked weights
    local classic = require("neural-open.algorithms.classic")
    local config = helpers.get_default_config()
    classic.init(config.algorithm_config.classic)
    classic.load_weights()

    -- Create a comprehensive test item with all features
    -- input_buf order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition
    item_with_full_data = {
      file = "/test/file.lua",
      text = "file.lua",
      score = 150.5,
      nos = {
        neural_score = 125.25,
        raw_features = {
          match = 100.0,
          virtual_name = 50.0,
          open = 1.0,
          alt = 1.0,
          proximity = 0.8,
          project = 1.0,
          frecency = 15.0,
          recency = 3.0,
        },
        input_buf = { 0.8500, 0.4200, 0.9375, 1.0000, 1.0000, 0.8000, 1.0000, 0.5333, 0, 0, 1 },
        is_open_buffer = true,
        is_alternate = true,
        recent_rank = 2,
        ctx = {
          algorithm = classic, -- Add algorithm to context
        },
      },
    }

    -- Create item with partial data (zeros for some features)
    -- input_buf order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition
    item_with_partial_data = {
      file = "/test/another.lua",
      text = "another.lua",
      score = 80.0,
      nos = {
        neural_score = 75.0,
        raw_features = {
          match = 90.0,
          virtual_name = 0,
          open = 0,
          alt = 0,
          proximity = 0.5,
          project = 1.0,
          frecency = 0,
          recency = 0,
        },
        input_buf = { 0.7500, 0, 0, 0, 0, 0.5000, 1.0000, 0, 0, 0, 1 },
        ctx = {
          algorithm = classic, -- Add algorithm to context
        },
      },
    }

    -- Create item without nos field
    item_without_nos = {
      file = "/test/basic.lua",
      text = "basic.lua",
    }

    -- Mock context with preview
    ctx = {
      item = nil,
      preview = {
        lines = {},
        reset = function(self)
          self.lines = {}
        end,
        minimal = function() end,
        set_lines = function(self, lines)
          self.lines = lines
        end,
        set_title = function() end,
        highlight = function() end,
      },
      picker = {
        items = function()
          return { item_with_full_data, item_with_partial_data }
        end,
      },
    }

    -- Mock config
    local test_config = helpers.get_default_config()
    test_config.algorithm = "classic"
    package.loaded["neural-open"] = {
      config = test_config,
    }

    -- Initialize the classic algorithm so registry will use it
    classic = require("neural-open.algorithms.classic")
    classic.init(test_config.algorithm_config.classic)

    -- Mock recent module
    package.loaded["neural-open.recent"] = {
      get_recency_list = function()
        return { "/test/recent1.lua", "/test/recent2.lua" }
      end,
      get_recency_map = function()
        return {
          ["/test/recent1.lua"] = { recent_rank = 1 },
          ["/test/recent2.lua"] = { recent_rank = 2 },
        }
      end,
    }
  end)

  describe("debug_preview", function()
    it("shows all features even when zero", function()
      ctx.item = item_with_partial_data
      debug.debug_preview(ctx)

      local lines = ctx.preview.lines
      local features_section_found = false
      local all_features_shown = {
        match = false,
        virtual_name = false,
        open = false,
        alt = false,
        proximity = false,
        project = false,
        frecency = false,
        recency = false,
      }

      for _, line in ipairs(lines) do
        if line:match("Features:") then
          features_section_found = true
        end

        for feature in pairs(all_features_shown) do
          local formatted_feature = feature:gsub("_", " "):sub(1, 1):upper() .. feature:gsub("_", " "):sub(2)
          -- Check for feature name with numeric values in the table row
          if line:match(formatted_feature) and line:match("%d+%.%d+") then
            all_features_shown[feature] = true
          end
        end
      end

      assert.is_true(features_section_found, "Features section should be present")
      for feature, shown in pairs(all_features_shown) do
        assert.is_true(shown, "Feature '" .. feature .. "' should be shown even if zero")
      end
    end)

    it("shows raw and normalized features on one line", function()
      ctx.item = item_with_full_data
      debug.debug_preview(ctx)

      local lines = ctx.preview.lines
      local found_combined_line = false

      for _, line in ipairs(lines) do
        -- Table row: "  Match               150.00  0.9933"
        if line:match("Match") and line:match("%d+%.%d+%s+%d+%.%d+") then
          found_combined_line = true
          break
        end
      end

      assert.is_true(found_combined_line, "Should have combined raw and normalized features on one line")
    end)

    it("shows all weighted component calculations even for zero features", function()
      ctx.item = item_with_partial_data
      debug.debug_preview(ctx)

      local lines = ctx.preview.lines
      local components_section_found = false
      local all_components_shown = {
        match = false,
        virtual_name = false,
        open = false,
        alt = false,
        proximity = false,
        project = false,
        frecency = false,
        recency = false,
      }

      for _, line in ipairs(lines) do
        if line:match("Weighted:") then
          components_section_found = true
        end

        for component in pairs(all_components_shown) do
          local formatted_component = component:gsub("_", " "):sub(1, 1):upper() .. component:gsub("_", " "):sub(2)
          -- Table row format: "  Match               0.9933  140.0 (140.0)     139.06"
          if line:match(formatted_component) and line:match("%d+%.%d+%s+%d+") then
            all_components_shown[component] = true
          end
        end
      end

      assert.is_true(components_section_found, "Weighted components section should be present")
      for component, shown in pairs(all_components_shown) do
        assert.is_true(shown, "Component '" .. component .. "' calculation should be shown even if zero")
      end
    end)

    it("handles items without nos field gracefully", function()
      ctx.item = item_without_nos
      assert.has_no.errors(function()
        debug.debug_preview(ctx)
      end)

      local lines = ctx.preview.lines
      assert.truthy(#lines > 0, "Should generate some output even without nos field")
    end)

    it("displays score totals prominently", function()
      ctx.item = item_with_full_data
      debug.debug_preview(ctx)

      local lines = ctx.preview.lines
      local found_neural_score = false
      local found_snacks_score = false

      for _, line in ipairs(lines) do
        if line:match("Total Neural Score:.*125%.25") then
          found_neural_score = true
        end
        if line:match("Final Snacks Score:.*150%.50") then
          found_snacks_score = true
        end
      end

      assert.is_true(found_neural_score, "Should display total neural score")
      assert.is_true(found_snacks_score, "Should display final Snacks score")
    end)

    it("shows metadata for open and alternate buffers", function()
      ctx.item = item_with_full_data
      debug.debug_preview(ctx)

      local lines = ctx.preview.lines
      local found_open_buffer = false
      local found_alternate = false
      local found_recent_rank = false

      for _, line in ipairs(lines) do
        if line:match("Open Buffer") then
          found_open_buffer = true
        end
        if line:match("Alternate Buffer") then
          found_alternate = true
        end
        if line:match("Recent Rank: 2") then
          found_recent_rank = true
        end
      end

      assert.is_true(found_open_buffer, "Should show open buffer metadata")
      assert.is_true(found_alternate, "Should show alternate buffer metadata")
      assert.is_true(found_recent_rank, "Should show recent rank metadata")
    end)
  end)
end)
