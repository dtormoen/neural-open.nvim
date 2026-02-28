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
    config.algorithm_config.classic.picker_name = "test"
    local classic_instance = classic.create_instance(config.algorithm_config.classic)
    classic_instance.load_weights()

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
          algorithm = classic_instance, -- Add algorithm to context
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
          algorithm = classic_instance, -- Add algorithm to context
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

  describe("item picker debug_preview", function()
    local item_ctx
    local item_picker_item
    local item_picker_item_empty_tracking

    before_each(function()
      local helpers = require("tests.helpers")

      -- Clear caches
      package.loaded["neural-open.algorithms.registry"] = nil
      package.loaded["neural-open.algorithms.classic"] = nil
      package.loaded["neural-open.algorithms.nn"] = nil
      package.loaded["neural-open.debug"] = nil

      debug = require("neural-open.debug")

      -- Mock weights module
      local item_default_weights = helpers.get_default_config().item_algorithm_config.classic.default_weights

      package.loaded["neural-open.weights"] = {
        get_weights = function()
          return vim.deepcopy(item_default_weights)
        end,
        get_default_weights = function()
          return vim.deepcopy(item_default_weights)
        end,
        save_weights = function() end,
      }

      -- Load classic algorithm with item feature names
      local classic = require("neural-open.algorithms.classic")
      local item_scorer = require("neural-open.item_scorer")
      local config = vim.deepcopy(helpers.get_default_config().item_algorithm_config.classic)
      config.picker_name = "test_items"
      config.feature_names = item_scorer.ITEM_FEATURE_NAMES
      local classic_instance = classic.create_instance(config)
      classic_instance.load_weights()

      -- Item with full tracking data
      item_picker_item = {
        text = "build all",
        value = "just build-all",
        nos = {
          neural_score = 42.5,
          item_id = "just build-all",
          raw_features = {
            match = 80.0,
            frecency = 5.0,
            cwd_frecency = 3.0,
            recency = 2,
            cwd_recency = 1,
            text_length_inv = 9,
            not_last_selected = 1,
            transition = 0.35,
          },
          input_buf = { 0.75, 0.83, 0.75, 0.98, 0.99, 0.53, 1, 0.35 },
          ctx = {
            cwd = "/test/project",
            algorithm = classic_instance,
            picker_name = "test_items",
            tracking_data = {
              frecency = {
                ["just build-all"] = 5.0,
                ["just test"] = 3.0,
                ["just lint"] = 1.0,
              },
              cwd_frecency = {
                ["just build-all"] = 3.0,
                ["just test"] = 2.0,
              },
              recency_rank = {
                ["just build-all"] = 1,
                ["just test"] = 2,
                ["just lint"] = 3,
              },
              cwd_recency_rank = {
                ["just build-all"] = 1,
                ["just test"] = 2,
              },
              last_selected = "just test",
            },
          },
        },
      }

      -- Item with empty tracking data
      item_picker_item_empty_tracking = {
        text = "new command",
        nos = {
          neural_score = 10.0,
          item_id = "new command",
          raw_features = {
            match = 50.0,
            frecency = 0,
            cwd_frecency = 0,
            recency = 0,
            cwd_recency = 0,
            text_length_inv = 11,
            not_last_selected = 1,
            transition = 0,
          },
          input_buf = { 0.5, 0, 0, 0, 0, 0.48, 1, 0 },
          ctx = {
            cwd = "/test/project",
            algorithm = classic_instance,
            picker_name = "test_items",
            tracking_data = {
              frecency = {},
              cwd_frecency = {},
              recency_rank = {},
              cwd_recency_rank = {},
              last_selected = nil,
            },
          },
        },
      }

      -- Mock context for item picker
      item_ctx = {
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
            return { item_picker_item, item_picker_item_empty_tracking }
          end,
        },
      }

      -- Mock config
      local test_config = helpers.get_default_config()
      test_config.algorithm = "classic"
      package.loaded["neural-open"] = {
        config = test_config,
      }
    end)

    it("shows all sections for item with full tracking data", function()
      item_ctx.item = item_picker_item
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local found_item_preview = false
      local found_text = false
      local found_value = false
      local found_picker = false
      local found_features = false
      local found_frecent_global = false
      local found_frecent_cwd = false
      local found_recent_cwd = false
      local found_metadata = false

      for _, line in ipairs(lines) do
        if line:match("Preview") then
          found_item_preview = true
        end
        if line:match("Text:.*build all") then
          found_text = true
        end
        if line:match("Value:.*just build%-all") then
          found_value = true
        end
        if line:match("Picker:.*test_items") then
          found_picker = true
        end
        if line:match("Features:") then
          found_features = true
        end
        if line:match("Frecent Items %(Global%)") then
          found_frecent_global = true
        end
        if line:match("Frecent Items %(CWD%)") then
          found_frecent_cwd = true
        end
        if line:match("Recent Items %(CWD%)") then
          found_recent_cwd = true
        end
        if line:match("Metadata") then
          found_metadata = true
        end
      end

      assert.is_true(found_item_preview, "Should show Preview title")
      assert.is_true(found_text, "Should show item text")
      assert.is_true(found_value, "Should show item value when different from text")
      assert.is_true(found_picker, "Should show picker name")
      assert.is_true(found_features, "Should show features section")
      assert.is_true(found_frecent_global, "Should show frecent items (global)")
      assert.is_true(found_frecent_cwd, "Should show frecent items (CWD)")
      assert.is_true(found_recent_cwd, "Should show recent items (CWD)")
      assert.is_true(found_metadata, "Should show metadata section")
    end)

    it("renders gracefully with empty tracking data", function()
      item_ctx.item = item_picker_item_empty_tracking
      assert.has_no.errors(function()
        debug.debug_preview(item_ctx)
      end)

      local lines = item_ctx.preview.lines
      assert.is_true(#lines > 0, "Should generate output")

      local found_item_preview = false
      local found_text = false

      for _, line in ipairs(lines) do
        if line:match("Preview") then
          found_item_preview = true
        end
        if line:match("Text:.*new command") then
          found_text = true
        end
      end

      assert.is_true(found_item_preview, "Should show Preview title")
      assert.is_true(found_text, "Should show item text")
    end)

    it("shows correct 8 feature names in algorithm debug_view", function()
      item_ctx.item = item_picker_item
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local item_features_shown = {
        match = false,
        frecency = false,
        cwd_frecency = false,
        recency = false,
        cwd_recency = false,
        text_length_inv = false,
        not_last_selected = false,
        transition = false,
      }

      for _, line in ipairs(lines) do
        for feature in pairs(item_features_shown) do
          local formatted = feature:gsub("_", " "):sub(1, 1):upper() .. feature:gsub("_", " "):sub(2)
          if line:match(formatted) and line:match("%d+%.%d+") then
            item_features_shown[feature] = true
          end
        end
      end

      for feature, shown in pairs(item_features_shown) do
        assert.is_true(shown, "Item feature '" .. feature .. "' should be shown")
      end

      -- Verify file-only features are NOT shown
      local file_only_found = false
      for _, line in ipairs(lines) do
        if line:match("Virtual name") and line:match("%d+%.%d+") then
          file_only_found = true
        end
      end
      assert.is_false(file_only_found, "File-only feature 'virtual_name' should NOT be shown for item pickers")
    end)

    it("does not show file-specific sections", function()
      item_ctx.item = item_picker_item
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local found_trigram = false
      local found_file_transitions = false
      local found_recent_files = false

      for _, line in ipairs(lines) do
        if line:match("Trigram Similarity") then
          found_trigram = true
        end
        if line:match("Transitions %(Current File%)") then
          found_file_transitions = true
        end
        if line:match("Recent Files") then
          found_recent_files = true
        end
      end

      assert.is_false(found_trigram, "Should NOT show trigram section for item pickers")
      assert.is_false(found_file_transitions, "Should NOT show file transitions for item pickers")
      assert.is_false(found_recent_files, "Should NOT show recent files for item pickers")
    end)

    it("shows item transition sections when transition data exists", function()
      -- Add transition_scores to context
      item_picker_item.nos.ctx.transition_scores = {
        ["just test"] = 0.45,
        ["just lint"] = 0.20,
      }

      -- Mock item_tracking module for get_transition_frecency
      local mock_time = os.time()
      local half_life = 30 * 24 * 3600
      local lambda = math.log(2) / half_life
      -- Deadline that gives a score of ~1: deadline = now + ln(1)/lambda = now
      local deadline_for_score_1 = mock_time + math.log(1) / lambda

      package.loaded["neural-open.item_tracking"] = {
        init = function() end,
        reset = function() end,
        get_transition_frecency = function()
          return {
            ["just build-all"] = { ["just test"] = deadline_for_score_1 },
          }
        end,
      }

      item_ctx.item = item_picker_item
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local found_from_last = false
      local found_all_items = false

      for _, line in ipairs(lines) do
        if line:match("Transitions %(From Last Selected%)") then
          found_from_last = true
        end
        if line:match("Transitions %(All Items%)") then
          found_all_items = true
        end
      end

      assert.is_true(found_from_last, "Should show Transitions (From Last Selected) section")
      assert.is_true(found_all_items, "Should show Transitions (All Items) section")

      -- Clean up
      package.loaded["neural-open.item_tracking"] = nil
    end)

    it("shows user preview content when user_preview captures sync output", function()
      item_ctx.item = item_picker_item
      item_ctx.meta = {
        nos_user_preview = function(preview_ctx)
          preview_ctx.preview:set_lines({
            "recipe build-all:",
            "    cargo build --release",
            "    echo 'done'",
          })
        end,
      }
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local found_recipe_line = false
      local found_numbered_line = false

      for _, line in ipairs(lines) do
        if line:match("recipe build%-all") then
          found_recipe_line = true
        end
        -- Should have numbered format like file preview: "  NNN  content"
        if line:match("^%s+%d+%s+") then
          found_numbered_line = true
        end
      end

      assert.is_true(found_recipe_line, "Should show user preview content")
      assert.is_true(found_numbered_line, "Should show numbered preview lines like file preview")
    end)

    it("falls back to text/value when user_preview is async (no set_lines)", function()
      item_ctx.item = item_picker_item
      item_ctx.meta = {
        nos_user_preview = function()
          -- Simulates async preview that doesn't call set_lines synchronously
        end,
      }
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local found_text = false

      for _, line in ipairs(lines) do
        if line:match("Text:.*build all") then
          found_text = true
        end
      end

      assert.is_true(found_text, "Should fall back to text display when preview capture fails")
    end)

    it("limits user preview capture to 10 lines", function()
      item_ctx.item = item_picker_item
      local many_lines = {}
      for i = 1, 20 do
        many_lines[i] = "line " .. i
      end
      item_ctx.meta = {
        nos_user_preview = function(preview_ctx)
          preview_ctx.preview:set_lines(many_lines)
        end,
      }
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local preview_line_count = 0

      for _, line in ipairs(lines) do
        -- Count numbered preview lines (format: "  NNN  content")
        if line:match("^%s+%d+%s+line %d+") then
          preview_line_count = preview_line_count + 1
        end
      end

      assert.equals(10, preview_line_count, "Should limit preview to 10 lines")
    end)

    it("falls back to text/value when no user_preview is provided", function()
      item_ctx.item = item_picker_item
      -- No meta.nos_user_preview set
      debug.debug_preview(item_ctx)

      local lines = item_ctx.preview.lines
      local found_text = false
      local found_value = false

      for _, line in ipairs(lines) do
        if line:match("Text:.*build all") then
          found_text = true
        end
        if line:match("Value:.*just build%-all") then
          found_value = true
        end
      end

      assert.is_true(found_text, "Should show text when no user preview")
      assert.is_true(found_value, "Should show value when no user preview")
    end)

    it("gracefully handles user_preview that errors", function()
      item_ctx.item = item_picker_item
      item_ctx.meta = {
        nos_user_preview = function()
          error("preview failed!")
        end,
      }
      assert.has_no.errors(function()
        debug.debug_preview(item_ctx)
      end)

      local lines = item_ctx.preview.lines
      local found_text = false
      for _, line in ipairs(lines) do
        if line:match("Text:.*build all") then
          found_text = true
        end
      end

      assert.is_true(found_text, "Should fall back to text display when preview errors")
    end)
  end)
end)
