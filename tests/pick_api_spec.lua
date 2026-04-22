local helpers = require("tests.helpers")

describe("public picker API", function()
  local neural_open
  local mock_db
  local original_os_time
  local mock_time

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()

    mock_time = 1000000000
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
    end

    -- Mock db module
    mock_db = {
      weights_data = {},
      tracking_data = {},
      get_weights = function(_picker_name, _latency_ctx)
        return vim.deepcopy(mock_db.weights_data)
      end,
      save_weights = function(_picker_name, data, _latency_ctx)
        mock_db.weights_data = vim.deepcopy(data)
      end,
      get_tracking = function(_picker_name, _latency_ctx)
        return vim.deepcopy(mock_db.tracking_data)
      end,
      save_tracking = function(_picker_name, data, _latency_ctx)
        mock_db.tracking_data = vim.deepcopy(data)
      end,
    }
    package.loaded["neural-open.db"] = mock_db

    -- Mock weights module
    package.loaded["neural-open.weights"] = {
      get_weights = function(_algo, _picker_name)
        return {}
      end,
      save_weights = function() end,
      get_default_weights = function(_algo)
        return {}
      end,
      reset_weights = function() end,
    }

    neural_open = require("neural-open")
    neural_open.setup({ algorithm = "classic" })
  end)

  after_each(function()
    os.time = original_os_time -- luacheck: ignore 122
    helpers.clear_plugin_modules()
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open.weights"] = nil
  end)

  describe("config", function()
    it("includes file_sources in default config", function()
      local defaults = helpers.get_default_config()
      assert.is_not_nil(defaults.file_sources)
      assert.are.same({ "buffers", "neural_recent", "files", "git_files" }, defaults.file_sources)
    end)

    it("allows overriding file_sources via setup", function()
      neural_open.setup({ file_sources = { "files", "git_files" } })
      assert.are.same({ "files", "git_files" }, neural_open.config.file_sources)
    end)
  end)

  describe("register_picker", function()
    it("stores config in the registry", function()
      neural_open.register_picker("test_picker", {
        title = "Test Picker",
        items = { { text = "item1" } },
      })

      -- Verify it can be opened (registration succeeded) via command completion
      local completions = neural_open.complete("", "NeuralOpen pick ", 18)
      assert.is_true(vim.tbl_contains(completions, "test_picker"))
    end)

    it("defaults type to 'item' when not specified", function()
      neural_open.register_picker("my_picker", {
        title = "My Picker",
      })

      -- Verify the picker is registered and completable
      local completions = neural_open.complete("", "NeuralOpen pick ", 18)
      assert.is_true(vim.tbl_contains(completions, "my_picker"))
    end)

    it("preserves type='file' when specified", function()
      neural_open.register_picker("file_picker", {
        type = "file",
        title = "File Picker",
      })

      local completions = neural_open.complete("", "NeuralOpen pick ", 18)
      assert.is_true(vim.tbl_contains(completions, "file_picker"))
    end)
  end)

  describe("command", function()
    it("handles pick subcommand", function()
      -- Register a picker first to avoid errors from Snacks
      neural_open.register_picker("cmd_picker", {
        title = "Command Picker",
        items = { { text = "item1" } },
      })

      -- Just test the error case for missing picker name
      local notifications = {}
      local orig_notify = vim.notify
      vim.notify = function(msg, level)
        notifications[#notifications + 1] = { msg = msg, level = level }
      end

      neural_open.command({ fargs = { "pick" } })

      vim.notify = orig_notify
      assert.equals(1, #notifications)
      assert.is_true(notifications[1].msg:find("Usage") ~= nil)
    end)
  end)

  describe("complete", function()
    it("includes 'pick' in subcommand completions", function()
      local completions = neural_open.complete("", "NeuralOpen ", 11)
      assert.is_true(vim.tbl_contains(completions, "pick"))
      assert.is_true(vim.tbl_contains(completions, "algorithm"))
      assert.is_true(vim.tbl_contains(completions, "reset"))
    end)

    it("completes registered picker names for pick subcommand", function()
      neural_open.register_picker("alpha_picker", { title = "Alpha" })
      neural_open.register_picker("beta_picker", { title = "Beta" })

      local completions = neural_open.complete("", "NeuralOpen pick ", 18)
      assert.is_true(vim.tbl_contains(completions, "alpha_picker"))
      assert.is_true(vim.tbl_contains(completions, "beta_picker"))
    end)

    it("filters picker names by prefix", function()
      neural_open.register_picker("alpha_picker", { title = "Alpha" })
      neural_open.register_picker("beta_picker", { title = "Beta" })

      local completions = neural_open.complete("al", "NeuralOpen pick al", 20)
      assert.equals(1, #completions)
      assert.equals("alpha_picker", completions[1])
    end)

    it("returns empty for pick when no pickers registered", function()
      local completions = neural_open.complete("", "NeuralOpen pick ", 18)
      assert.are.same({}, completions)
    end)
  end)

  describe("create_item_confirm_handler", function()
    it("calls user confirm function when provided", function()
      local confirm_called = false
      local confirm_picker = nil
      local confirm_item = nil

      neural_open.register_picker("confirm_test", {
        title = "Confirm Test",
        items = { { text = "item1" } },
        confirm = function(picker, item)
          confirm_called = true
          confirm_picker = picker
          confirm_item = item
        end,
      })

      -- Build a source config to get the confirm handler
      -- We need to trigger the build indirectly. The easiest way is to
      -- ensure the picker is registered, then inspect the built source.
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("confirm_test")

      local source = snacks_sources["neural_open_confirm_test"]
      assert.is_not_nil(source)
      assert.is_not_nil(source.confirm)

      -- Call confirm with a mock picker and item
      local mock_picker = {
        items = function()
          return {
            { nos = { item_id = "item1" }, text = "item1" },
          }
        end,
      }
      local mock_item = {
        text = "item1",
        nos = {
          item_id = "item1",
          ctx = {
            cwd = "/test",
            algorithm = {
              update_weights = function() end,
            },
          },
        },
      }

      source.confirm(mock_picker, mock_item)

      assert.is_true(confirm_called)
      assert.equals(mock_picker, confirm_picker)
      assert.equals(mock_item, confirm_item)

      package.loaded["snacks"] = nil
    end)

    it("records item selection for tracking on confirm", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("tracking_test", {
        title = "Tracking Test",
        items = { { text = "tracked_item" } },
      })

      local source = snacks_sources["neural_open_tracking_test"]

      -- Call confirm
      local mock_picker = {
        items = function()
          return {
            { nos = { item_id = "tracked_item" }, text = "tracked_item" },
          }
        end,
      }
      local mock_item = {
        text = "tracked_item",
        nos = {
          item_id = "tracked_item",
          ctx = {
            cwd = "/test/project",
            algorithm = {
              update_weights = function() end,
            },
          },
        },
      }

      source.confirm(mock_picker, mock_item)

      -- The item_tracking.record_selection is called inside vim.schedule,
      -- so we need to flush the scheduler
      vim.wait(100, function()
        return false
      end)

      -- Check that item_tracking recorded the selection
      local item_tracking = require("neural-open.item_tracking")
      local tracking_data = item_tracking.get_tracking_data("tracking_test", "/test/project")
      assert.is_true(tracking_data.frecency["tracked_item"] > 0)

      package.loaded["snacks"] = nil
    end)
  end)

  describe("build_item_source_config", function()
    it("builds source with static items", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      local test_items = { { text = "item1" }, { text = "item2" } }
      neural_open.pick("static_items", {
        title = "Static Items",
        items = test_items,
      })

      local source = snacks_sources["neural_open_static_items"]
      assert.is_not_nil(source)
      assert.is_not_nil(source.finder)
      assert.is_not_nil(source.transform)
      assert.is_not_nil(source.matcher)
      assert.is_not_nil(source.confirm)
      assert.equals("Static Items", source.title)

      package.loaded["snacks"] = nil
    end)

    it("builds source with custom finder", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("custom_finder", {
        title = "Custom Finder",
        finder = function()
          return { { text = "dynamic_item" } }
        end,
      })

      local source = snacks_sources["neural_open_custom_finder"]
      assert.is_not_nil(source)
      assert.is_not_nil(source.finder)

      package.loaded["snacks"] = nil
    end)

    it("forwards actions and win config from user picker_config", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      local user_actions = {
        add_args = function() end,
      }
      local user_win = {
        input = {
          keys = {
            ["<C-a>"] = { "add_args", mode = { "n", "i" }, desc = "Add args" },
          },
        },
      }

      neural_open.pick("forwarding_test", {
        title = "Forwarding Test",
        items = { { text = "item1" } },
        actions = user_actions,
        win = user_win,
      })

      local source = snacks_sources["neural_open_forwarding_test"]
      assert.is_not_nil(source)
      assert.equals(user_actions, source.actions)
      assert.equals(user_win, source.win)

      package.loaded["snacks"] = nil
    end)
  end)

  describe("build_file_source_config", function()
    it("builds file source with correct pipeline", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("custom_files", {
        type = "file",
        title = "Custom Files",
        finder = function()
          return {}
        end,
      })

      local source = snacks_sources["neural_open_custom_files"]
      assert.is_not_nil(source)
      assert.equals("file", source.format)
      assert.is_not_nil(source.transform)
      assert.is_not_nil(source.matcher)
      assert.is_true(source.matcher.sort_empty)
      assert.is_true(source.matcher.frecency)
      assert.equals(false, source.matcher.cwd_bonus)
      assert.equals("Custom Files", source.title)

      package.loaded["snacks"] = nil
    end)

    it("uses default confirm_handler for file pickers without custom confirm", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("file_no_confirm", {
        type = "file",
        title = "File No Confirm",
        finder = function()
          return {}
        end,
      })

      local source = snacks_sources["neural_open_file_no_confirm"]
      assert.is_not_nil(source.confirm)
      -- The confirm should be the default confirm_handler (a function)
      assert.equals("function", type(source.confirm))

      package.loaded["snacks"] = nil
    end)
  end)

  describe("pick", function()
    it("registers and opens a picker that was not previously registered", function()
      neural_open._initialized = true
      local picked_source = nil
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function(source_name)
            picked_source = source_name
          end,
        },
      }

      neural_open.pick("new_picker", {
        title = "New Picker",
        items = { { text = "hello" } },
      })

      assert.equals("neural_open_new_picker", picked_source)
      assert.is_not_nil(snacks_sources["neural_open_new_picker"])

      -- Verify it's now in completions
      local completions = neural_open.complete("", "NeuralOpen pick ", 18)
      assert.is_true(vim.tbl_contains(completions, "new_picker"))

      package.loaded["snacks"] = nil
    end)

    it("merges opts over previously registered config", function()
      neural_open._initialized = true
      local pick_opts = nil
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function(source_name, opts)
            pick_opts = opts
          end,
        },
      }

      -- Register with initial config
      neural_open.register_picker("merge_test", {
        title = "Original Title",
        items = { { text = "item1" } },
      })

      -- Pick with overrides
      neural_open.pick("merge_test", { title = "Override Title" })

      assert.equals("Override Title", pick_opts.title)

      package.loaded["snacks"] = nil
    end)

    it("uses existing registration when opts is empty", function()
      neural_open._initialized = true
      local pick_opts = nil
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function(source_name, opts)
            pick_opts = opts
          end,
        },
      }

      neural_open.register_picker("existing_test", {
        title = "Existing Title",
        items = { { text = "item1" } },
      })

      -- Pick without additional opts
      neural_open.pick("existing_test")

      assert.equals("Existing Title", pick_opts.title)

      package.loaded["snacks"] = nil
    end)

    it("dispatches to item source for type='item'", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("item_type_test", {
        type = "item",
        title = "Item Picker",
        items = { { text = "item1" } },
      })

      local source = snacks_sources["neural_open_item_type_test"]
      assert.is_not_nil(source)
      -- Item pickers don't have frecency in matcher
      assert.is_nil(source.matcher.frecency)

      package.loaded["snacks"] = nil
    end)

    it("dispatches to file source for type='file'", function()
      neural_open._initialized = true
      local snacks_sources = {}
      package.loaded["snacks"] = {
        picker = {
          sources = snacks_sources,
          pick = function() end,
        },
      }

      neural_open.pick("file_type_test", {
        type = "file",
        title = "File Picker",
        finder = function()
          return {}
        end,
      })

      local source = snacks_sources["neural_open_file_type_test"]
      assert.is_not_nil(source)
      -- File pickers have frecency in matcher
      assert.is_true(source.matcher.frecency)

      package.loaded["snacks"] = nil
    end)
  end)
end)
