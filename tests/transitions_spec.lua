describe("transitions module", function()
  local helpers = require("tests.helpers")
  local transitions
  local mock_db
  local mock_neural_open

  before_each(function()
    helpers.setup()

    -- Mock db module
    mock_db = {
      weights_data = {},
      get_weights = function()
        return vim.deepcopy(mock_db.weights_data)
      end,
      save_weights = function(data)
        mock_db.weights_data = vim.deepcopy(data)
      end,
    }

    -- Mock neural-open module for config
    mock_neural_open = {
      config = helpers.create_test_config({
        transition_history_size = 5, -- Small size for testing
      }),
    }

    -- Replace modules
    package.loaded["neural-open.db"] = mock_db
    package.loaded["neural-open"] = mock_neural_open
    package.loaded["neural-open.transitions"] = nil

    -- Load transitions module fresh
    transitions = require("neural-open.transitions")
  end)

  after_each(function()
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open"] = nil
    package.loaded["neural-open.transitions"] = nil
  end)

  describe("record_transition", function()
    it("should add transition to history with normalized paths", function()
      transitions.record_transition("/path/to/a.lua", "/path/to/b.lua")

      local data = mock_db.get_weights()
      assert.equals(1, #data.transition_history)
      assert.is_not_nil(data.transition_history[1].from)
      assert.is_not_nil(data.transition_history[1].to)
      assert.is_number(data.transition_history[1].timestamp)
    end)

    it("should append multiple transitions in order", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")
      transitions.record_transition("/path/a.lua", "/path/c.lua")
      transitions.record_transition("/path/b.lua", "/path/d.lua")

      local data = mock_db.get_weights()
      assert.equals(3, #data.transition_history)
    end)

    it("should enforce ring buffer limit", function()
      -- Add 6 transitions (limit is 5)
      for i = 1, 6 do
        transitions.record_transition("/path/a.lua", "/path/b" .. i .. ".lua")
      end

      local data = mock_db.get_weights()
      assert.equals(5, #data.transition_history)

      -- First entry should be removed, so b2.lua should be the oldest
      -- We can't check exact path due to normalization, but we can check count
      local to_paths = {}
      for _, entry in ipairs(data.transition_history) do
        table.insert(to_paths, entry.to)
      end

      -- Should not contain b1.lua (removed)
      local has_b1 = false
      for _, path in ipairs(to_paths) do
        if path:match("b1%.lua") then
          has_b1 = true
          break
        end
      end
      assert.is_false(has_b1)

      -- Should contain b6.lua (most recent)
      local has_b6 = false
      for _, path in ipairs(to_paths) do
        if path:match("b6%.lua") then
          has_b6 = true
          break
        end
      end
      assert.is_true(has_b6)
    end)

    it("should enforce limit when adding many entries at once", function()
      -- Add 10 transitions (limit is 5)
      for i = 1, 10 do
        transitions.record_transition("/path/a.lua", "/path/b" .. i .. ".lua")
      end

      local data = mock_db.get_weights()
      assert.equals(5, #data.transition_history)

      -- Should contain b6-b10 (last 5)
      local to_paths = {}
      for _, entry in ipairs(data.transition_history) do
        table.insert(to_paths, entry.to)
      end

      -- Should have b10 (most recent)
      local has_b10 = false
      for _, path in ipairs(to_paths) do
        if path:match("b10%.lua") then
          has_b10 = true
          break
        end
      end
      assert.is_true(has_b10)
    end)

    it("should preserve existing weights data when adding transitions", function()
      -- Setup existing weights
      mock_db.weights_data = {
        classic = {
          match = 100,
          virtual_name = 90,
        },
        nn = {
          weights = { { { 1, 2, 3 } } },
        },
      }

      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local data = mock_db.get_weights()
      assert.is_not_nil(data.classic)
      assert.equals(100, data.classic.match)
      assert.is_not_nil(data.nn)
      assert.is_not_nil(data.nn.weights)
      assert.equals(1, #data.transition_history)
    end)
  end)

  describe("compute_scores_from", function()
    it("should return empty map when no transitions", function()
      local scores = transitions.compute_scores_from("/path/a.lua")
      assert.same({}, scores)
    end)

    it("should return empty map when no matching source file", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
        },
      }

      local scores = transitions.compute_scores_from("/path/different.lua")
      assert.same({}, scores)
    end)

    it("should calculate correct score for single transition", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
        },
      }

      local scores = transitions.compute_scores_from("/path/a.lua")
      assert.equals(0.5, scores["/path/b.lua"]) -- 1-1/(1+1) = 0.5
    end)

    it("should calculate correct score for multiple transitions to same file", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 124 },
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 125 },
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 126 },
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 127 },
        },
      }

      local scores = transitions.compute_scores_from("/path/a.lua")
      -- 1 - 1/(1+5) = 1 - 1/6 = 0.8333...
      assert.is_near(0.833, scores["/path/b.lua"], 0.001)
    end)

    it("should calculate scores for multiple destinations", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 124 },
          { from = "/path/a.lua", to = "/path/c.lua", timestamp = 125 },
          { from = "/path/a.lua", to = "/path/c.lua", timestamp = 126 },
          { from = "/path/a.lua", to = "/path/c.lua", timestamp = 127 },
          { from = "/path/a.lua", to = "/path/c.lua", timestamp = 128 },
          { from = "/path/a.lua", to = "/path/c.lua", timestamp = 129 },
        },
      }

      local scores = transitions.compute_scores_from("/path/a.lua")

      -- b.lua: 2 transitions = 1-1/(1+2) = 0.666...
      assert.is_near(0.666, scores["/path/b.lua"], 0.001)

      -- c.lua: 5 transitions = 1-1/(1+5) = 0.833...
      assert.is_near(0.833, scores["/path/c.lua"], 0.001)
    end)

    it("should only count transitions from the specified source", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 124 },
          { from = "/path/different.lua", to = "/path/b.lua", timestamp = 125 },
          { from = "/path/different.lua", to = "/path/b.lua", timestamp = 126 },
        },
      }

      local scores = transitions.compute_scores_from("/path/a.lua")

      -- Should only count 2 transitions (from a.lua -> b.lua)
      assert.is_near(0.666, scores["/path/b.lua"], 0.001)
    end)

    it("should handle edge case with zero transitions", function()
      mock_db.weights_data = {
        transition_history = {},
      }

      local scores = transitions.compute_scores_from("/path/a.lua")
      assert.same({}, scores)
    end)

    it("should normalize paths before lookup", function()
      -- Set up transitions with normalized paths
      local temp_dir = helpers.create_temp_dir()
      local file_a = temp_dir .. "/a.lua"
      local file_b = temp_dir .. "/b.lua"

      -- Write dummy files to ensure they exist for normalization
      vim.fn.writefile({}, file_a)
      vim.fn.writefile({}, file_b)

      mock_db.weights_data = {
        transition_history = {
          { from = file_a, to = file_b, timestamp = 123 },
        },
      }

      local scores = transitions.compute_scores_from(file_a)
      assert.is_not_nil(scores[file_b])
      assert.equals(0.5, scores[file_b])

      -- Cleanup
      helpers.cleanup_temp_dir(temp_dir)
    end)
  end)

  describe("ring buffer behavior", function()
    it("should maintain exactly max_size entries", function()
      -- Add exactly max_size entries
      for i = 1, 5 do
        transitions.record_transition("/path/a.lua", "/path/b" .. i .. ".lua")
      end

      local data = mock_db.get_weights()
      assert.equals(5, #data.transition_history)
    end)

    it("should remove oldest entry when exceeding limit by 1", function()
      -- Add max_size + 1 entries
      for i = 1, 6 do
        transitions.record_transition("/path/a.lua", "/path/b" .. i .. ".lua")
      end

      local data = mock_db.get_weights()
      assert.equals(5, #data.transition_history)

      -- First entry (b1) should be removed
      local has_b1 = false
      for _, entry in ipairs(data.transition_history) do
        if entry.to:match("b1%.lua") then
          has_b1 = true
          break
        end
      end
      assert.is_false(has_b1)
    end)

    it("should handle custom transition_history_size config", function()
      -- Update config with larger size
      mock_neural_open.config.transition_history_size = 10

      -- Add 11 entries
      for i = 1, 11 do
        transitions.record_transition("/path/a.lua", "/path/b" .. i .. ".lua")
      end

      local data = mock_db.get_weights()
      assert.equals(10, #data.transition_history)
    end)
  end)

  describe("integration with db module", function()
    it("should use atomic writes through db.save_weights", function()
      local save_count = 0
      local original_save = mock_db.save_weights
      mock_db.save_weights = function(data)
        save_count = save_count + 1
        original_save(data)
      end

      transitions.record_transition("/path/a.lua", "/path/b.lua")
      assert.equals(1, save_count)
    end)

    it("should handle missing transition_history gracefully", function()
      -- Start with no transition_history
      mock_db.weights_data = {}

      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local data = mock_db.get_weights()
      assert.is_not_nil(data.transition_history)
      assert.equals(1, #data.transition_history)
    end)

    it("should handle nil weights data gracefully", function()
      -- db.get_weights returns nil
      mock_db.get_weights = function()
        return nil
      end

      transitions.record_transition("/path/a.lua", "/path/b.lua")

      -- Should not error and should create new weights data
      local data = mock_db.weights_data
      assert.is_not_nil(data.transition_history)
      assert.equals(1, #data.transition_history)
    end)
  end)
end)
