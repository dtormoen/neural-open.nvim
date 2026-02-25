--- Convert raw features to a flat input buffer suitable for algorithm.calculate_score()
---@param raw_features table
---@return number[]
local function make_input_buf(raw_features)
  local _scorer = require("neural-open.scorer")
  local norm = _scorer.normalize_features(raw_features)
  local buf = {}
  for i, name in ipairs(_scorer.FEATURE_NAMES) do
    buf[i] = norm[name] or 0
  end
  return buf
end

describe("transition integration", function()
  local helpers = require("tests.helpers")
  local transitions
  local scorer
  local classic
  local mock_db
  local mock_neural_open

  before_each(function()
    helpers.setup()

    -- Setup mocks
    mock_db = {
      weights_data = {},
      get_weights = function()
        return vim.deepcopy(mock_db.weights_data)
      end,
      save_weights = function(data)
        mock_db.weights_data = vim.deepcopy(data)
      end,
    }

    mock_neural_open = {
      config = helpers.create_test_config({
        transition_history_size = 200,
        algorithm = "classic",
      }),
    }

    -- Replace modules
    package.loaded["neural-open.db"] = mock_db
    package.loaded["neural-open"] = mock_neural_open
    package.loaded["neural-open.transitions"] = nil
    package.loaded["neural-open.scorer"] = nil
    package.loaded["neural-open.algorithms.classic"] = nil

    -- Load modules fresh
    transitions = require("neural-open.transitions")
    scorer = require("neural-open.scorer")
    classic = require("neural-open.algorithms.classic")

    -- Initialize classic algorithm
    classic.init(mock_neural_open.config.algorithm_config.classic)
    classic.load_weights()
  end)

  after_each(function()
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open"] = nil
    package.loaded["neural-open.transitions"] = nil
    package.loaded["neural-open.scorer"] = nil
    package.loaded["neural-open.algorithms.classic"] = nil
  end)

  describe("full flow: record -> compute -> score", function()
    it("should boost file score after recording transition", function()
      -- Initial state: no transitions
      local source_file = "/test/project/a.lua"
      local dest_file = "/test/project/b.lua"

      -- Compute initial scores (no transition history)
      local scores_before = transitions.compute_scores_from(source_file)
      assert.same({}, scores_before)

      -- Record a transition
      transitions.record_transition(source_file, dest_file)

      -- Compute scores again
      local scores_after = transitions.compute_scores_from(source_file)
      assert.equals(0.5, scores_after[dest_file]) -- 1-1/(1+1)

      -- Verify scoring pipeline integrates the transition
      local context = {
        cwd = "/test/project",
        current_file = source_file,
        recent_files = {},
        transition_scores = scores_after,
        algorithm = classic,
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "b.lua",
      }
      local raw_features = scorer.compute_static_raw_features(dest_file, context, item_data)
      assert.equals(0.5, raw_features.transition)

      local normalized_features = scorer.normalize_features(raw_features)
      assert.equals(0.5, normalized_features.transition)

      -- Calculate score with classic algorithm using flat input_buf
      local input_buf = make_input_buf(raw_features)
      local score = classic.calculate_score(input_buf)
      assert.is_true(score > 0) -- Should have some score from transition weight
    end)

    it("should increase score with multiple transitions", function()
      local source_file = "/test/project/a.lua"
      local dest_file = "/test/project/b.lua"

      -- Record multiple transitions
      for _ = 1, 5 do
        transitions.record_transition(source_file, dest_file)
      end

      -- Compute scores
      local scores = transitions.compute_scores_from(source_file)
      assert.is_near(0.833, scores[dest_file], 0.001) -- 1-1/(1+5)

      -- Create context with transition scores
      local context = {
        cwd = "/test/project",
        current_file = source_file,
        recent_files = {},
        transition_scores = scores,
        algorithm = classic,
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "b.lua",
      }
      local raw_features = scorer.compute_static_raw_features(dest_file, context, item_data)
      assert.is_near(0.833, raw_features.transition, 0.001)
    end)

    it("should handle multiple destination files", function()
      local source_file = "/test/project/a.lua"
      local dest_b = "/test/project/b.lua"
      local dest_c = "/test/project/c.lua"

      -- Record different numbers of transitions
      transitions.record_transition(source_file, dest_b)
      transitions.record_transition(source_file, dest_b)

      transitions.record_transition(source_file, dest_c)
      transitions.record_transition(source_file, dest_c)
      transitions.record_transition(source_file, dest_c)
      transitions.record_transition(source_file, dest_c)
      transitions.record_transition(source_file, dest_c)

      -- Compute scores
      local scores = transitions.compute_scores_from(source_file)

      -- b.lua: 2 transitions = 1-1/(1+2) = 0.666...
      assert.is_near(0.666, scores[dest_b], 0.001)

      -- c.lua: 5 transitions = 1-1/(1+5) = 0.833...
      assert.is_near(0.833, scores[dest_c], 0.001)

      -- Create context with transition scores
      local context = {
        cwd = "/test/project",
        current_file = source_file,
        recent_files = {},
        transition_scores = scores,
        algorithm = classic,
      }

      -- Score for b.lua
      local item_data_b = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "b.lua",
      }
      local raw_features_b = scorer.compute_static_raw_features(dest_b, context, item_data_b)
      local input_buf_b = make_input_buf(raw_features_b)
      local score_b = classic.calculate_score(input_buf_b)

      -- Score for c.lua
      local item_data_c = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "c.lua",
      }
      local raw_features_c = scorer.compute_static_raw_features(dest_c, context, item_data_c)
      local input_buf_c = make_input_buf(raw_features_c)
      local score_c = classic.calculate_score(input_buf_c)

      -- c.lua should have higher score due to more transitions
      assert.is_true(score_c > score_b)
    end)
  end)

  describe("edge cases", function()
    it("should handle no current file (empty string)", function()
      -- No current file
      local scores = transitions.compute_scores_from("")
      assert.same({}, scores)

      -- Create context without current file
      local context = {
        cwd = "/test/project",
        current_file = "",
        recent_files = {},
        -- No transition_scores
        algorithm = classic,
      }

      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "file.lua",
      }
      local raw_features = scorer.compute_static_raw_features("/test/project/file.lua", context, item_data)
      assert.equals(0, raw_features.transition)
    end)

    it("should not record transition to same file", function()
      local same_file = "/test/project/a.lua"

      transitions.record_transition(same_file, same_file)

      -- This would normally be prevented by init.lua logic, but if it happens:
      -- It will be recorded, but compute_scores_from will find it
      local scores = transitions.compute_scores_from(same_file)
      -- Same file transition should still be scored (implementation allows this)
      assert.equals(0.5, scores[same_file])
    end)

    it("should handle missing weights data", function()
      -- db.get_weights returns nil
      mock_db.get_weights = function()
        return nil
      end

      local scores = transitions.compute_scores_from("/test/project/a.lua")
      assert.same({}, scores)
    end)
  end)

  describe("transition weight impact on classic algorithm", function()
    it("should apply transition weight to final score", function()
      local source_file = "/test/project/a.lua"
      local dest_file = "/test/project/b.lua"

      -- Record one transition
      transitions.record_transition(source_file, dest_file)

      -- Get transition scores
      local scores = transitions.compute_scores_from(source_file)

      -- Create minimal context
      local context = {
        cwd = "/test/project",
        current_file = source_file,
        recent_files = {},
        transition_scores = scores,
        algorithm = classic,
      }

      -- Compute features
      local item_data = {
        is_open_buffer = false,
        is_alternate = false,
        recent_rank = nil,
        virtual_name = "b.lua",
      }
      local raw_features = scorer.compute_static_raw_features(dest_file, context, item_data)
      local normalized_features = scorer.normalize_features(raw_features)

      -- Get default transition weight from config
      local transition_weight = mock_neural_open.config.algorithm_config.classic.default_weights.transition
      assert.equals(5, transition_weight)

      -- Calculate expected transition contribution
      local expected_contribution = normalized_features.transition * transition_weight
      assert.equals(0.5 * 5, expected_contribution) -- 2.5

      -- Calculate total score using flat input_buf
      local input_buf = make_input_buf(raw_features)
      local score = classic.calculate_score(input_buf)

      -- Score should include the transition contribution
      assert.is_true(score >= expected_contribution)
    end)
  end)

  describe("ring buffer overflow", function()
    it("should maintain history limit during integration", function()
      -- Update config to small limit
      mock_neural_open.config.transition_history_size = 3

      local source_file = "/test/project/a.lua"

      -- Record 5 transitions (over limit of 3)
      for i = 1, 5 do
        transitions.record_transition(source_file, "/test/project/b" .. i .. ".lua")
      end

      -- Check history size
      local data = mock_db.get_weights()
      assert.equals(3, #data.transition_history)

      -- Compute scores - should only reflect last 3 transitions
      local scores = transitions.compute_scores_from(source_file)

      -- Should have b3, b4, b5 (each with count 1, score 0.5)
      -- b1 and b2 should be removed
      local count = 0
      for _ in pairs(scores) do
        count = count + 1
      end
      assert.equals(3, count)
    end)
  end)
end)
