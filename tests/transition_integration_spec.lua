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
  local original_os_time
  local mock_time

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
        algorithm = "classic",
      }),
    }

    -- Control time for deterministic tests
    mock_time = 1000000000
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
    end

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
    os.time = original_os_time -- luacheck: ignore 122
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open"] = nil
    package.loaded["neural-open.transitions"] = nil
    package.loaded["neural-open.scorer"] = nil
    package.loaded["neural-open.algorithms.classic"] = nil
  end)

  describe("full flow: record -> compute -> score", function()
    it("should boost file score after recording transition", function()
      local source_file = "/test/project/a.lua"
      local dest_file = "/test/project/b.lua"

      -- Initial state: no transitions
      local scores_before = transitions.compute_scores_from(source_file)
      assert.same({}, scores_before)

      -- Record a transition
      transitions.record_transition(source_file, dest_file)

      -- Compute scores again
      local scores_after = transitions.compute_scores_from(source_file)
      assert.is_near(0.2, scores_after[dest_file], 0.001) -- 1-1/(1+1/4)

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
      assert.is_near(0.2, raw_features.transition, 0.001)

      local normalized_features = scorer.normalize_features(raw_features)
      assert.is_near(0.2, normalized_features.transition, 0.001)

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
      assert.is_near(0.556, scores[dest_file], 0.001) -- 1-1/(1+5/4)

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
      assert.is_near(0.556, raw_features.transition, 0.001)
    end)

    it("should handle multiple destination files", function()
      local source_file = "/test/project/a.lua"
      local dest_b = "/test/project/b.lua"
      local dest_c = "/test/project/c.lua"

      -- Record different numbers of transitions
      transitions.record_transition(source_file, dest_b)
      transitions.record_transition(source_file, dest_b)

      for _ = 1, 5 do
        transitions.record_transition(source_file, dest_c)
      end

      -- Compute scores
      local scores = transitions.compute_scores_from(source_file)

      -- b.lua: 2 visits → 1-1/(1+2/4) ≈ 0.333
      assert.is_near(0.333, scores[dest_b], 0.001)
      -- c.lua: 5 visits → 1-1/(1+5/4) ≈ 0.556
      assert.is_near(0.556, scores[dest_c], 0.001)

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
      local scores = transitions.compute_scores_from("")
      assert.same({}, scores)

      local context = {
        cwd = "/test/project",
        current_file = "",
        recent_files = {},
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

    it("should not error on self-transition", function()
      local same_file = "/test/project/a.lua"

      transitions.record_transition(same_file, same_file)

      local scores = transitions.compute_scores_from(same_file)
      assert.is_near(0.2, scores[same_file], 0.001)
    end)

    it("should handle missing weights data", function()
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

      transitions.record_transition(source_file, dest_file)

      local scores = transitions.compute_scores_from(source_file)

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
      assert.is_near(0.2 * 5, expected_contribution, 0.01) -- 1.0

      -- Calculate total score using flat input_buf
      local input_buf = make_input_buf(raw_features)
      local score = classic.calculate_score(input_buf)

      -- Score should include the transition contribution
      assert.is_true(score >= expected_contribution)
    end)
  end)

  describe("frecency decay in integration", function()
    it("should show lower scores after time passes", function()
      local source_file = "/test/project/a.lua"
      local dest_file = "/test/project/b.lua"

      -- Record transitions at current time
      for _ = 1, 3 do
        transitions.record_transition(source_file, dest_file)
      end

      -- Score now
      local scores_now = transitions.compute_scores_from(source_file)

      -- Advance 60 days (two half-lives)
      mock_time = mock_time + 60 * 24 * 3600

      -- Score after 60 days
      local scores_later = transitions.compute_scores_from(source_file)

      assert.is_true(scores_later[dest_file] < scores_now[dest_file])
    end)
  end)
end)
