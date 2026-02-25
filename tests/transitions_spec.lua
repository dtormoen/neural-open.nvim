describe("transitions module", function()
  local helpers = require("tests.helpers")
  local transitions
  local mock_db
  local original_os_time
  local mock_time

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

    -- Control time for deterministic tests
    mock_time = 1000000000 -- fixed timestamp
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
    end

    -- Replace modules
    package.loaded["neural-open.db"] = mock_db
    package.loaded["neural-open.transitions"] = nil

    -- Load transitions module fresh
    transitions = require("neural-open.transitions")
  end)

  after_each(function()
    os.time = original_os_time -- luacheck: ignore 122
    package.loaded["neural-open.db"] = nil
    package.loaded["neural-open.transitions"] = nil
  end)

  describe("record_transition", function()
    it("should create frecency entry for new transition", function()
      transitions.record_transition("/path/to/a.lua", "/path/to/b.lua")

      local data = mock_db.get_weights()
      assert.is_not_nil(data.transition_frecency)
      assert.is_not_nil(data.transition_frecency["/path/to/a.lua"])
      assert.is_number(data.transition_frecency["/path/to/a.lua"]["/path/to/b.lua"])
    end)

    it("should increase score with repeated visits", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")
      local data1 = mock_db.get_weights()
      local deadline1 = data1.transition_frecency["/path/a.lua"]["/path/b.lua"]

      transitions.record_transition("/path/a.lua", "/path/b.lua")
      local data2 = mock_db.get_weights()
      local deadline2 = data2.transition_frecency["/path/a.lua"]["/path/b.lua"]

      -- Second visit should have a later deadline (higher score)
      assert.is_true(deadline2 > deadline1)
    end)

    it("should track multiple destinations from same source", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")
      transitions.record_transition("/path/a.lua", "/path/c.lua")
      transitions.record_transition("/path/a.lua", "/path/d.lua")

      local data = mock_db.get_weights()
      local dests = data.transition_frecency["/path/a.lua"]
      assert.is_not_nil(dests["/path/b.lua"])
      assert.is_not_nil(dests["/path/c.lua"])
      assert.is_not_nil(dests["/path/d.lua"])
    end)

    it("should preserve existing weights data", function()
      mock_db.weights_data = {
        classic = { match = 100 },
        nn = { weights = { { { 1, 2, 3 } } } },
      }

      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local data = mock_db.get_weights()
      assert.equals(100, data.classic.match)
      assert.is_not_nil(data.nn.weights)
      assert.is_not_nil(data.transition_frecency)
    end)

    it("should remove legacy transition_history on record", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
        },
      }

      transitions.record_transition("/path/x.lua", "/path/y.lua")

      local data = mock_db.get_weights()
      assert.is_nil(data.transition_history)
      assert.is_not_nil(data.transition_frecency)
    end)
  end)

  describe("compute_scores_from", function()
    it("should return empty map when no transitions", function()
      local scores = transitions.compute_scores_from("/path/a.lua")
      assert.same({}, scores)
    end)

    it("should return empty map when no matching source", function()
      -- Record transition for different source
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local scores = transitions.compute_scores_from("/path/different.lua")
      assert.same({}, scores)
    end)

    it("should compute score for single visit", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local scores = transitions.compute_scores_from("/path/a.lua")
      -- Single visit: score=1, normalized=1-1/(1+1/4)=0.2
      assert.is_near(0.2, scores["/path/b.lua"], 0.001)
    end)

    it("should compute higher score for multiple visits", function()
      for _ = 1, 5 do
        transitions.record_transition("/path/a.lua", "/path/b.lua")
      end

      local scores = transitions.compute_scores_from("/path/a.lua")
      -- 5 visits: score=5, normalized=1-1/(1+5/4)≈0.556
      assert.is_near(0.556, scores["/path/b.lua"], 0.001)
    end)

    it("should compute scores for multiple destinations", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      for _ = 1, 5 do
        transitions.record_transition("/path/a.lua", "/path/c.lua")
      end

      local scores = transitions.compute_scores_from("/path/a.lua")

      -- b.lua: 2 visits → 1-1/(1+2/4)≈0.333
      assert.is_near(0.333, scores["/path/b.lua"], 0.001)
      -- c.lua: 5 visits → 1-1/(1+5/4)≈0.556
      assert.is_near(0.556, scores["/path/c.lua"], 0.001)
    end)

    it("should only return scores for specified source", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")
      transitions.record_transition("/path/a.lua", "/path/b.lua")
      transitions.record_transition("/path/other.lua", "/path/b.lua")
      transitions.record_transition("/path/other.lua", "/path/b.lua")

      local scores = transitions.compute_scores_from("/path/a.lua")
      -- Only 2 visits from a.lua → 1-1/(1+2/4)≈0.333
      assert.is_near(0.333, scores["/path/b.lua"], 0.001)
    end)

    it("should remove legacy transition_history on compute", function()
      mock_db.weights_data = {
        transition_history = {
          { from = "/path/a.lua", to = "/path/b.lua", timestamp = 123 },
        },
      }

      transitions.compute_scores_from("/path/a.lua")

      local data = mock_db.get_weights()
      assert.is_nil(data.transition_history)
    end)
  end)

  describe("time decay", function()
    it("should decay scores over time", function()
      -- Record a transition at current time
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local scores_now = transitions.compute_scores_from("/path/a.lua")

      -- Advance time by 30 days (one half-life)
      mock_time = mock_time + 30 * 24 * 3600

      local scores_later = transitions.compute_scores_from("/path/a.lua")

      -- Score should be lower after 30 days
      assert.is_true(scores_later["/path/b.lua"] < scores_now["/path/b.lua"])
    end)

    it("should halve raw score after one half-life", function()
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      -- Raw score is 1.0 right now
      -- After 30 days, raw score should be ~0.5
      mock_time = mock_time + 30 * 24 * 3600

      local scores = transitions.compute_scores_from("/path/a.lua")
      -- Normalized: 1-1/(1+0.5/4) ≈ 0.111
      assert.is_near(0.111, scores["/path/b.lua"], 0.01)
    end)

    it("should accumulate visits correctly with time", function()
      -- Visit at time 0
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      -- Visit again at same time - raw score = 2
      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local scores = transitions.compute_scores_from("/path/a.lua")
      -- 2 visits: 1-1/(1+2/4)≈0.333
      assert.is_near(0.333, scores["/path/b.lua"], 0.001)
    end)
  end)

  describe("pruning", function()
    it("should prune destinations when exceeding per-source limit", function()
      -- Temporarily reduce limit for testing
      local orig = transitions.MAX_DESTINATIONS_PER_SOURCE
      transitions.MAX_DESTINATIONS_PER_SOURCE = 3

      -- Record 5 different destinations from same source
      for i = 1, 5 do
        transitions.record_transition("/path/a.lua", "/path/d" .. i .. ".lua")
      end

      local data = mock_db.get_weights()
      local dests = data.transition_frecency["/path/a.lua"]
      local count = 0
      for _ in pairs(dests) do
        count = count + 1
      end
      assert.equals(3, count)

      transitions.MAX_DESTINATIONS_PER_SOURCE = orig
    end)

    it("should keep highest-scoring destinations when pruning", function()
      local orig = transitions.MAX_DESTINATIONS_PER_SOURCE
      transitions.MAX_DESTINATIONS_PER_SOURCE = 3

      -- Record d1 at current time, then advance time so d1 decays
      transitions.record_transition("/path/a.lua", "/path/d1.lua")
      mock_time = mock_time + 60 * 24 * 3600 -- 60 days later, d1 score ≈ 0.25

      -- Record d2 (3 visits, score=3) and d3 (1 visit, score=1) at new time
      for _ = 1, 3 do
        transitions.record_transition("/path/a.lua", "/path/d2.lua")
      end
      transitions.record_transition("/path/a.lua", "/path/d3.lua")

      -- Now reduce limit and add d4 to trigger pruning
      transitions.MAX_DESTINATIONS_PER_SOURCE = 3
      transitions.record_transition("/path/a.lua", "/path/d4.lua")

      local data = mock_db.get_weights()
      local dests = data.transition_frecency["/path/a.lua"]

      -- d2 (3 visits, fresh), d3 (1 visit, fresh), d4 (1 visit, fresh) should survive
      -- d1 (1 visit, decayed to ~0.25) should be pruned
      assert.is_not_nil(dests["/path/d2.lua"])
      assert.is_not_nil(dests["/path/d3.lua"])
      assert.is_not_nil(dests["/path/d4.lua"])
      assert.is_nil(dests["/path/d1.lua"])

      transitions.MAX_DESTINATIONS_PER_SOURCE = orig
    end)

    it("should prune sources when exceeding total source limit", function()
      local orig = transitions.MAX_SOURCES
      transitions.MAX_SOURCES = 2

      -- Record transitions from 3 different sources
      transitions.record_transition("/path/s1.lua", "/path/d.lua")
      transitions.record_transition("/path/s2.lua", "/path/d.lua")
      transitions.record_transition("/path/s3.lua", "/path/d.lua")

      local data = mock_db.get_weights()
      local source_count = 0
      for _ in pairs(data.transition_frecency) do
        source_count = source_count + 1
      end
      assert.equals(2, source_count)

      transitions.MAX_SOURCES = orig
    end)
  end)

  describe("edge cases", function()
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

    it("should handle nil weights data gracefully", function()
      mock_db.get_weights = function()
        return nil
      end

      transitions.record_transition("/path/a.lua", "/path/b.lua")

      local data = mock_db.weights_data
      assert.is_not_nil(data.transition_frecency)
    end)
  end)
end)
