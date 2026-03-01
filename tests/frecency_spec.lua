describe("frecency module", function()
  local helpers = require("tests.helpers")
  local frecency

  local original_os_time
  local mock_time

  before_each(function()
    helpers.setup()

    mock_time = 1000000000
    original_os_time = os.time
    os.time = function() -- luacheck: ignore 122
      return mock_time
    end

    package.loaded["neural-open.frecency"] = nil
    frecency = require("neural-open.frecency")
  end)

  after_each(function()
    os.time = original_os_time -- luacheck: ignore 122
    package.loaded["neural-open.frecency"] = nil
  end)

  describe("deadline_to_score / score_to_deadline", function()
    it("roundtrips correctly", function()
      local now = mock_time
      local deadline = now + 100000
      local score = frecency.deadline_to_score(deadline, now)
      local recovered = frecency.score_to_deadline(score, now)
      assert.is_near(deadline, recovered, 0.001)
    end)

    it("halves score after one half-life", function()
      local now = mock_time
      local deadline = now + 100000
      local score_now = frecency.deadline_to_score(deadline, now)
      local score_later = frecency.deadline_to_score(deadline, now + frecency.HALF_LIFE)
      assert.is_near(score_now / 2, score_later, score_now * 0.001)
    end)
  end)

  describe("bump", function()
    it("returns deadline for score=1 when given nil", function()
      local now = mock_time
      local result = frecency.bump(nil, now)
      local score = frecency.deadline_to_score(result, now)
      assert.is_near(1.0, score, 0.001)
    end)

    it("increments score by 1 from existing deadline", function()
      local now = mock_time
      local initial_deadline = frecency.score_to_deadline(5, now)
      local bumped_deadline = frecency.bump(initial_deadline, now)
      local bumped_score = frecency.deadline_to_score(bumped_deadline, now)
      assert.is_near(6.0, bumped_score, 0.001)
    end)
  end)

  describe("normalize_transition", function()
    it("returns expected value with divisor=4", function()
      assert.is_near(0.2, frecency.normalize_transition(1, 4), 0.001)
      assert.is_near(0.5556, frecency.normalize_transition(5, 4), 0.001)
    end)

    it("returns expected value with divisor=8", function()
      assert.is_near(0.1111, frecency.normalize_transition(1, 8), 0.001)
      assert.is_near(0.3846, frecency.normalize_transition(5, 8), 0.001)
    end)

    it("returns 0 for raw_score=0", function()
      assert.equals(0, frecency.normalize_transition(0, 4))
    end)
  end)

  describe("prune_map", function()
    it("keeps top N entries by score", function()
      local now = mock_time
      local map = {
        a = frecency.score_to_deadline(10, now),
        b = frecency.score_to_deadline(1, now),
        c = frecency.score_to_deadline(5, now),
        d = frecency.score_to_deadline(3, now),
        e = frecency.score_to_deadline(8, now),
      }

      frecency.prune_map(map, 3, now)

      assert.is_not_nil(map.a)
      assert.is_not_nil(map.c)
      assert.is_not_nil(map.e)
      assert.is_nil(map.b)
      assert.is_nil(map.d)
    end)

    it("is a no-op when count <= max", function()
      local now = mock_time
      local map = {
        a = frecency.score_to_deadline(10, now),
        b = frecency.score_to_deadline(5, now),
      }

      frecency.prune_map(map, 3, now)

      assert.is_not_nil(map.a)
      assert.is_not_nil(map.b)
    end)
  end)

  describe("prune_nested", function()
    it("keeps top N sources by total score", function()
      local now = mock_time
      local nested = {
        src_a = { d1 = frecency.score_to_deadline(10, now) },
        src_b = { d1 = frecency.score_to_deadline(1, now) },
        src_c = { d1 = frecency.score_to_deadline(5, now), d2 = frecency.score_to_deadline(3, now) },
        src_d = { d1 = frecency.score_to_deadline(2, now) },
        src_e = { d1 = frecency.score_to_deadline(9, now) },
      }

      frecency.prune_nested(nested, 3, now)

      -- Top 3 by total score: src_a(10), src_e(9), src_c(8)
      assert.is_not_nil(nested.src_a)
      assert.is_not_nil(nested.src_e)
      assert.is_not_nil(nested.src_c)
      assert.is_nil(nested.src_b)
      assert.is_nil(nested.src_d)
    end)

    it("is a no-op when count <= max", function()
      local now = mock_time
      local nested = {
        src_a = { d1 = frecency.score_to_deadline(10, now) },
        src_b = { d1 = frecency.score_to_deadline(5, now) },
      }

      frecency.prune_nested(nested, 3, now)

      assert.is_not_nil(nested.src_a)
      assert.is_not_nil(nested.src_b)
    end)
  end)
end)
