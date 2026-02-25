describe("naive algorithm", function()
  local naive

  before_each(function()
    naive = require("neural-open.algorithms.naive")
  end)

  describe("calculate_score", function()
    -- Flat array order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition, not_current

    it("should sum all normalized features", function()
      local input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0.2, 0, 0, 1 }

      local score = naive.calculate_score(input_buf)
      assert.are.equal(2.8, score)
    end)

    it("should handle zero values", function()
      local input_buf = { 0.8, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 1 }

      local score = naive.calculate_score(input_buf)
      assert.are.equal(2.0, score)
    end)

    it("should handle all-zero features", function()
      local input_buf = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
      local score = naive.calculate_score(input_buf)
      assert.are.equal(0, score)
    end)
  end)

  describe("update_weights", function()
    it("should not modify anything (no learning)", function()
      -- Naive algorithm doesn't learn, so this should be a no-op
      local selected_item = { neural_rank = 5 }
      local ranked_items = {}

      -- Should not error
      naive.update_weights(selected_item, ranked_items)
    end)
  end)

  describe("debug_view", function()
    it("should return debug information", function()
      -- Flat order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition, not_current
      local item = {
        nos = {
          neural_score = 2.5,
          input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0, 1 },
        },
      }

      local lines = naive.debug_view(item)

      assert.is_table(lines)
      assert.is_true(#lines > 0)

      -- Check for expected content
      local content = table.concat(lines, "\n")
      assert.is_true(content:find("Naive Algorithm") ~= nil)
      assert.is_true(content:find("2.5") ~= nil) -- Score should be shown
    end)

    it("should display transition feature in debug output", function()
      -- Flat order: match, virtual_name, frecency, open, alt, proximity, project, recency, trigram, transition, not_current
      local item = {
        nos = {
          neural_score = 2.5,
          input_buf = { 0.8, 0, 0.5, 0, 0, 0.3, 0, 0, 0, 0.4, 1 },
        },
      }

      local lines = naive.debug_view(item)
      local content = table.concat(lines, "\n")

      -- Verify transition appears in the normalized features section
      assert.is_true(content:find("Transition") ~= nil, "Debug view should display transition feature")
      assert.is_true(
        content:find("0.4") ~= nil or content:find("0.4000") ~= nil,
        "Debug view should show transition value"
      )
    end)
  end)

  describe("get_name", function()
    it("should return algorithm name", function()
      assert.are.equal("naive", naive.get_name())
    end)
  end)
end)
