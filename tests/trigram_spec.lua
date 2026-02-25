describe("trigram module", function()
  local trigrams

  before_each(function()
    trigrams = require("neural-open.trigrams")
  end)

  describe("compute_trigrams", function()
    it("should return empty set for empty string", function()
      local result = trigrams.compute_trigrams("")
      assert.are.same({}, result)
    end)

    it("should return empty set for nil input", function()
      local result = trigrams.compute_trigrams(nil)
      assert.are.same({}, result)
    end)

    it("should return empty set for strings shorter than 3 chars", function()
      assert.are.same({}, trigrams.compute_trigrams("a"))
      assert.are.same({}, trigrams.compute_trigrams("ab"))
    end)

    it("should generate correct trigrams for 3-char string", function()
      local result = trigrams.compute_trigrams("abc")
      assert.are.same({ abc = true }, result)
    end)

    it("should generate correct trigrams for longer strings", function()
      local result = trigrams.compute_trigrams("hello")
      assert.are.same({
        hel = true,
        ell = true,
        llo = true,
      }, result)
    end)

    it("should be case insensitive", function()
      local result1 = trigrams.compute_trigrams("Hello")
      local result2 = trigrams.compute_trigrams("hello")
      assert.are.same(result1, result2)
    end)

    it("should handle duplicate trigrams", function()
      local result = trigrams.compute_trigrams("aaaa")
      assert.are.same({ aaa = true }, result)
    end)

    it("should handle special characters", function()
      local result = trigrams.compute_trigrams("test-file.js")
      assert.are.same({
        ["tes"] = true,
        ["est"] = true,
        ["st-"] = true,
        ["t-f"] = true,
        ["-fi"] = true,
        ["fil"] = true,
        ["ile"] = true,
        ["le."] = true,
        ["e.j"] = true,
        [".js"] = true,
      }, result)
    end)
  end)

  describe("dice_coefficient", function()
    it("should return 0 for empty sets", function()
      local result = trigrams.dice_coefficient({}, {})
      assert.are.equal(0, result)
    end)

    it("should return 0 for completely different sets", function()
      local set1 = { abc = true, bcd = true }
      local set2 = { xyz = true, yzw = true }
      local result = trigrams.dice_coefficient(set1, set2)
      assert.are.equal(0, result)
    end)

    it("should return 1 for identical sets", function()
      local set1 = { abc = true, bcd = true, cde = true }
      local set2 = { abc = true, bcd = true, cde = true }
      local result = trigrams.dice_coefficient(set1, set2)
      assert.are.equal(1, result)
    end)

    it("should calculate correct coefficient for partial overlap", function()
      local set1 = { abc = true, bcd = true, cde = true }
      local set2 = { bcd = true, cde = true, def = true }
      local result = trigrams.dice_coefficient(set1, set2)
      -- 2 * 2 / (3 + 3) = 4/6 = 0.666...
      assert.is_near(0.6666667, result, 0.0001)
    end)

    it("should handle asymmetric sets", function()
      local set1 = { abc = true }
      local set2 = { abc = true, bcd = true, cde = true }
      local result = trigrams.dice_coefficient(set1, set2)
      -- 2 * 1 / (1 + 3) = 2/4 = 0.5
      assert.are.equal(0.5, result)
    end)

    it("should handle one empty set", function()
      local set1 = { abc = true, bcd = true }
      local set2 = {}
      local result = trigrams.dice_coefficient(set1, set2)
      assert.are.equal(0, result)
    end)
  end)

  describe("integration with virtual names", function()
    it("should compute similarity between similar file names", function()
      local name1 = "user_controller.rb"
      local name2 = "user_service.rb"

      local trigrams1 = trigrams.compute_trigrams(name1)
      local trigrams2 = trigrams.compute_trigrams(name2)
      local similarity = trigrams.dice_coefficient(trigrams1, trigrams2)

      -- Should have some similarity due to "user" and ".rb" parts
      assert.is_true(similarity > 0.2)
      assert.is_true(similarity < 0.8)
    end)

    it("should compute high similarity for very similar names", function()
      local name1 = "test_helper.js"
      local name2 = "test_helpers.js"

      local trigrams1 = trigrams.compute_trigrams(name1)
      local trigrams2 = trigrams.compute_trigrams(name2)
      local similarity = trigrams.dice_coefficient(trigrams1, trigrams2)

      -- Should have high similarity
      assert.is_true(similarity > 0.7)
    end)

    it("should compute low similarity for different names", function()
      local name1 = "index.html"
      local name2 = "database.yml"

      local trigrams1 = trigrams.compute_trigrams(name1)
      local trigrams2 = trigrams.compute_trigrams(name2)
      local similarity = trigrams.dice_coefficient(trigrams1, trigrams2)

      -- Should have low similarity
      assert.is_true(similarity < 0.3)
    end)

    it("should handle virtual names with parent directories", function()
      local name1 = "components/index.js"
      local name2 = "helpers/index.js"

      local trigrams1 = trigrams.compute_trigrams(name1)
      local trigrams2 = trigrams.compute_trigrams(name2)
      local similarity = trigrams.dice_coefficient(trigrams1, trigrams2)

      -- Should have moderate similarity due to "index.js" part
      assert.is_true(similarity > 0.3)
      assert.is_true(similarity < 0.7)
    end)
  end)
end)
