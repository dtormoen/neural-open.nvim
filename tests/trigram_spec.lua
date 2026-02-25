-- Pack 3 bytes into a single integer key (matches trigrams.lua encoding)
local function pack(a, b, c)
  return string.byte(a) * 65536 + string.byte(b) * 256 + string.byte(c)
end

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
      assert.are.same({ [pack("a", "b", "c")] = true }, result)
    end)

    it("should generate correct trigrams for longer strings", function()
      local result = trigrams.compute_trigrams("hello")
      assert.are.same({
        [pack("h", "e", "l")] = true,
        [pack("e", "l", "l")] = true,
        [pack("l", "l", "o")] = true,
      }, result)
    end)

    it("should be case insensitive", function()
      local result1 = trigrams.compute_trigrams("Hello")
      local result2 = trigrams.compute_trigrams("hello")
      assert.are.same(result1, result2)
    end)

    it("should handle duplicate trigrams", function()
      local result = trigrams.compute_trigrams("aaaa")
      assert.are.same({ [pack("a", "a", "a")] = true }, result)
    end)

    it("should produce correct count of trigrams", function()
      local result = trigrams.compute_trigrams("test-file.js")
      local count = 0
      for _ in pairs(result) do
        count = count + 1
      end
      assert.equals(10, count) -- 12 chars - 2 = 10 trigrams
    end)
  end)

  describe("dice_coefficient", function()
    it("should return 0 for empty sets", function()
      local result = trigrams.dice_coefficient({}, {})
      assert.are.equal(0, result)
    end)

    it("should return 0 for completely different sets", function()
      local set1 = { [pack("a", "b", "c")] = true, [pack("b", "c", "d")] = true }
      local set2 = { [pack("x", "y", "z")] = true, [pack("y", "z", "w")] = true }
      local result = trigrams.dice_coefficient(set1, set2)
      assert.are.equal(0, result)
    end)

    it("should return 1 for identical sets", function()
      local set1 = { [pack("a", "b", "c")] = true, [pack("b", "c", "d")] = true, [pack("c", "d", "e")] = true }
      local set2 = { [pack("a", "b", "c")] = true, [pack("b", "c", "d")] = true, [pack("c", "d", "e")] = true }
      local result = trigrams.dice_coefficient(set1, set2)
      assert.are.equal(1, result)
    end)

    it("should calculate correct coefficient for partial overlap", function()
      local set1 = { [pack("a", "b", "c")] = true, [pack("b", "c", "d")] = true, [pack("c", "d", "e")] = true }
      local set2 = { [pack("b", "c", "d")] = true, [pack("c", "d", "e")] = true, [pack("d", "e", "f")] = true }
      local result = trigrams.dice_coefficient(set1, set2)
      -- 2 * 2 / (3 + 3) = 4/6 = 0.666...
      assert.is_near(0.6666667, result, 0.0001)
    end)

    it("should handle asymmetric sets", function()
      local set1 = { [pack("a", "b", "c")] = true }
      local set2 = { [pack("a", "b", "c")] = true, [pack("b", "c", "d")] = true, [pack("c", "d", "e")] = true }
      local result = trigrams.dice_coefficient(set1, set2)
      -- 2 * 1 / (1 + 3) = 2/4 = 0.5
      assert.are.equal(0.5, result)
    end)

    it("should handle one empty set", function()
      local set1 = { [pack("a", "b", "c")] = true, [pack("b", "c", "d")] = true }
      local set2 = {}
      local result = trigrams.dice_coefficient(set1, set2)
      assert.are.equal(0, result)
    end)
  end)

  describe("count_trigrams", function()
    it("should return 0 for empty table", function()
      assert.equals(0, trigrams.count_trigrams({}))
    end)

    it("should count entries correctly", function()
      local tris = trigrams.compute_trigrams("hello")
      assert.equals(3, trigrams.count_trigrams(tris))
    end)

    it("should count entries for longer strings", function()
      local tris = trigrams.compute_trigrams("test-file.js")
      assert.equals(10, trigrams.count_trigrams(tris))
    end)
  end)

  describe("dice_coefficient_direct", function()
    it("should return 0 when trigrams1 is empty", function()
      assert.equals(0, trigrams.dice_coefficient_direct({}, 0, "hello"))
    end)

    it("should return 0 for nil text", function()
      local tris = trigrams.compute_trigrams("hello")
      assert.equals(0, trigrams.dice_coefficient_direct(tris, 3, nil))
    end)

    it("should return 0 for empty text", function()
      local tris = trigrams.compute_trigrams("hello")
      assert.equals(0, trigrams.dice_coefficient_direct(tris, 3, ""))
    end)

    it("should return 0 for text shorter than 3 chars", function()
      local tris = trigrams.compute_trigrams("hello")
      assert.equals(0, trigrams.dice_coefficient_direct(tris, 3, "ab"))
    end)

    it("should match dice_coefficient for identical strings", function()
      local text = "hello"
      local tris = trigrams.compute_trigrams(text)
      local size = trigrams.count_trigrams(tris)
      local expected = trigrams.dice_coefficient(tris, trigrams.compute_trigrams(text))
      assert.equals(expected, trigrams.dice_coefficient_direct(tris, size, text))
    end)

    it("should match dice_coefficient for different strings", function()
      local pairs_to_test = {
        { "test.lua", "test.js" },
        { "user_controller.rb", "user_service.rb" },
        { "index.js", "main.js" },
        { "components/index.js", "helpers/index.js" },
        { "Hello", "hello" },
        { "test_helper.js", "test_helpers.js" },
        { "index.html", "database.yml" },
      }
      for _, pair in ipairs(pairs_to_test) do
        local tris1 = trigrams.compute_trigrams(pair[1])
        local size1 = trigrams.count_trigrams(tris1)
        local tris2 = trigrams.compute_trigrams(pair[2])
        local expected = trigrams.dice_coefficient(tris1, tris2)
        local actual = trigrams.dice_coefficient_direct(tris1, size1, pair[2])
        assert.are.near(expected, actual, 1e-15, "mismatch for " .. pair[1] .. " vs " .. pair[2])
      end
    end)

    it("should be case insensitive", function()
      local tris = trigrams.compute_trigrams("hello")
      local size = trigrams.count_trigrams(tris)
      local lower_result = trigrams.dice_coefficient_direct(tris, size, "hello")
      local upper_result = trigrams.dice_coefficient_direct(tris, size, "HELLO")
      assert.equals(lower_result, upper_result)
    end)

    it("should handle duplicate trigrams in text2", function()
      local tris = trigrams.compute_trigrams("aaaa")
      local size = trigrams.count_trigrams(tris)
      local expected = trigrams.dice_coefficient(tris, trigrams.compute_trigrams("aaaa"))
      assert.equals(expected, trigrams.dice_coefficient_direct(tris, size, "aaaa"))
    end)
  end)

  describe("dice_coefficient regression", function()
    -- Pin exact dice_coefficient values for the full compute_trigrams -> dice_coefficient pipeline.
    -- These must remain identical after any internal representation changes (e.g. byte-based keys).

    local function dice(a, b)
      return trigrams.dice_coefficient(trigrams.compute_trigrams(a), trigrams.compute_trigrams(b))
    end

    it("returns 1 for identical strings", function()
      assert.equals(1, dice("hello", "hello"))
      assert.equals(1, dice("abc", "abc"))
    end)

    it("returns 0 for strings with no shared trigrams", function()
      assert.equals(0, dice("foo", "bar"))
    end)

    it("returns 0 for strings shorter than 3 characters", function()
      assert.equals(0, dice("ab", "ab"))
      assert.equals(0, dice("a", "a"))
      assert.equals(0, dice("", ""))
    end)

    it("computes exact scores for file name pairs", function()
      -- 6/11 shared trigrams: "est", ".lu"/".js" differ, but "tes", "est" overlap
      assert.are.near(6 / 11, dice("test.lua", "test.js"), 1e-15)

      -- user_controller.rb vs user_service.rb: 2/7
      assert.are.near(2 / 7, dice("user_controller.rb", "user_service.rb"), 1e-15)

      -- index.js vs main.js: share ".js" and partial overlap
      assert.are.near(2 / 11, dice("index.js", "main.js"), 1e-15)

      -- components/index.js vs helpers/index.js: shared "index.js" portion
      assert.are.near(16 / 31, dice("components/index.js", "helpers/index.js"), 1e-15)
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
