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
      local result, count = trigrams.compute_trigrams("")
      assert.are.same({}, result)
      assert.equals(0, count)
    end)

    it("should return empty set for nil input", function()
      local result, count = trigrams.compute_trigrams(nil)
      assert.are.same({}, result)
      assert.equals(0, count)
    end)

    it("should return empty set for strings shorter than 3 chars", function()
      local result1, count1 = trigrams.compute_trigrams("a")
      assert.are.same({}, result1)
      assert.equals(0, count1)
      local result2, count2 = trigrams.compute_trigrams("ab")
      assert.are.same({}, result2)
      assert.equals(0, count2)
    end)

    it("should generate correct trigrams for 3-char string", function()
      local result, count = trigrams.compute_trigrams("abc")
      assert.are.same({ [pack("a", "b", "c")] = true }, result)
      assert.equals(1, count)
    end)

    it("should generate correct trigrams for longer strings", function()
      local result, count = trigrams.compute_trigrams("hello")
      assert.are.same({
        [pack("h", "e", "l")] = true,
        [pack("e", "l", "l")] = true,
        [pack("l", "l", "o")] = true,
      }, result)
      assert.equals(3, count)
    end)

    it("should be case insensitive", function()
      local result1 = trigrams.compute_trigrams("Hello")
      local result2 = trigrams.compute_trigrams("hello")
      assert.are.same(result1, result2)
    end)

    it("should handle duplicate trigrams", function()
      local result, count = trigrams.compute_trigrams("aaaa")
      assert.are.same({ [pack("a", "a", "a")] = true }, result)
      assert.equals(1, count)
    end)

    it("should produce correct count of trigrams", function()
      local _, count = trigrams.compute_trigrams("test-file.js")
      assert.equals(10, count) -- 12 chars - 2 = 10 unique trigrams
    end)
  end)

  describe("dice_coefficient_direct", function()
    it("should return 0 when trigrams1 is empty", function()
      assert.equals(0, trigrams.dice_coefficient_direct({}, 0, "hello"))
    end)

    it("should return 0 for nil text", function()
      local tris, size = trigrams.compute_trigrams("hello")
      assert.equals(0, trigrams.dice_coefficient_direct(tris, size, nil))
    end)

    it("should return 0 for empty text", function()
      local tris, size = trigrams.compute_trigrams("hello")
      assert.equals(0, trigrams.dice_coefficient_direct(tris, size, ""))
    end)

    it("should return 0 for text shorter than 3 chars", function()
      local tris, size = trigrams.compute_trigrams("hello")
      assert.equals(0, trigrams.dice_coefficient_direct(tris, size, "ab"))
    end)

    it("should return 1 for identical strings", function()
      local tris, size = trigrams.compute_trigrams("hello")
      assert.equals(1, trigrams.dice_coefficient_direct(tris, size, "hello"))
    end)

    it("should compute correct coefficient for different strings", function()
      local pairs_to_test = {
        { "test.lua", "test.js", 6 / 11 },
        { "user_controller.rb", "user_service.rb", 2 / 7 },
        { "index.js", "main.js", 2 / 11 },
        { "components/index.js", "helpers/index.js", 16 / 31 },
        { "Hello", "hello", 1 },
        { "test_helper.js", "test_helpers.js", 0.8 },
        { "index.html", "database.yml", 0 },
      }
      for _, pair in ipairs(pairs_to_test) do
        local tris1, size1 = trigrams.compute_trigrams(pair[1])
        local actual = trigrams.dice_coefficient_direct(tris1, size1, pair[2])
        assert.are.near(pair[3], actual, 1e-15, "mismatch for " .. pair[1] .. " vs " .. pair[2])
      end
    end)

    it("should be case insensitive", function()
      local tris, size = trigrams.compute_trigrams("hello")
      local lower_result = trigrams.dice_coefficient_direct(tris, size, "hello")
      local upper_result = trigrams.dice_coefficient_direct(tris, size, "HELLO")
      assert.equals(lower_result, upper_result)
    end)

    it("should handle duplicate trigrams in text2", function()
      local tris, size = trigrams.compute_trigrams("aaaa")
      assert.equals(1, trigrams.dice_coefficient_direct(tris, size, "aaaa"))
    end)
  end)

  describe("integration with virtual names", function()
    it("should compute similarity between similar file names", function()
      local tris1, size1 = trigrams.compute_trigrams("user_controller.rb")
      local similarity = trigrams.dice_coefficient_direct(tris1, size1, "user_service.rb")

      -- Should have some similarity due to "user" and ".rb" parts
      assert.is_true(similarity > 0.2)
      assert.is_true(similarity < 0.8)
    end)

    it("should compute high similarity for very similar names", function()
      local tris1, size1 = trigrams.compute_trigrams("test_helper.js")
      local similarity = trigrams.dice_coefficient_direct(tris1, size1, "test_helpers.js")

      -- Should have high similarity
      assert.is_true(similarity > 0.7)
    end)

    it("should compute low similarity for different names", function()
      local tris1, size1 = trigrams.compute_trigrams("index.html")
      local similarity = trigrams.dice_coefficient_direct(tris1, size1, "database.yml")

      -- Should have low similarity
      assert.is_true(similarity < 0.3)
    end)

    it("should handle virtual names with parent directories", function()
      local tris1, size1 = trigrams.compute_trigrams("components/index.js")
      local similarity = trigrams.dice_coefficient_direct(tris1, size1, "helpers/index.js")

      -- Should have moderate similarity due to "index.js" part
      assert.is_true(similarity > 0.3)
      assert.is_true(similarity < 0.7)
    end)
  end)
end)
