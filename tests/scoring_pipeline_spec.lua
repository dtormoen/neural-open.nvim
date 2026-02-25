local helpers = require("tests.helpers")

describe("scoring pipeline regression", function()
  local scorer
  local nn

  -- Feature order must match scorer.FEATURE_NAMES
  local FEATURE_ORDER = {
    "match",
    "virtual_name",
    "frecency",
    "open",
    "alt",
    "proximity",
    "project",
    "recency",
    "trigram",
    "transition",
    "not_current",
  }

  --- Convert named normalized features to flat array in FEATURE_ORDER
  local function features_to_flat(norm)
    local buf = {}
    for i, name in ipairs(FEATURE_ORDER) do
      buf[i] = norm[name] or 0
    end
    return buf
  end

  before_each(function()
    helpers.clear_plugin_modules()

    -- Provide config so scorer.calculate_recency_score can read recency_list_size
    require("neural-open")
    package.loaded["neural-open"].config = helpers.get_default_config()

    scorer = require("neural-open.scorer")
    nn = require("neural-open.algorithms.nn")

    math.randomseed(42)
    local config = helpers.create_algorithm_config("nn", {
      architecture = { 11, 4, 1 },
      optimizer = "sgd",
      learning_rate = 0.1,
      batch_size = 4,
      history_size = 10,
      match_dropout = 0,
      warmup_steps = 0,
      dropout_rates = { 0 },
    })
    nn.init(config.algorithm_config.nn)
  end)

  describe("normalize_features formulas", function()
    it("normalizes match with sigmoid: 0->0, 100->0.5, 200->~0.88", function()
      local base = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0,
      }

      -- score=0 -> 0
      local norm = scorer.normalize_features(base)
      assert.equals(0, norm.match)

      -- score=100 -> 1/(1+exp(-0.02*100+2)) = 1/(1+exp(0)) = 0.5
      base.match = 100
      norm = scorer.normalize_features(base)
      assert.are.near(0.5, norm.match, 1e-10)

      -- score=200 -> 1/(1+exp(-0.02*200+2)) = 1/(1+exp(-2))
      base.match = 200
      norm = scorer.normalize_features(base)
      assert.are.near(1 / (1 + math.exp(-2)), norm.match, 1e-10)
    end)

    it("normalizes virtual_name with same sigmoid as match", function()
      local base = {
        match = 0,
        virtual_name = 100,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0,
      }

      local norm = scorer.normalize_features(base)
      assert.are.near(0.5, norm.virtual_name, 1e-10)
    end)

    it("normalizes frecency: 0->0, 8->0.5, large->~1", function()
      local base = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0,
      }

      local norm = scorer.normalize_features(base)
      assert.equals(0, norm.frecency)

      base.frecency = 8
      norm = scorer.normalize_features(base)
      assert.are.near(0.5, norm.frecency, 1e-10)

      base.frecency = 10000
      norm = scorer.normalize_features(base)
      assert.is_true(norm.frecency > 0.999)
    end)

    it("passes through open, alt, proximity, project, trigram, transition, not_current", function()
      local raw = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 1,
        alt = 1,
        proximity = 0.73,
        project = 1,
        recency = 0,
        trigram = 0.42,
        transition = 0.88,
        not_current = 1,
      }

      local norm = scorer.normalize_features(raw)
      assert.equals(1, norm.open)
      assert.equals(1, norm.alt)
      assert.equals(0.73, norm.proximity)
      assert.equals(1, norm.project)
      assert.equals(0.42, norm.trigram)
      assert.equals(0.88, norm.transition)
      assert.equals(1, norm.not_current)
    end)

    it("normalizes recency: rank=1->1.0, rank=50->0.51, rank=100->0.01, rank=0->0", function()
      local base = {
        match = 0,
        virtual_name = 0,
        frecency = 0,
        open = 0,
        alt = 0,
        proximity = 0,
        project = 0,
        recency = 0,
        trigram = 0,
        transition = 0,
      }

      -- rank=0 -> 0
      local norm = scorer.normalize_features(base)
      assert.equals(0, norm.recency)

      -- rank=1, max=100 -> (100 - 1 + 1) / 100 = 1.0
      base.recency = 1
      norm = scorer.normalize_features(base)
      assert.are.near(1.0, norm.recency, 1e-10)

      -- rank=50, max=100 -> (100 - 50 + 1) / 100 = 0.51
      base.recency = 50
      norm = scorer.normalize_features(base)
      assert.are.near(0.51, norm.recency, 1e-10)

      -- rank=100, max=100 -> (100 - 100 + 1) / 100 = 0.01
      base.recency = 100
      norm = scorer.normalize_features(base)
      assert.are.near(0.01, norm.recency, 1e-10)
    end)
  end)

  describe("normalize + nn.calculate_score pipeline", function()
    it("is deterministic: same raw_features produce same score", function()
      local raw = {
        match = 150,
        virtual_name = 80,
        frecency = 12,
        open = 1,
        alt = 0,
        proximity = 0.6,
        project = 1,
        recency = 5,
        trigram = 0.3,
        transition = 0.1,
      }

      local norm = scorer.normalize_features(raw)
      local input_buf = features_to_flat(norm)
      local score1 = nn.calculate_score(input_buf)
      local score2 = nn.calculate_score(input_buf)

      assert.is_number(score1)
      assert.near(score1, score2, 1e-10)
    end)

    it("produces score in [0, 100]", function()
      local raw = {
        match = 150,
        virtual_name = 80,
        frecency = 12,
        open = 1,
        alt = 0,
        proximity = 0.6,
        project = 1,
        recency = 5,
        trigram = 0.3,
        transition = 0.1,
      }

      local norm = scorer.normalize_features(raw)
      local input_buf = features_to_flat(norm)
      local score = nn.calculate_score(input_buf)

      assert.is_true(score >= 0, "score should be >= 0, got " .. score)
      assert.is_true(score <= 100, "score should be <= 100, got " .. score)
    end)
  end)

  describe("input_buf equivalence", function()
    it("flat array indices match named normalized_features in FEATURE_ORDER", function()
      local raw = {
        match = 120,
        virtual_name = 60,
        frecency = 20,
        open = 1,
        alt = 0,
        proximity = 0.45,
        project = 1,
        recency = 10,
        trigram = 0.55,
        transition = 0.2,
      }

      local norm = scorer.normalize_features(raw)

      -- Build the flat array in the same order calculate_score fills input_buf
      local expected_input = {}
      for i, name in ipairs(FEATURE_ORDER) do
        expected_input[i] = norm[name] or 0
      end

      -- Verify each position matches the named field
      assert.are.near(norm.match, expected_input[1], 1e-15)
      assert.are.near(norm.virtual_name, expected_input[2], 1e-15)
      assert.are.near(norm.frecency, expected_input[3], 1e-15)
      assert.equals(norm.open, expected_input[4])
      assert.equals(norm.alt, expected_input[5])
      assert.equals(norm.proximity, expected_input[6])
      assert.equals(norm.project, expected_input[7])
      assert.are.near(norm.recency, expected_input[8], 1e-15)
      assert.equals(norm.trigram, expected_input[9])
      assert.equals(norm.transition, expected_input[10])
      assert.equals(norm.not_current, expected_input[11])

      -- 11 features total
      assert.equals(11, #expected_input)
    end)
  end)

  describe("calculate_score with flat array", function()
    it("produces consistent scores from normalized features", function()
      local raw = {
        match = 150,
        virtual_name = 80,
        frecency = 12,
        open = 1,
        alt = 0,
        proximity = 0.6,
        project = 1,
        recency = 5,
        trigram = 0.3,
        transition = 0.1,
      }

      -- Build input_buf from normalized features
      local norm = scorer.normalize_features(raw)
      local input_buf = {
        norm.match,
        norm.virtual_name,
        norm.frecency,
        norm.open,
        norm.alt,
        norm.proximity,
        norm.project,
        norm.recency,
        norm.trigram,
        norm.transition,
        norm.not_current,
      }
      local score = nn.calculate_score(input_buf)

      -- Build the same input_buf via the helper and verify identical result
      local input_buf2 = features_to_flat(norm)
      local score2 = nn.calculate_score(input_buf2)

      assert.are.near(score, score2, 1e-10)
    end)

    it("equivalence holds across diverse feature vectors", function()
      local test_vectors = {
        {
          match = 0,
          virtual_name = 0,
          frecency = 0,
          open = 0,
          alt = 0,
          proximity = 0,
          project = 0,
          recency = 0,
          trigram = 0,
          transition = 0,
        },
        {
          match = 300,
          virtual_name = 200,
          frecency = 100,
          open = 1,
          alt = 1,
          proximity = 1,
          project = 1,
          recency = 1,
          trigram = 1,
          transition = 1,
          not_current = 1,
        },
        {
          match = 50,
          virtual_name = 0,
          frecency = 4,
          open = 1,
          alt = 0,
          proximity = 0.3,
          project = 1,
          recency = 25,
          trigram = 0.8,
          transition = 0.5,
          not_current = 1,
        },
        {
          match = 1,
          virtual_name = 300,
          frecency = 1,
          open = 0,
          alt = 1,
          proximity = 0,
          project = 0,
          recency = 100,
          trigram = 0,
          transition = 0.1,
          not_current = 1,
        },
      }

      for i, raw in ipairs(test_vectors) do
        local norm = scorer.normalize_features(raw)
        local input_buf = features_to_flat(norm)
        local score = nn.calculate_score(input_buf)

        -- Re-score with same input to verify determinism
        local score2 = nn.calculate_score(input_buf)

        assert.are.near(score, score2, 1e-10, "non-deterministic on vector " .. i)
        assert.is_true(score >= 0 and score <= 100, "score out of range on vector " .. i)
      end
    end)
  end)

  describe("normalize_match_score regression", function()
    it("returns exact values for known inputs", function()
      -- sigmoid: raw > 0 ? 1/(1+exp(-0.02*raw+2)) : 0
      assert.equals(0, scorer.normalize_match_score(0))
      assert.equals(0, scorer.normalize_match_score(-5))
      assert.are.near(0.5, scorer.normalize_match_score(100), 1e-15)
      assert.are.near(1 / (1 + math.exp(-0.02 * 50 + 2)), scorer.normalize_match_score(50), 1e-15)
      assert.are.near(1 / (1 + math.exp(-0.02 * 150 + 2)), scorer.normalize_match_score(150), 1e-15)
      assert.are.near(1 / (1 + math.exp(-0.02 * 200 + 2)), scorer.normalize_match_score(200), 1e-15)
      assert.are.near(1 / (1 + math.exp(-0.02 * 300 + 2)), scorer.normalize_match_score(300), 1e-15)
      assert.are.near(1 / (1 + math.exp(-0.02 * 1 + 2)), scorer.normalize_match_score(1), 1e-15)
    end)
  end)

  describe("normalize_frecency regression", function()
    it("returns exact values for known inputs", function()
      -- formula: raw > 0 ? 1 - 1/(1+raw/8) : 0
      assert.equals(0, scorer.normalize_frecency(0))
      assert.equals(0, scorer.normalize_frecency(-5))
      assert.are.near(0.5, scorer.normalize_frecency(8), 1e-15)
      assert.are.near(1 / 9, scorer.normalize_frecency(1), 1e-15)
      assert.are.near(1 / 3, scorer.normalize_frecency(4), 1e-15)
      assert.are.near(2 / 3, scorer.normalize_frecency(16), 1e-15)
      assert.are.near(100 / 108, scorer.normalize_frecency(100), 1e-15)
    end)
  end)

  describe("on_match_handler unified input_buf path", function()
    it("produces identical scores whether built from normalize_features or inline normalization", function()
      local raw = {
        match = 120,
        virtual_name = 60,
        frecency = 15,
        open = 1,
        alt = 0,
        proximity = 0.7,
        project = 1,
        recency = 3,
        trigram = 0.45,
        transition = 0.2,
      }

      -- Path 1: normalize_features → features_to_flat → calculate_score
      local norm = scorer.normalize_features(raw)
      local input_buf_from_norm = features_to_flat(norm)
      local score_from_norm = nn.calculate_score(input_buf_from_norm)

      -- Path 2: simulate on_match_handler - pre-fill static features at transform time,
      -- then update dynamic slots [1..3] per keystroke using scorer helpers
      local input_buf = {
        0, -- [1] match   (dynamic, updated per keystroke)
        0, -- [2] virtual_name (dynamic, updated per keystroke)
        0, -- [3] frecency (dynamic, updated per keystroke)
        norm.open,
        norm.alt,
        norm.proximity,
        norm.project,
        norm.recency,
        norm.trigram,
        norm.transition,
        norm.not_current,
      }
      input_buf[1] = scorer.normalize_match_score(raw.match)
      input_buf[2] = scorer.normalize_match_score(raw.virtual_name)
      input_buf[3] = scorer.normalize_frecency(raw.frecency)

      local score_from_buf = nn.calculate_score(input_buf)

      assert.are.near(score_from_norm, score_from_buf, 1e-10)
    end)
  end)
end)
