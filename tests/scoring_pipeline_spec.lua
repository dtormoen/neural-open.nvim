local helpers = require("tests.helpers")

describe("scoring pipeline regression", function()
  local scorer
  local nn

  -- Feature order expected by nn.calculate_score (must match features_to_input)
  local FEATURE_ORDER =
    { "match", "virtual_name", "frecency", "open", "alt", "proximity", "project", "recency", "trigram", "transition" }

  before_each(function()
    helpers.clear_plugin_modules()

    -- Provide config so scorer.calculate_recency_score can read recency_list_size
    require("neural-open")
    package.loaded["neural-open"].config = helpers.get_default_config()

    scorer = require("neural-open.scorer")
    nn = require("neural-open.algorithms.nn")

    math.randomseed(42)
    local config = helpers.create_algorithm_config("nn", {
      architecture = { 10, 4, 1 },
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

    it("passes through open, alt, proximity, project, trigram, transition", function()
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
      }

      local norm = scorer.normalize_features(raw)
      assert.equals(1, norm.open)
      assert.equals(1, norm.alt)
      assert.equals(0.73, norm.proximity)
      assert.equals(1, norm.project)
      assert.equals(0.42, norm.trigram)
      assert.equals(0.88, norm.transition)
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
      local score1 = nn.calculate_score(norm)
      local score2 = nn.calculate_score(norm)

      assert.is_number(score1)
      assert.equals(score1, score2)
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
      local score = nn.calculate_score(norm)

      assert.is_true(score >= 0, "score should be >= 0, got " .. score)
      assert.is_true(score <= 100, "score should be <= 100, got " .. score)
    end)
  end)

  describe("nn_input equivalence", function()
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

      -- 10 features total
      assert.equals(10, #expected_input)
    end)
  end)

  describe("calculate_score_direct equivalence", function()
    it("produces identical score to normalize + calculate_score", function()
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

      -- Standard path
      local norm = scorer.normalize_features(raw)
      local standard_score = nn.calculate_score(norm)

      -- Fast path: build nn_input the same way source.lua does
      local nn_input = {
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
      }
      local direct_score = nn.calculate_score_direct(nn_input)

      assert.are.near(standard_score, direct_score, 1e-10)
    end)
  end)
end)
