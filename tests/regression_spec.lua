--- Regression test for the scoring pipeline.
--- Ensures that transform → on_match_handler → algorithm.calculate_score
--- produces consistent, reproducible scores across commits.
---
--- Changes that break these tests either need a fix, or when scores
--- legitimately change, the update should be isolated in a separate
--- commit scoped to the smallest possible change so it's easy to
--- review what caused the score shift.
---
--- To re-capture expected scores after an intentional change:
---   REGRESSION_CAPTURE=1 just test tests/regression_spec.lua

local helpers = require("tests.helpers")
helpers.setup()

-- Load fixture
local fixture_path = "tests/fixtures/regression_test_cases.json"
local fixture_file = io.open(fixture_path, "r")
assert(fixture_file, "Failed to open fixture file: " .. fixture_path)
local fixture_json = fixture_file:read("*a")
fixture_file:close()
local fixture = vim.json.decode(fixture_json)

local TOLERANCE = fixture.tolerance
local CAPTURE_MODE = os.getenv("REGRESSION_CAPTURE") == "1"

--- Round to 6 decimal places to match capture precision and avoid float drift.
local function round6(x)
  return math.floor(x * 1e6 + 0.5) / 1e6
end

--- Write the fixture back to disk with updated expected scores.
local function save_fixture()
  local compact = vim.json.encode(fixture)
  local formatted = vim.fn.system({ "python3", "-m", "json.tool", "--indent", "2", "--sort-keys" }, compact)
  assert(vim.v.shell_error == 0, "python3 json formatting failed")
  local f = io.open(fixture_path, "w")
  assert(f, "Failed to open fixture file for writing: " .. fixture_path)
  f:write(formatted)
  f:close()
  print("Updated fixture file: " .. fixture_path)
end

--- Create a mock matcher that returns deterministic scores from a lookup table.
---@param query string
---@param match_score_lookup table<string, number> text → score
---@return table Mock matcher compatible with scorer.on_match_handler
local function create_mock_matcher(query, match_score_lookup)
  return {
    filter = { search = query },
    match = function(_self, mock_item)
      return match_score_lookup[mock_item.text] or 0
    end,
  }
end

--- Build the match_score_lookup from phase match_scores and items.
--- Maps item.file → match score, and item.nos.virtual_name → virtual_name score.
---@param items table[] Transformed items with nos.virtual_name populated
---@param phase_match_scores table<string, table> file → {match, virtual_name}
---@return table<string, number>
local function build_match_lookup(items, phase_match_scores)
  local lookup = {}
  for _, item in ipairs(items) do
    local scores = phase_match_scores[item.file]
    if scores then
      -- Map item.file (used as item.text) → match score
      lookup[item.file] = scores.match or 0
      -- Map virtual_name → virtual_name score only when it differs from file,
      -- otherwise the match score already covers both lookups (same text = same score)
      if item.nos and item.nos.virtual_name and item.nos.virtual_name ~= item.file and scores.virtual_name then
        lookup[item.nos.virtual_name] = scores.virtual_name
      end
    end
  end
  return lookup
end

local ALGORITHM_NAMES = { "nn", "classic", "naive" }

describe("scoring pipeline regression", function()
  local scorer, source, algorithms, neural_open

  if CAPTURE_MODE then
    teardown(function()
      save_fixture()
    end)
  end

  before_each(function()
    helpers.clear_plugin_modules()

    -- Mock db → returns empty so bundled NN defaults / config defaults are used
    package.loaded["neural-open.db"] = {
      get_weights = function()
        return {}
      end,
      save_weights = function() end,
    }

    -- Load real modules (including real weights module for classic defaults)
    neural_open = require("neural-open")
    neural_open.setup({ algorithm = "nn" })

    scorer = require("neural-open.scorer")
    source = require("neural-open.source")

    -- Initialize all algorithms
    algorithms = {}

    package.loaded["neural-open.algorithms.nn"] = nil
    local nn = require("neural-open.algorithms.nn")
    nn.init(neural_open.config.algorithm_config.nn)
    nn.load_weights()
    algorithms.nn = nn

    local classic = require("neural-open.algorithms.classic")
    classic.init(neural_open.config.algorithm_config.classic)
    classic.load_weights()
    algorithms.classic = classic

    local naive = require("neural-open.algorithms.naive")
    naive.init(neural_open.config.algorithm_config.naive or {})
    naive.load_weights()
    algorithms.naive = naive
  end)

  for _, test_case in ipairs(fixture.test_cases) do
    describe(test_case.name, function()
      local items
      local ctx_data

      before_each(function()
        -- Build the NosContext from fixture
        local trigrams = require("neural-open.trigrams")
        local current_file_virtual_name =
          scorer.get_virtual_name(test_case.context.current_file, neural_open.config.special_files)
        local current_file_trigrams = trigrams.compute_trigrams(current_file_virtual_name)

        -- Build recent_files map with {recent_rank = N} structure
        local recent_files = {}
        for path, rank in pairs(test_case.context.recent_files) do
          recent_files[path] = { recent_rank = rank }
        end

        ctx_data = {
          recent_files = recent_files,
          alternate_buf = test_case.context.alternate_buf,
          cwd = test_case.context.cwd,
          current_file = test_case.context.current_file,
          current_file_trigrams = current_file_trigrams,
          current_file_trigrams_size = current_file_trigrams and trigrams.count_trigrams(current_file_trigrams) or 0,
          current_file_virtual_name = current_file_virtual_name,
          algorithm = algorithms.nn,
          transition_scores = test_case.context.transition_scores or {},
        }

        -- Build a Snacks-like picker context
        local picker_ctx = {
          meta = {
            nos_ctx = ctx_data,
            done = {},
          },
        }

        -- Create the transform function
        local transform = source.create_neural_transform(neural_open.config, scorer, {})

        -- Transform each item
        items = {}
        for _, item_def in ipairs(test_case.items) do
          local item = {
            file = item_def.file,
            text = item_def.file,
            cwd = test_case.context.cwd,
            frecency = item_def.frecency or 0,
            idx = #items + 1,
            score = 0,
          }
          if item_def.buf then
            item.buf = item_def.buf
          end

          local result = transform(item, picker_ctx)
          if result and result ~= false then
            table.insert(items, result)
          end
        end
      end)

      it("produces expected normalized paths", function()
        for i, item_def in ipairs(test_case.items) do
          local item = items[i]
          assert.is_not_nil(item, "Item " .. i .. " (" .. item_def.file .. ") was filtered out by transform")
          assert.equals(
            item_def.expected_normalized_path,
            item.nos.normalized_path,
            "normalized_path mismatch for " .. item_def.file
          )
          assert.equals(item_def.expected_normalized_path, item._path, "_path mismatch for " .. item_def.file)
        end
      end)

      for _, phase in ipairs(test_case.phases) do
        for _, algo_name in ipairs(ALGORITHM_NAMES) do
          it(phase.name .. " [" .. algo_name .. "] produces expected scores", function()
            -- Swap the algorithm on the shared context
            ctx_data.algorithm = algorithms[algo_name]

            -- Build mock matcher from phase match_scores
            local match_lookup = build_match_lookup(items, phase.match_scores or {})
            local mock_matcher = create_mock_matcher(phase.query, match_lookup)

            -- Run on_match_handler for each item
            for _, item in ipairs(items) do
              scorer.on_match_handler(mock_matcher, item)
            end

            -- Ensure nested table exists for this algorithm
            phase.expected_scores[algo_name] = phase.expected_scores[algo_name] or {}
            local algo_expected = phase.expected_scores[algo_name]

            -- Check scores
            for _, item in ipairs(items) do
              local path = item.nos.normalized_path
              local expected = algo_expected[path]

              if CAPTURE_MODE then
                local rounded = round6(item.score)
                algo_expected[path] = rounded
                print(
                  string.format(
                    "CAPTURE [%s] [%s] [%s] %s = %.6f",
                    test_case.name,
                    phase.name,
                    algo_name,
                    path,
                    rounded
                  )
                )
              else
                assert.is_not_nil(
                  expected,
                  "No expected score for " .. path .. " [" .. algo_name .. "] in phase '" .. phase.name .. "'"
                )
                assert.near(
                  expected,
                  item.score,
                  TOLERANCE,
                  string.format(
                    "Score mismatch for %s [%s] in phase '%s': expected %.6f, got %.6f (diff %.6f)",
                    path,
                    algo_name,
                    phase.name,
                    expected,
                    item.score,
                    math.abs(expected - item.score)
                  )
                )
              end
            end
          end)
        end
      end
    end)
  end
end)
