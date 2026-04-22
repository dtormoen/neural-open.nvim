local helpers = require("tests.helpers")

-- Mock context that mirrors the surface area of snacks.picker.finder.ctx used
-- by git_files: `clone(opts)` and `opts(opts)`. See
-- snacks.nvim/lua/snacks/picker/core/finder.lua (Ctx).
local function make_mock_ctx(picker_opts)
  local Ctx = {}
  Ctx.__index = Ctx

  function Ctx:clone(opts)
    return setmetatable({ _opts = opts }, { __index = self })
  end

  function Ctx:opts(opts)
    self._opts = setmetatable(opts or {}, { __index = self._opts or self.picker.opts })
    return self._opts
  end

  return setmetatable({
    picker = { opts = picker_opts or {} },
    meta = {},
  }, Ctx)
end

describe("git_files pin wrapper (M._pin_git_files)", function()
  local neural_open

  before_each(function()
    helpers.setup()
    helpers.clear_plugin_modules()
    neural_open = require("neural-open")
  end)

  after_each(function()
    helpers.clear_plugin_modules()
  end)

  it("passes cwd=git_root to the inner finder's opts", function()
    local captured_opts
    local inner = function(opts, _ctx)
      captured_opts = opts
      return {}
    end

    local wrapped = neural_open._pin_git_files(inner, "/repo")
    local ctx = make_mock_ctx({})
    wrapped({}, ctx)

    assert.equals("/repo", captured_opts.cwd)
  end)

  it("propagates cwd through the cloned ctx so ctx:opts() chain sees it", function()
    -- This simulates what snacks.picker.source.proc does: it calls
    -- `ctx:opts({ cmd=..., args=..., transform=... })` and then reads
    -- `opts.cwd` off the result. Without clone-based propagation, opts.cwd
    -- falls through to picker.opts.cwd (nil) and proc spawns in uv.cwd().
    local proc_seen_cwd
    local inner = function(_opts, ctx)
      -- Mimic how git.files hands a new table into ctx:opts and then reads
      -- the resulting opts.cwd down in proc.
      local proc_opts = ctx:opts({ cmd = "git", args = { "ls-files" } })
      proc_seen_cwd = proc_opts.cwd
      return {}
    end

    local wrapped = neural_open._pin_git_files(inner, "/repo")
    -- picker.opts.cwd is nil, as is the default when nvim is started from
    -- a subdirectory with no explicit picker cwd.
    local ctx = make_mock_ctx({})
    wrapped({}, ctx)

    assert.equals("/repo", proc_seen_cwd)
  end)

  it("prefers the pinned cwd over picker.opts.cwd", function()
    local proc_seen_cwd
    local inner = function(_opts, ctx)
      local proc_opts = ctx:opts({})
      proc_seen_cwd = proc_opts.cwd
      return {}
    end

    local wrapped = neural_open._pin_git_files(inner, "/repo")
    -- Even when picker already has a (different) cwd, the pin should win,
    -- because git_files must list relative to git_root.
    local ctx = make_mock_ctx({ cwd = "/somewhere/else" })
    wrapped({}, ctx)

    assert.equals("/repo", proc_seen_cwd)
  end)

  it("does not mutate the caller's opts table", function()
    local inner = function(_opts, _ctx)
      return {}
    end

    local wrapped = neural_open._pin_git_files(inner, "/repo")
    local outer_opts = { some_flag = true }
    wrapped(outer_opts, make_mock_ctx({}))

    assert.is_nil(outer_opts.cwd)
    assert.is_true(outer_opts.some_flag)
  end)

  it("end-to-end: git_files paths dedup against files-source paths in a subdir", function()
    -- Reproduces the bug this wrapper fixes. Scenario:
    --   git_root   = /repo
    --   uv.cwd()   = /repo/sub  (nvim launched from subdir)
    --   file A lives at /repo/sub/A
    --
    -- files source:   item.file="A",     item.cwd="/repo/sub"  -> /repo/sub/A
    -- git_files:      item.file="sub/A", item.cwd="/repo"      -> /repo/sub/A (dedup!)
    --
    -- If the wrapper fails to propagate cwd to proc, git_files would run
    -- ls-files from /repo/sub (yielding "A"), stamp item.cwd=/repo, and
    -- our transform would resolve to /repo/A -- a file that does not exist.
    local source = require("neural-open.source")
    local scorer = require("neural-open.scorer")
    local transform = source.create_neural_transform({ special_files = {} }, scorer, {})

    -- Mock git_files-like finder: reads cwd via ctx:opts({}) (like proc does)
    -- and emits items that would come from `git ls-files` run in that cwd.
    local fake_fs = {
      ["/repo"] = { "sub/A" },
      ["/repo/sub"] = { "A" },
    }
    local git_files_inner = function(_opts, ctx)
      local proc_opts = ctx:opts({ cmd = "git", args = { "ls-files" } })
      local emit_cwd = proc_opts.cwd
      local files = fake_fs[emit_cwd] or {}
      local items = {}
      for _, rel in ipairs(files) do
        items[#items + 1] = { file = rel, cwd = emit_cwd }
      end
      return items
    end

    local wrapped = neural_open._pin_git_files(git_files_inner, "/repo")
    -- Simulate: nvim launched from /repo/sub, picker.opts.cwd not set
    -- (the bug-trigger condition).
    local git_items = wrapped({}, make_mock_ctx({}))

    -- Files source: run from the subdir, emit A relative to /repo/sub
    local files_items = { { file = "A", cwd = "/repo/sub" } }

    local tx_ctx = { meta = {} }
    local results = {}
    for _, group in ipairs({ files_items, git_items }) do
      for _, item in ipairs(group) do
        local res = transform(item, tx_ctx)
        if res ~= false then
          results[#results + 1] = res._path
        end
      end
    end

    assert.equals(1, #results, "expected dedup to a single file, got: " .. vim.inspect(results))
    assert.equals("/repo/sub/A", results[1])
  end)
end)
