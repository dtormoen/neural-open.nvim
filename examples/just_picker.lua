-- Just Recipe Picker for neural-open.nvim
--
-- Opens a picker listing all Just recipes in the current project.
-- Neural-open learns which recipes you run most frequently and ranks them higher.
--
-- This demonstrates the generic pick() API: any source of string items can become
-- a neural-open picker. The picker name ("just_recipes") isolates learned weights
-- so usage patterns don't bleed between different pickers.
--
-- Requirements:
--   - just (https://github.com/casey/just) installed and on PATH
--   - A justfile in the current project
--   - toggleterm.nvim (optional, for TermExec command)
--
-- Usage:
--   require("examples.just_picker").pick()
--
-- Or register once and bind to a key:
--   require("examples.just_picker").register()
--   vim.keymap.set("n", "<leader>j", function()
--     require("neural-open").pick("just_recipes")
--   end)

local M = {}

--- Parse `just --list` output into picker items.
--- Each item needs `text` (display/search) and `value` (identity for tracking).
---@return table[] items
local function get_recipes()
  local handle = io.popen("just --list --unsorted 2>/dev/null")
  if not handle then
    return {}
  end

  local output = handle:read("*a")
  handle:close()

  local items = {}
  for line in output:gmatch("[^\n]+") do
    -- just --list output format: "    recipe-name # description" or "    recipe-name"
    local name, desc = line:match("^%s+([%w_-]+)%s+#%s+(.+)$")
    if not name then
      name = line:match("^%s+([%w_-]+)")
    end
    if name then
      items[#items + 1] = {
        text = name, -- Displayed in picker and used for fuzzy matching
        desc = desc or "", -- Optional description shown alongside
        value = name, -- Identity key for frecency/recency tracking
      }
    end
  end
  return items
end

-- Picker configuration: defined once, used by both register() and pick()
local picker_config = {
  title = "Just Recipes",
  finder = function()
    return get_recipes()
  end,
  -- Format: how each item appears in the picker list
  format = function(item)
    local ret = {}
    ret[#ret + 1] = { item.text, "Function" }
    if item.desc and item.desc ~= "" then
      ret[#ret + 1] = { " " }
      ret[#ret + 1] = { item.desc, "Comment" }
    end
    return ret
  end,
  -- Preview: shown in the preview pane when an item is focused
  preview = function(ctx)
    Snacks.picker.preview.cmd({ "just", "--show", ctx.item.text }, ctx, { ft = "just" })
  end,
  -- Confirm: action when an item is selected (neural-open records the selection automatically)
  confirm = function(picker, item)
    picker:close()
    vim.cmd(string.format('TermExec cmd="just %s"', item.value))
  end,
}

--- Register the just_recipes picker without opening it.
function M.register()
  require("neural-open").register_picker("just_recipes", picker_config)
end

--- Register (if needed) and open the just_recipes picker.
function M.pick()
  require("neural-open").pick("just_recipes", picker_config)
end

return M
