-- Make Target Picker for neural-open.nvim
--
-- Opens a picker listing all Make targets in the current project.
-- Neural-open learns which targets you run most frequently and ranks them higher.
--
-- This is a minimal example: no format or preview — just finder and confirm.
-- Neural-open provides default formatting when none is specified.
--
-- Requirements:
--   - make installed and on PATH
--   - A Makefile in the current working directory
--   - toggleterm.nvim (optional, for TermExec command)
--
-- Usage:
--   require("examples.make_picker").pick()
--
-- Or register once and bind to a key:
--   require("examples.make_picker").register()
--   vim.keymap.set("n", "<leader>m", function()
--     require("neural-open").pick("make_targets")
--   end)

local M = {}

--- Parse Makefile targets into picker items.
---@return table[] items
local function get_targets()
  local makefile = vim.fn.getcwd() .. "/Makefile"
  local f = io.open(makefile, "r")
  if not f then
    return {}
  end

  local items = {}
  local seen = {}
  for line in f:lines() do
    -- Match target lines: "target-name:" (basic pattern — adapt for your Makefile conventions)
    local target = line:match("^([%w_%-]+)%s*:")
    if target and not target:match("^%.") and not seen[target] then
      seen[target] = true
      items[#items + 1] = {
        text = target,
        value = target,
      }
    end
  end
  f:close()
  return items
end

-- Picker configuration: defined once, used by both register() and pick()
local picker_config = {
  title = "Make Targets",
  finder = function()
    return get_targets()
  end,
  confirm = function(picker, item)
    picker:close()
    vim.cmd(string.format('TermExec cmd="make %s"', item.value))
  end,
}

--- Register the make_targets picker without opening it.
function M.register()
  require("neural-open").register_picker("make_targets", picker_config)
end

--- Register (if needed) and open the make_targets picker.
function M.pick()
  require("neural-open").pick("make_targets", picker_config)
end

return M
