-- Neovim Commands Picker for neural-open.nvim
--
-- Opens a picker listing all user-defined Neovim commands.
-- Neural-open learns which commands you run most frequently and ranks them higher.
--
-- This demonstrates using Neovim APIs as a data source. Any function that returns
-- a list of {text, value, desc} tables works as a neural-open finder.
--
-- Usage:
--   require("examples.vim_commands_picker").pick()
--
-- Or register once and bind to a key:
--   require("examples.vim_commands_picker").register()
--   vim.keymap.set("n", "<leader>:", function()
--     require("neural-open").pick("vim_commands")
--   end)

local M = {}

--- Collect user-defined Neovim commands into picker items.
---@return table[] items
local function get_commands()
  local cmds = vim.api.nvim_get_commands({})
  local items = {}
  for name, def in pairs(cmds) do
    items[#items + 1] = {
      text = name, -- Command name: displayed and fuzzy-matched
      desc = def.definition or "", -- Command definition shown as description
      value = name, -- Identity for frecency/recency tracking
    }
  end
  return items
end

-- Picker configuration: defined once, used by both register() and pick()
local picker_config = {
  title = "Commands",
  finder = function()
    return get_commands()
  end,
  format = function(item)
    local ret = {}
    ret[#ret + 1] = { item.text, "Function" }
    if item.desc and item.desc ~= "" then
      ret[#ret + 1] = { " " }
      ret[#ret + 1] = { item.desc, "Comment" }
    end
    return ret
  end,
  confirm = function(picker, item)
    picker:close()
    vim.cmd(item.value)
  end,
}

--- Register the vim_commands picker without opening it.
function M.register()
  require("neural-open").register_picker("vim_commands", picker_config)
end

--- Register (if needed) and open the vim_commands picker.
function M.pick()
  require("neural-open").pick("vim_commands", picker_config)
end

return M
