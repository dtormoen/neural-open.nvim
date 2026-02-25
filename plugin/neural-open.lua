-- Plugin file for neural-open
-- Provides commands and mappings with lazy loading (no requires at load time)

if vim.g.loaded_neural_open then
  return
end
vim.g.loaded_neural_open = true

-- Main command with subcommands (lazy require - module loads only when invoked)
vim.api.nvim_create_user_command("NeuralOpen", function(args)
  require("neural-open").command(args)
end, {
  nargs = "*",
  complete = function(arg_lead, cmd_line, cursor_pos)
    return require("neural-open").complete(arg_lead, cmd_line, cursor_pos)
  end,
  desc = "Neural file picker with learning",
})

-- Plug mapping (lazy require - module loads only when triggered)
vim.keymap.set("n", "<Plug>(NeuralOpen)", function()
  require("neural-open").open()
end, { silent = true, desc = "Open NeuralOpen picker" })

-- Track buffer focus for persistent recency list (lazy-requires recent.lua directly)
vim.api.nvim_create_autocmd("BufEnter", {
  group = vim.api.nvim_create_augroup("NeuralOpenRecency", { clear = true }),
  callback = function()
    local buf = vim.api.nvim_get_current_buf()
    if not vim.api.nvim_buf_is_valid(buf) then
      return
    end
    if not vim.bo[buf].buflisted or vim.bo[buf].buftype ~= "" then
      return
    end
    local name = vim.api.nvim_buf_get_name(buf)
    if name == "" then
      return
    end
    require("neural-open.recent").record_buffer_focus(name)
  end,
  desc = "Track buffer focus for NeuralOpen recency",
})

-- Flush recency list to disk before exiting Neovim
vim.api.nvim_create_autocmd("VimLeavePre", {
  group = "NeuralOpenRecency",
  callback = function()
    require("neural-open.recent").flush()
  end,
  desc = "Persist NeuralOpen recency list on exit",
})
