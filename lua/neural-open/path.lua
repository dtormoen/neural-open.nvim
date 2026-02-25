local M = {}

local _has_fs_normalize = (function()
  local ok, result = pcall(function()
    return vim.fs ~= nil and vim.fs.normalize ~= nil
  end)
  return ok and result
end)()
local _normalize_opts = { expand_env = false }

---@param path string
---@return string
function M.normalize(path)
  if _has_fs_normalize then
    return vim.fs.normalize(path, _normalize_opts)
  else
    return vim.fn.fnamemodify(path, ":p")
  end
end

return M
