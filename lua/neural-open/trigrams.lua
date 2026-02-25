local M = {}

---@param text string
---@return table<string, boolean>
function M.compute_trigrams(text)
  local trigrams = {}

  if not text or text == "" then
    return trigrams
  end

  text = text:lower()
  local len = #text

  if len < 3 then
    return trigrams
  end

  for i = 1, len - 2 do
    local trigram = text:sub(i, i + 2)
    trigrams[trigram] = true
  end

  return trigrams
end

---@param trigrams1 table<string, boolean>
---@param trigrams2 table<string, boolean>
---@return number
function M.dice_coefficient(trigrams1, trigrams2)
  local size1 = 0
  local size2 = 0
  local intersection = 0

  for trigram in pairs(trigrams1) do
    size1 = size1 + 1
    if trigrams2[trigram] then
      intersection = intersection + 1
    end
  end

  for _ in pairs(trigrams2) do
    size2 = size2 + 1
  end

  if size1 == 0 and size2 == 0 then
    return 0
  end

  return (2 * intersection) / (size1 + size2)
end

return M
