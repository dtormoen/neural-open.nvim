local M = {}

local string_byte = string.byte
local string_lower = string.lower

---@param text string
---@return table<number, boolean>
function M.compute_trigrams(text)
  local tris = {}

  if not text or text == "" then
    return tris
  end

  text = string_lower(text)
  local len = #text

  if len < 3 then
    return tris
  end

  -- Use packed integer keys (b1*65536 + b2*256 + b3) instead of string.sub()
  -- to avoid per-trigram string allocation while maintaining correct Dice coefficients.
  -- Slide a 3-byte window using byte arithmetic.
  local b1, b2 = string_byte(text, 1, 2)
  for i = 3, len do
    local b3 = string_byte(text, i)
    tris[b1 * 65536 + b2 * 256 + b3] = true
    b1, b2 = b2, b3
  end

  return tris
end

---@param trigrams1 table<number, boolean>
---@param trigrams2 table<number, boolean>
---@return number
function M.dice_coefficient(trigrams1, trigrams2)
  local size1 = 0
  local size2 = 0
  local intersection = 0

  for tri in pairs(trigrams1) do
    size1 = size1 + 1
    if trigrams2[tri] then
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

--- Decode a packed integer trigram key back to a 3-character string (for debug display)
---@param packed number Packed key (b1*65536 + b2*256 + b3)
---@return string 3-character trigram
function M.unpack_trigram(packed)
  local b1 = math.floor(packed / 65536)
  local b2 = math.floor((packed % 65536) / 256)
  local b3 = packed % 256
  return string.char(b1, b2, b3)
end

return M
