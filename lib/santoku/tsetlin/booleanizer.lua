local arr = require("santoku.array")
local it = require("santoku.iter")
local num = require("santoku.num")

local function thresholds (observations, samples, bit)
  local thresholds = {}
  for o in it.pairs(observations) do
    arr.push(thresholds, o)
  end
  arr.sort(thresholds)
  bit = bit or 1
  local n = 1
  local step = num.floor(#thresholds / samples)
  for i = 1, #thresholds, step do
    thresholds[n] = { value = thresholds[i], bit = bit }
    n = n + 1
    bit = bit + 1
  end
  arr.clear(thresholds, n)
  return thresholds, bit - 1
end

return {
  thresholds = thresholds,
}
