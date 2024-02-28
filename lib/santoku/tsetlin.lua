local tm = require("santoku.tsetlin.capi")
local tm_update = tm.update
local tm_predict = tm.predict

local tbl = require("santoku.table")
local t_assign = tbl.assign

local function train (t, ps, ss, s)
  for i = 1, #ps do
    tm_update(t, ps[i], ss[i], s)
  end
end

local function evaluate (t, ps, ss, generate_confusion)
  local confusion = {}
  local correct = 0
  for i = 1, #ps do
    local guess = tm_predict(t, ps[i])
    if guess == ss[i] then
      correct = correct + 1
    elseif generate_confusion then
      confusion[ss[i]] = confusion[ss[i]] or {}
      confusion[ss[i]][guess] = (confusion[ss[i]][guess] or 0) + 1
    end
  end
  return correct / #ps, confusion
end

return t_assign({
  train = train,
  evaluate = evaluate,
}, tm, false)
