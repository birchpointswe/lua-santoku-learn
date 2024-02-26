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

local function evaluate (t, ps, ss)
  local correct = 0
  for i = 1, #ps do
    if tm_predict(t, ps[i]) == ss[i] then
      correct = correct + 1
    end
  end
  return correct / #ps
end

return t_assign({
  train = train,
  evaluate = evaluate,
}, tm, false)
