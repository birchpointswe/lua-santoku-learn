local tm = require("santoku.tsetlin.capi")
local tm_update = tm.update
local tm_predict = tm.predict

local tbl = require("santoku.table")
local t_assign = tbl.assign

local arr = require("santoku.array")
local a_sort = arr.sort

local it = require("santoku.iter")
local it_flatten = it.flatten
local it_map = it.map
local it_collect = it.collect
local it_pairs = it.pairs

local function train (t, ps, ss, s)
  for i = 1, #ps do
    tm_update(t, ps[i], ss[i], s)
  end
end

local function evaluate (t, ps, ss, track_stats)

  local confusion = track_stats and {}
  local predictions = track_stats and {}
  local observations = track_stats and {}
  local correct = 0

  for i = 1, #ps do
    local guess = tm_predict(t, ps[i])
    if track_stats then
      predictions[guess] = (predictions[guess] or 0) + 1
    end
    if track_stats then
      observations[ss[i]] = (observations[ss[i]] or 0) + 1
    end
    if guess == ss[i] then
      correct = correct + 1
    elseif track_stats then
      confusion[ss[i]] = confusion[ss[i]] or {}
      confusion[ss[i]][guess] = (confusion[ss[i]][guess] or 0) + 1
    end
  end

  local confusion_ranked = track_stats and a_sort(it_collect(it_flatten(it_map(function (e, ps)
    return it_map(function (p, c)
      return { expected = e, predicted = p, count = c, ratio = c / (#ss - correct) }
    end, it_pairs(ps))
  end, it_pairs(confusion)))), function (a, b)
    return a.ratio > b.ratio
  end)

  local predictions_ranked = track_stats and a_sort(it_collect(it_map(function (c, n)
    return {
      class = c,
      count = n, frequency_predicted = n / #ps,
      frequency_observed = observations[c] / #ps
    }
  end, it_pairs(predictions))), function (a, b)
    return a.count > b.count
  end)

  return correct / #ps, confusion_ranked or nil, predictions_ranked or nil

end

return t_assign({
  train = train,
  evaluate = evaluate,
}, tm, false)
