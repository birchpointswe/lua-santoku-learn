local tm = require("santoku.tsetlin.capi")
local tm_evaluate = tm.evaluate

local tbl = require("santoku.table")
local t_merge = tbl.merge

local arr = require("santoku.array")
local a_sort = arr.sort

local it = require("santoku.iter")
local it_flatten = it.flatten
local it_map = it.map
local it_collect = it.collect
local it_pairs = it.pairs

local function evaluate (t, n, ps, ss, track_stats)

  local correct, confusion, predictions, observations =
    tm_evaluate(t, n, ps, ss, track_stats)

  local confusion_ranked = track_stats and a_sort(it_collect(it_flatten(it_map(function (e, ps)
    return it_map(function (p, c)
      return { expected = e, predicted = p, count = c, ratio = c / (n - correct) }
    end, it_pairs(ps))
  end, it_pairs(confusion)))), function (a, b)
    return a.ratio > b.ratio
  end)

  local predictions_ranked = track_stats and a_sort(it_collect(it_map(function (c, p)
    return {
      class = c,
      count = p, frequency_predicted = p / n,
      frequency_observed = observations[c] and (observations[c] / n) or 0
    }
  end, it_pairs(predictions))), function (a, b)
    return a.count > b.count
  end)

  return correct / n, confusion_ranked or nil, predictions_ranked or nil

end

return t_merge({
  evaluate = evaluate,
}, tm)
