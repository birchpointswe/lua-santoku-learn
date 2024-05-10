local tm = require("santoku.tsetlin.capi")
local tm_type = tm.type
local tm_evaluate = tm.evaluate
local tm_create = tm.create

local tbl = require("santoku.table")
local t_merge = tbl.merge

local arr = require("santoku.array")
local a_sort = arr.sort

local it = require("santoku.iter")
local it_flatten = it.flatten
local it_map = it.map
local it_collect = it.collect
local it_pairs = it.pairs

local function evaluate (t, ...)

  if tm_type(t) == "classifier" then

    local n, ps, ss, track_stats = ...

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

  else

    return tm_evaluate(t, ...)

  end

end

local function classifier (...)
  return tm_create("classifier", ...)
end

local function recurrent_classifier (...)
  return tm_create("recurrent_classifier", ...)
end

local function encoder (...)
  return tm_create("encoder", ...)
end

local function recurrent_encoder (...)
  return tm_create("recurrent_encoder", ...)
end

local function auto_encoder (...)
  return tm_create("auto_encoder", ...)
end

local function regressor (...)
  return tm_create("regressor", ...)
end

local function recurrent_regressor (...)
  return tm_create("recurrent_regressor", ...)
end

return t_merge({

  evaluate = evaluate,

  classifier = classifier,
  recurrent_classifier = recurrent_classifier,

  encoder = encoder,
  recurrent_encoder = recurrent_encoder,
  auto_encoder = auto_encoder,

  regressor = regressor,
  recurrent_regressor = recurrent_regressor,

}, tm)
