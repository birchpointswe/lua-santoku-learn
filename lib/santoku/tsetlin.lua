local tm = require("santoku.tsetlin.capi")
local arr = require("santoku.array")
local it = require("santoku.iter")

local function evaluate (t, opts, ...)
  local typ = tm.type(t)
  if typ == "classifier" then
    local correct, confusion, predictions, observations =
      tm.evaluate(t, opts, ...)
    local confusion_ranked = opts.stats and arr.sort(it.collect(it.flatten(it.map(function (e, ps)
      return it.map(function (p, c)
        return { expected = e, predicted = p, count = c, ratio = c / (opts.samples - correct) }
      end, it.pairs(ps))
    end, it.pairs(confusion)))), function (a, b)
      return a.ratio > b.ratio
    end)
    local predictions_ranked = opts.stats and arr.sort(it.collect(it.map(function (c, p)
      return {
        class = c,
        count = p, frequency_predicted = p / opts.samples,
        frequency_observed = observations[c] and (observations[c] / opts.samples) or 0
      }
    end, it.pairs(predictions))), function (a, b)
      return a.count > b.count
    end)
    return correct / opts.samples, confusion_ranked or nil, predictions_ranked or nil
  elseif typ == "encoder" then
    return tm.evaluate(t, opts, ...) / opts.samples
  else
    return tm.evaluate(t, opts, ...)
  end
end

local function wrap (t)
  return {
    train = function (...)
      return tm.train(t, ...)
    end,
    evaluate = function (...)
      return evaluate(t, ...)
    end,
    predict = function (...)
      return tm.predict(t, ...)
    end,
    destroy = function (...)
      return tm.destroy(t, ...)
    end,
    persist = function (...)
      return tm.persist(t, ...)
    end,
    type = function (...)
      return tm.type(t, ...)
    end,
  }
end

return {
  load = function (...)
    return wrap(tm.load(...))
  end,
  classifier = function (...)
    return wrap(tm.create("classifier", ...))
  end,
  encoder = function (...)
    return wrap(tm.create("encoder", ...))
  end,
  align = tm.align
}
