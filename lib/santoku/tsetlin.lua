local tm = require("santoku.tsetlin.capi")
local arr = require("santoku.array")
local it = require("santoku.iter")

local function evaluate (t, opts, ...)
  if tm.type(t) == "classifier" then
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
  else
    return tm.evaluate(t, opts, ...)
  end
end

local function wrap (t, threads)
  return {
    train = function (opts, ...)
      opts.threads = threads
      for i = 1, opts.iterations do
        tm.train(t, opts, ...)
        if opts.each then
          if opts.each(i) == false then
            break
          end
        end
      end
    end,
    evaluate = function (opts, ...)
      opts.threads = threads
      return evaluate(t, opts, ...)
    end,
    update = function (...)
      return tm.update(t, ...)
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
  load = function (opts, ...)
    return wrap(tm.load(opts, ...), opts.threads)
  end,
  classifier = function (opts, ...)
    return wrap(tm.create("classifier", opts, ...), opts.threads)
  end,
  encoder = function (opts, ...)
    return wrap(tm.create("encoder", opts, ...), opts.threads)
  end,
  auto_encoder = function (opts, ...)
    return wrap(tm.create("auto_encoder", opts, ...), opts.threads)
  end,
  regressor = function (opts, ...)
    return wrap(tm.create("regressor", opts, ...), opts.threads)
  end,
}
