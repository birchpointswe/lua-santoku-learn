local tm = require("santoku.tsetlin.capi")

local function wrap (t)
  return {
    train = function (...)
      return tm.train(t, ...)
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
