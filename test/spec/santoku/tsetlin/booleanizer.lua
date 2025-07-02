local booleanizer = require("santoku.tsetlin.booleanizer")
local dvec = require("santoku.dvec")
local test = require("santoku.test")
local serialize = require("santoku.serialize")

test("booleanizer", function ()

  test("continuous", function ()
    local bzr = booleanizer.create({ n_thresholds = 1 })
    local data = dvec.create({ 1, 2, 3, 4, 5, 6, 7, 8, 9 })
    local dims = 3
    bzr:observe(data, dims)
    bzr:finalize()
    local bits = bzr:encode(data, dims) -- luacheck: ignore
    -- print(serialize(bits:table()))
  end)

end)

