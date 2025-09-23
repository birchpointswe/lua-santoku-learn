local booleanizer = require("santoku.tsetlin.booleanizer")
local dvec = require("santoku.dvec")
local ivec = require("santoku.ivec")
local test = require("santoku.test")


test("booleanizer", function ()

  test("continuous", function ()
    local bzr = booleanizer.create({ n_thresholds = 1 })
    local data = dvec.create({ 1, 2, 3, 4, 5, 6, 7, 8, 9 })
    local dims = 3
    bzr:observe(data, dims)
    bzr:finalize()
    local bits = bzr:encode(data, dims) -- luacheck: ignore

    local top_v = ivec.create(2)
    top_v:fill_indices()
    bzr:restrict(top_v)
  end)

end)

