local booleanizer = require("santoku.tsetlin.booleanizer")
local dvec = require("santoku.dvec")
local ivec = require("santoku.ivec")
local test = require("santoku.test")
-- local serialize = require("santoku.serialize")

test("booleanizer", function ()

  test("continuous", function ()
    local bzr = booleanizer.create({ n_thresholds = 1 })
    local data = dvec.create({ 1, 2, 3, 4, 5, 6, 7, 8, 9 })
    local dims = 3
    bzr:observe(data, dims)
    bzr:finalize()
    local bits = bzr:encode(data, dims) -- luacheck: ignore
    -- print(serialize(bits:table()))
    local top_v = ivec.create(2)
    top_v:fill_indices()
    bzr:restrict(top_v)
  end)

  test("entity-attribute-value", function ()
    local bzr = booleanizer.create({ n_thresholds = 2 })

    -- Observe: (feature_name, value)
    bzr:observe("title", "The Great Gatsby")
    bzr:observe("title", "1984")
    bzr:observe("author", "F. Scott Fitzgerald")
    bzr:observe("author", "George Orwell")
    bzr:observe("year", 1925)
    bzr:observe("year", 1949)
    bzr:observe("rating", 4.5)
    bzr:observe("rating", 4.8)

    bzr:finalize()

    -- Encode: (sample_id, feature_name, value, output_ivec)
    local bits1 = ivec.create()
    bzr:encode(0, "title", "The Great Gatsby", bits1)
    bzr:encode(0, "author", "F. Scott Fitzgerald", bits1)
    bzr:encode(0, "year", 1925, bits1)
    bzr:encode(0, "rating", 4.5, bits1)

    local bits2 = ivec.create()
    bzr:encode(1, "title", "1984", bits2)
    bzr:encode(1, "author", "George Orwell", bits2)
    bzr:encode(1, "year", 1949, bits2)
    bzr:encode(1, "rating", 4.8, bits2)

    assert(bits1:size() > 0)
    assert(bits2:size() > 0)
  end)

end)

