local booleanizer = require("santoku.learn.booleanizer")
local test = require("santoku.test")

test("booleanizer", function ()

  test("continuous", function ()
    local bzr = booleanizer.create()
    local data = { 1, 2, 3, 4, 5, 6, 7, 8, 9 }
    local dims = 3
    local samples = #data / dims
    for s = 0, samples - 1 do
      for d = 0, dims - 1 do
        bzr:observe(d, data[s * dims + d + 1])
      end
    end
    bzr:finalize()
    local n_bits, n_dense = bzr:features()
    assert(n_bits == 0)
    assert(n_dense == 3)
    local rows = {}
    for s = 0, samples - 1 do
      local r = {}
      for d = 0, dims - 1 do r[d] = data[s * dims + d + 1] end
      rows[s + 1] = r
    end
    local bits, dense = bzr:encode({ samples = rows, cols = { 0, 1, 2 } })
    local nr, nc = bits:shape()
    assert(nr == 3)
    assert(nc == 0)
    assert(bits:neighbors():size() == 0)
    local dr, dc = dense:shape()
    assert(dr == 3)
    assert(dc == 3)
  end)

  test("mixed", function ()
    local bzr = booleanizer.create({ categorical = { 0 } })
    local data = { 1, 2, 3, 1, 5, 6, 2, 8, 9 }
    local dims = 3
    local samples = #data / dims
    for s = 0, samples - 1 do
      for d = 0, dims - 1 do
        bzr:observe(d, data[s * dims + d + 1])
      end
    end
    bzr:finalize()
    local n_bits, n_dense = bzr:features()
    assert(n_bits == 2)
    assert(n_dense == 2)
    local rows = {}
    for s = 0, samples - 1 do
      local r = {}
      for d = 0, dims - 1 do r[d] = data[s * dims + d + 1] end
      rows[s + 1] = r
    end
    local bits, dense = bzr:encode({ samples = rows, cols = { 0, 1, 2 } })
    assert(bits:neighbors():size() == 3)
    local dr, dc = dense:shape()
    assert(dr == 3)
    assert(dc == 2)
    local top_v = require("santoku.ivec").create(1)
    top_v:fill_indices()
    bzr:restrict(top_v)
    local n_bits2 = bzr:features()
    assert(n_bits2 == 1)
  end)

  test("entity-attribute-value", function ()
    local bzr = booleanizer.create()
    bzr:observe("title", "The Great Gatsby")
    bzr:observe("title", "1984")
    bzr:observe("author", "F. Scott Fitzgerald")
    bzr:observe("author", "George Orwell")
    bzr:observe("year", 1925)
    bzr:observe("year", 1949)
    bzr:observe("rating", 4.5)
    bzr:observe("rating", 4.8)
    bzr:finalize()
    local n_bits, n_dense = bzr:features()
    assert(n_bits == 4)
    assert(n_dense == 2)
    local cols = { "title", "author", "year", "rating" }
    local rows = {
      { title = "The Great Gatsby", author = "F. Scott Fitzgerald", year = 1925, rating = 4.5 },
      { title = "1984", author = "George Orwell", year = 1949, rating = 4.8 },
    }
    local bits, dense = bzr:encode({ samples = rows, cols = cols })
    assert(bits:neighbors():size() == 4)
    local dr, dc = dense:shape()
    assert(dr == 2)
    assert(dc == 2)
  end)

end)
