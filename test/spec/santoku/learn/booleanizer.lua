local booleanizer = require("santoku.learn.booleanizer")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
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
    local offsets = ivec.create()
    local neighbors = ivec.create()
    local dense = dvec.create()
    offsets:push(0)
    for s = 0, samples - 1 do
      for d = 0, dims - 1 do
        bzr:encode(d, data[s * dims + d + 1], neighbors, dense)
      end
      offsets:push(neighbors:size())
    end
    assert(offsets:size() == 4)
    assert(neighbors:size() == 0)
    assert(dense:size() == 9)
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
    local offsets = ivec.create()
    local neighbors = ivec.create()
    local dense = dvec.create()
    offsets:push(0)
    for s = 0, samples - 1 do
      for d = 0, dims - 1 do
        bzr:encode(d, data[s * dims + d + 1], neighbors, dense)
      end
      offsets:push(neighbors:size())
    end
    assert(offsets:size() == 4)
    assert(neighbors:size() == 3)
    assert(dense:size() == 6)
    local top_v = ivec.create(1)
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
    local offsets = ivec.create()
    local neighbors = ivec.create()
    local dense = dvec.create()
    offsets:push(0)
    bzr:encode("title", "The Great Gatsby", neighbors, dense)
    bzr:encode("author", "F. Scott Fitzgerald", neighbors, dense)
    bzr:encode("year", 1925, neighbors, dense)
    bzr:encode("rating", 4.5, neighbors, dense)
    offsets:push(neighbors:size())
    bzr:encode("title", "1984", neighbors, dense)
    bzr:encode("author", "George Orwell", neighbors, dense)
    bzr:encode("year", 1949, neighbors, dense)
    bzr:encode("rating", 4.8, neighbors, dense)
    offsets:push(neighbors:size())
    assert(neighbors:size() == 4)
    assert(dense:size() == 4)
  end)

end)
