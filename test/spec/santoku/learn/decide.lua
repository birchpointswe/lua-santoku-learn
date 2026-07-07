local test = require("santoku.test")
local err = require("santoku.error")
local assert = err.assert
local decide = require("santoku.learn.decide")
local fvec = require("santoku.fvec")
local ivec = require("santoku.ivec")
local csr = require("santoku.csr")

local function mkscores (rows, nl)
  local f = fvec.create(#rows * nl)
  for i = 1, #rows do for c = 1, nl do f:set((i - 1) * nl + (c - 1), rows[i][c]) end end
  return f
end

local function mkexpected (labels, nl)
  local off, nbr = { 0 }, {}
  for i = 1, #labels do nbr[i] = labels[i]; off[i + 1] = i end
  return csr.create({ offsets = ivec.create(off), neighbors = ivec.create(nbr), n_cols = nl })
end

local function argmax (r)
  local bi, bv = 0, -math.huge
  for c = 1, #r do if r[c] > bv then bv = r[c]; bi = c - 1 end end
  return bi
end

-- Single-label decode is plain argmax (micro-F1/accuracy-optimal, calibration-invariant).
test("decide single: predict == argmax", function ()
  local nl = 3
  local rows = { { 5, 0, 0 }, { 0.5, 0, 0.49 }, { 0, 1, 2 }, { 9, 8, 0 } }
  local g = decide.create({ n_labels = nl, single = true })
  local pred = g:predict({ scores = mkscores(rows, nl), n_samples = #rows })
  for i = 1, #rows do assert(pred:get(i - 1) == argmax(rows[i]), "row " .. i) end
end)

-- Perfectly separable: argmax gets everything right -> macro-F1 = accuracy = 1.
test("decide single: separable scores perfect", function ()
  local nl = 3
  local rows, labels = {}, {}
  for c = 0, nl - 1 do
    for _ = 1, 4 do
      local r = { 0, 0, 0 }; r[c + 1] = 6.0
      rows[#rows + 1] = r; labels[#labels + 1] = c
    end
  end
  local g = decide.create({ n_labels = nl, single = true })
  local macro, m = g:score({ scores = mkscores(rows, nl), n_samples = #rows,
    expected = mkexpected(labels, nl) })
  assert(macro > 0.999 and m.accuracy > 0.999, "expected perfect, got macro=" .. macro .. " acc=" .. m.accuracy)
end)

-- Persist/load round-trip: identical argmax decisions (v2 payload).
test("decide single: persist/load round-trip", function ()
  local nl = 3
  local rows = { { 5, 0, 0 }, { 0, 5, 0 }, { 0.50, 0, 0.49 }, { 1, 3, 2 } }
  local g = decide.create({ n_labels = nl, single = true })
  local l1 = g:predict({ scores = mkscores(rows, nl), n_samples = #rows })
  local path = os.tmpname()
  g:persist(path)
  local h = decide.load(path)
  os.remove(path)
  local l2 = h:predict({ scores = mkscores(rows, nl), n_samples = #rows })
  for i = 1, #rows do assert(l1:get(i - 1) == l2:get(i - 1), "row " .. i) end
end)
