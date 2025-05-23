local serialize = require("santoku.serialize") -- luacheck: ignore
local tm = require("santoku.tsetlin")
local mtx = require("santoku.matrix.integer")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local err = require("santoku.error")

local M = {}

M.read_snli_pairs = function (fp, max)
  max = max or num.huge
  local sentences = {}
  local pos = {}
  local neg = {}
  local ns = 0
  for line in it.drop(1, fs.lines(fp)) do
    local chunks = str.gmatch(line, "[^\t]+")
    local label = chunks()
    chunks = it.drop(4, chunks)
    local a = chunks()
    local b = chunks()
    err.assert(label and a and b, "unable to parse line", line)
    label = label == "entailment" and 1 or label == "contradiction" and 0 or nil
    if label then
      local na = sentences[a]
      local nb = sentences[b]
      if not na then
        ns = ns + 1
        na = ns
        sentences[a] = na
        sentences[na] = a
      end
      if not nb then
        ns = ns + 1
        nb = ns
        sentences[b] = nb
        sentences[nb] = b
      end
      if label == 1 then
        arr.push(pos, na, nb)
      else
        arr.push(neg, na, nb)
      end
      if (#pos + #neg) / 2 >= max then
        break
      end
    end
  end
  return {
    pos = pos,
    neg = neg,
    n_pos = #pos / 2,
    n_neg = #neg / 2,
    raw_sentences = sentences,
    n_sentences = ns,
  }
end

local function _split_snli_pairs (dataset, split, prop, start, size)
  local pairs_to = {}
  local pairs_from = dataset[prop]
  for i = start, start + size - 1 do
    local ia, ib =
      pairs_from[(i - start) * 2 + 1],
      pairs_from[(i - start) * 2 + 2]
    local sa, sb =
      dataset.raw_sentences[ia],
      dataset.raw_sentences[ib]
    local na, nb =
      split.raw_sentences[sa],
      split.raw_sentences[sb]
    if not na then
      split.n_sentences = split.n_sentences + 1
      na = split.n_sentences
      split.raw_sentences[na] = sa
      split.raw_sentences[sa] = na
    end
    if not nb then
      split.n_sentences = split.n_sentences + 1
      nb = split.n_sentences
      split.raw_sentences[nb] = sb
      split.raw_sentences[sb] = nb
    end
    arr.push(pairs_to, na - 1, nb - 1)
  end
  split[prop] = mtx.create(pairs_to)
end

M.split_snli_pairs = function (dataset, ratio)

  local test, train = {}, {}

  train.n_pos = num.floor(dataset.n_pos * ratio)
  train.n_neg = num.floor(dataset.n_neg * ratio)
  train.n_sentences = 0
  train.raw_sentences = {}

  test.n_pos = dataset.n_pos - train.n_pos
  test.n_neg = dataset.n_neg - train.n_neg
  test.n_sentences = 0
  test.raw_sentences = {}

  _split_snli_pairs(dataset, train, "pos", 1, train.n_pos)
  _split_snli_pairs(dataset, train, "neg", 1, train.n_neg)

  _split_snli_pairs(dataset, test, "pos", train.n_pos + 1, test.n_pos)
  _split_snli_pairs(dataset, test, "neg", train.n_neg + 1, test.n_neg)

  return train, test

end

M.read_binary_mnist = function (fp, n_features, max)
  local n_features_aligned = num.round(n_features, tm.align)
  local problems = {}
  local solutions = {}
  local bits = {}
  for l in fs.lines(fp) do
    arr.clear(bits)
    local n = 0
    for bit in str.gmatch(l, "%S+") do
      if n == n_features then
        local s = tonumber(bit)
        solutions[#solutions + 1] = s
        break
      end
      bit = bit == "1"
      if bit then
        arr.push(bits, n)
      end
      n = n + 1
    end
    if n ~= n_features then
      error("bitmap length mismatch")
    else
      local p = mtx.create(bits)
      mtx.reshape(p, mtx.values(p), 1);
      problems[#problems + 1] = p
    end
    if max and #problems >= max then
      break
    end
  end
  return {
    problems = problems,
    solutions = solutions,
    n_features = n_features_aligned
  }
end

local function _split_binary_mnist (dataset, s, e)
  local n = e - s + 1
  local ps = mtx.create(0, 1)
  local ss = {}
  for i = s, e do
    local p = mtx.create(dataset.problems[i])
    mtx.add(p, (i - s) * dataset.n_features)
    mtx.extend(ps, p)
    arr.push(ss, dataset.solutions[i])
  end
  ss = mtx.create(ss)
  return {
    problems = ps,
    solutions = ss,
    n = n
  }
end

M.split_binary_mnist = function (dataset, ratio)
  local n_train = num.floor(#dataset.problems * ratio)
  return
    _split_binary_mnist(dataset, 1, n_train),
    _split_binary_mnist(dataset, n_train + 1, #dataset.problems)
end

return M
