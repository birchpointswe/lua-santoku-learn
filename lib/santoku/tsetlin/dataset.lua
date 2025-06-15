require("santoku.rvec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local tbl = require("santoku.table")
local tm = require("santoku.tsetlin")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local err = require("santoku.error")

local M = {}

M.read_snli_pairs = function (fp, max, single_split)
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
  local ds = {
    pos = pos,
    neg = neg,
    n_pos = #pos / 2,
    n_neg = #neg / 2,
    raw_sentences = sentences,
    n_sentences = ns,
  }
  if single_split then
    return M.split_snli_pairs(ds)
  else
    return ds
  end
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
  split[prop] = ivec.create(pairs_to)
end

M.split_snli_pairs = function (dataset, ratio)

  if not ratio then
    local train = tbl.merge({}, dataset)
    train.n_pos = dataset.n_pos
    train.n_neg = dataset.n_neg
    train.n_sentences = 0
    train.raw_sentences = {}
    _split_snli_pairs(dataset, train, "pos", 1, train.n_pos)
    _split_snli_pairs(dataset, train, "neg", 1, train.n_neg)
    return train
  end

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
  for l in fs.lines(fp) do
    local bits = {}
    local n = 0
    for bit in str.gmatch(l, "%S+") do
      if n == n_features then
        local s = tonumber(bit)
        solutions[#solutions + 1] = s
        break
      end
      bit = bit == "1"
      if bit then
        bits[#bits + 1] = n
      end
      n = n + 1
    end
    if n ~= n_features then
      error("bitmap length mismatch")
    else
      local p = ivec.create(bits)
      problems[#problems + 1] = p
    end
    if max and #problems >= max then
      break
    end
  end
  arr.shuffle(problems, solutions)
  return {
    problems = problems,
    solutions = solutions,
    n_features = n_features_aligned
  }
end

local function _split_binary_mnist (dataset, s, e)
  local n = e - s + 1
  local ps = ivec.create()
  local ss = {}
  for i = s, e do
    local p = ivec.create(dataset.problems[i])
    p:add((i - s) * dataset.n_features)
    ps:copy(p)
    arr.push(ss, dataset.solutions[i])
  end
  ss = ivec.create(ss)
  return {
    problems = ps,
    solutions = ss,
    n = n
  }
end

M.add_binary_mnist_pairs = function (split, n_centers, n_negatives)
  local pos, neg = ivec.create(), ivec.create()
  local label = split.solutions
  local N = split.n
  local class_to_indices = {}
  for i = 0, N - 1 do
    local y = label:get(i)
    if not class_to_indices[y] then
      class_to_indices[y] = {}
    end
    table.insert(class_to_indices[y], i)
  end
  local centers_by_class = {}
  for class, indices in pairs(class_to_indices) do
    local count = #indices
    local centers = {}
    for _ = 1, math.min(n_centers, count) do
      local idx = math.random(#indices)
      table.insert(centers, indices[idx])
      table.remove(indices, idx)
    end
    centers_by_class[class] = centers
    for _, i in ipairs(indices) do
      local center = centers[math.random(#centers)]
      pos:push(i)
      pos:push(center)
    end
  end
  local classes = {}
  for class in pairs(centers_by_class) do
    table.insert(classes, class)
  end
  for i = 1, #classes do
    local class_i = classes[i]
    local centers_i = centers_by_class[class_i]
    for _, ci in ipairs(centers_i) do
      for _ = 1, n_negatives do
        local other_class
        repeat
          other_class = classes[math.random(#classes)]
        until other_class ~= class_i
        local cj_list = centers_by_class[other_class]
        if cj_list and #cj_list > 0 then
          local cj = cj_list[math.random(#cj_list)]
          neg:push(ci)
          neg:push(cj)
        end
      end
    end
  end
  split.pos = pos
  split.neg = neg
end

M.split_binary_mnist = function (dataset, ratio)
  if ratio >= 1 then
    return
      _split_binary_mnist(dataset, 1, #dataset.problems)
  else
    local n_train = num.floor(#dataset.problems * ratio)
    return
      _split_binary_mnist(dataset, 1, n_train),
      _split_binary_mnist(dataset, n_train + 1, #dataset.problems)
  end
end

M.read_imdb = function (dir, max)
  local problems = {}
  local solutions = {}
  local pos = it.paste(1, it.take(max or math.huge, fs.files(dir .. "/pos")))
  local neg = it.paste(0, it.take(max or math.huge, fs.files(dir .. "/neg")))
  local samples = it.map(function (label, text)
    return label, fs.readfile(text)
  end, it.chain(pos, neg))
  for label, review in samples do
    solutions[#solutions + 1] = label
    problems[#problems + 1] = review
  end
  arr.shuffle(problems, solutions)
  return {
    n = #problems,
    problems = problems,
    solutions = solutions
  }
end

local function _split_imdb (dataset, s, e)
  local ps = arr.copy({}, dataset.problems, 1, s, e)
  local ss = arr.copy({}, dataset.solutions, 1, s, e)
  ss = ivec.create(ss)
  return {
    n = #ps,
    problems = ps,
    solutions = ss
  }
end

M.split_imdb = function (dataset, ratio)
  local n_train = num.floor(#dataset.problems * ratio)
  return
    _split_imdb(dataset, 1, n_train),
    _split_imdb(dataset, n_train + 1, #dataset.problems)
end

M.read_glove = function (fp, max)
  local n_dims = nil
  local embeddings = dvec.create()
  local words = {}
  local added = 0
  for l in it.take(max or math.huge, fs.lines(fp)) do
    local chunks = str.gmatch(l, "%S+")
    local word = chunks()
    words[#words + 1] = word
    words[word] = #words
    for n in chunks do
      n = err.assert(tonumber(n), "error parsing dimension")
      embeddings:push(n)
      added = added + 1
    end
    if not n_dims then
      n_dims = added
    end
    err.assert(added == n_dims, "dimension mismatch", "expected", n_dims, "found", added)
    added = 0
  end
  embeddings:shrink()
  return {
    n_embeddings = #words,
    embeddings = embeddings,
    n_dims = n_dims,
    words = words
  }
end

return M
