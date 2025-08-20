require("santoku.rvec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local tbl = require("santoku.table")
local inv = require("santoku.tsetlin.inv")
local pvec = require("santoku.pvec")
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

M.read_binary_mnist = function (fp, n_features, max, class_max)
  local class_cnt = {}
  local offsets = ivec.create()
  local problems = ivec.create()
  local solutions = ivec.create()
  local n = 0
  local tmp = ivec.create()
  for l in fs.lines(fp) do
    if max and n >= max then
      break
    end
    local start = problems:size()
    local feature = 0
    tmp:clear()
    local cls = nil
    for s in str.gmatch(l, "%S+") do
      if feature == n_features then
        cls = s
        break
      elseif s == "1" then
        tmp:push(feature)
      elseif s ~= "0" then
        err.error("unexpected string", s)
      end
      feature = feature + 1
    end
    if not cls then -- luacheck: ignore
      -- malformed row: no label; skip silently (or log)
    else
      class_cnt[cls] = (class_cnt[cls] or 0) + 1
      if not (class_max and class_cnt[cls] > class_max) then
        offsets:push(start)
        problems:copy(tmp, 0, tmp:size(), problems:size())
        solutions:push(tonumber(cls))
        n = n + 1
      end
    end
  end
  return {
    offsets = offsets,
    problems = problems,
    solutions = solutions,
    n_labels = 10,
    n_features = n_features,
    n = n,
  }
end

local function _split_binary_mnist (dataset, s, e)
  local ps = ivec.create()
  local ss = ivec.create()
  for i = s, e do
    local pss = dataset.offsets:get(i - 1)
    local pse = i == dataset.n and dataset.problems:size() or dataset.offsets:get(i)
    local m = ps:size()
    ps:copy(dataset.problems, pss, pse, m)
    ps:add((i - s) * dataset.n_features, m, ps:size())
  end
  ss:copy(dataset.solutions, s - 1, e, 0)
  return {
    problems = ps,
    solutions = ss,
    n_labels = dataset.n_labels,
    n_features = dataset.n_features,
    n = e - s + 1,
  }
end

M.split_binary_mnist = function (dataset, ratio)
  if ratio >= 1 then
    return
      _split_binary_mnist(dataset, 1, dataset.n)
  else
    local n_train = num.floor(dataset.n * ratio)
    return
      _split_binary_mnist(dataset, 1, n_train),
      _split_binary_mnist(dataset, n_train + 1, dataset.n)
  end
end

local function canonicalize (a, b)
  if a < b then
    return a, b
  else
    return b, a
  end
end

M.random_pairs = function (ids, edges_per_node)
  edges_per_node = edges_per_node or 3  -- Default to 3 random edges per node
  local edges = pvec.create()
  local n = ids:size()
  -- Early return for small graphs
  if n <= 1 then
    return edges
  end
  -- Each node gets edges_per_node random connections
  for i = 0, n - 1 do
    local id1 = ids:get(i)
    for _ = 1, edges_per_node do
      -- Pick a random target
      local idx2 = num.random(n) - 1  -- num.random returns 1-based
      -- Avoid self-loops
      if idx2 == i then
        idx2 = (idx2 + 1) % n
      end
      local id2 = ids:get(idx2)
      -- Add edge (undirected, so just one direction)
      if id1 < id2 then
        edges:push(id1, id2)
      else
        edges:push(id2, id1)
      end
    end
  end
  return edges
end

M.anchor_pairs = function (ids, n_anchors)
  if n_anchors == nil or n_anchors < 1 then
    n_anchors = 1
  end
  local edges = pvec.create()
  local anchors = {}
  local sofar = 0
  while sofar < n_anchors do
    local a = ids:get(num.random(ids:size()) - 1)
    if not anchors[a] then
      anchors[a] = true
      sofar = sofar + 1
      for id in ids:each() do
        if id ~= a then
          edges:push(id, a)
        end
      end
    end
  end
  return edges
end

-- Convert list of ids/classes into an index where "features" are class labels
M.classes_index = function (ids, classes)
  local fids = ivec.create(classes:size())
  local nfid = 0
  local fididx = {}
  for idx, lbl in classes:ieach() do
    local fid = fididx[lbl]
    if not fid then
      fid = nfid
      nfid = nfid + 1
      fididx[lbl] = fid
    end
    fids:set(idx, fid)
  end
  for idx, fid in fids:ieach() do
    fids:set(idx, idx * nfid + fid)
  end
  local index = inv.create({ features = nfid })
  index:add(fids, ids)
  return index
end

M.multiclass_pairs = function (ids, labels, n_anchors_pos, n_anchors_neg, index, eps_pos, eps_neg)
  err.assert(ids:size() == labels:size(), "ids and labels must align")
  if n_anchors_pos == nil then
    n_anchors_pos = 1
  end
  if n_anchors_neg == nil then
    n_anchors_neg = (n_anchors_pos > 0) and n_anchors_pos or 1
  end
  local pos, neg = pvec.create(), pvec.create()
  local class_to_ids, class_to_anchors = {}, {}
  local classes = ivec.create()
  local label_of_id   = {}
  for i, y in labels:ieach() do
    local id = ids:get(i)
    if y ~= -1 then
      local prev = label_of_id[id]
      if prev == nil then -- first time we see this id
        label_of_id[id] = y
        if not class_to_ids[y] then
          class_to_ids[y] = ivec.create()
          classes:push(y)
        end
        class_to_ids[y]:push(id)
      elseif prev ~= y then -- conflicting label
        err.abort(("id %d has labels %d and %d"):format(id, prev, y))
      end
    end
  end
  local shuffle = ivec.create()
  for class in classes:each() do
    class_to_anchors[class] = ivec.create()
  end
  if n_anchors_pos > 0 then
    for class in classes:each() do
      local idvec = class_to_ids[class]
      local k = num.min(n_anchors_pos, idvec:size())
      if k > 0 then
        shuffle:resize(idvec:size())
        shuffle:fill_indices()
        shuffle:shuffle()
        local anchors = class_to_anchors[class]
        for i = 0, k - 1 do
          anchors:push(idvec:get(shuffle:get(i)))
        end
      end
    end
  end
  if n_anchors_pos > 0 then
    for class in classes:each() do
      local idvec, anchors = class_to_ids[class], class_to_anchors[class]
      for id in idvec:each() do
        for a_id in anchors:each() do
          if id ~= a_id then
            if index and eps_pos then
              if index:distance(id, a_id) <= eps_pos then
                pos:push(canonicalize(id, a_id))
              end
            else
              pos:push(canonicalize(id, a_id))
            end
          end
        end
      end
    end
  end
  if n_anchors_neg > 0 then
    local global_pool = ivec.create()
    for _, vec in pairs(class_to_ids) do
      for id in vec:each() do
        global_pool:push(id)
      end
    end
    local gsize = global_pool:size()
    for class in classes:each() do
      local idvec = class_to_ids[class]
      for id in idvec:each() do
        for _ = 1, n_anchors_neg do
          local tries, max_try, a_id = 0, gsize * 3
          repeat
            a_id = global_pool:get(num.random(gsize) - 1)
            tries = tries + 1
          until (label_of_id[a_id] ~= class and
                 (not index or not eps_neg or index:distance(id, a_id) >= eps_neg))
                or tries >= max_try
          if tries < max_try then
            neg:push(canonicalize(id, a_id))
          end
        end
      end
    end
  end
  pos:xasc()
  neg:xasc()
  return pos, neg
end

M.read_imdb = function (dir, max)
  local problems = {}
  local solutions = {}
  local pos = it.paste(1, it.take(max or num.huge, fs.files(dir .. "/pos")))
  local neg = it.paste(0, it.take(max or num.huge, fs.files(dir .. "/neg")))
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
  for l in it.take(max or num.huge, fs.lines(fp)) do
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
