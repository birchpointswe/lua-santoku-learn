require("santoku.rvec")
local serialize = require("santoku.serialize") -- luacheck: ignore
local inv = require("santoku.tsetlin.inv")
local pvec = require("santoku.pvec")
local ivec = require("santoku.ivec")
local it = require("santoku.iter")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local err = require("santoku.error")

local M = {}

M.read_binary_mnist = function (fp, n_features, max, class_max)
  local class_cnt = {}
  local offsets = ivec.create()
  local problems = ivec.create()
  local solutions = ivec.create()
  local n = 0
  for l in fs.lines(fp) do
    if max and n >= max then
      break
    end
    local start = problems:size()
    local tokens = {}
    local token_count = 0
    for token in str.gmatch(l, "%S+") do
      token_count = token_count + 1
      tokens[token_count] = token
      if token_count > n_features then
        break
      end
    end
    if token_count <= n_features then -- luacheck: ignore
      -- skip
    else
      local cls = tokens[n_features + 1]
      class_cnt[cls] = (class_cnt[cls] or 0) + 1

      if not (class_max and class_cnt[cls] > class_max) then
        offsets:push(start)
        for feature = 1, n_features do
          if tokens[feature] == "1" then
            problems:push(feature - 1)
          elseif tokens[feature] ~= "0" then
            err.error("unexpected string", tokens[feature])
          end
        end
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
    local offset = (i - s) * dataset.n_features
    for j = pss, pse - 1 do
      ps:push(dataset.problems:get(j) + offset)
    end
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
  edges_per_node = edges_per_node or 3
  local edges = pvec.create()
  local n = ids:size()
  if n <= 1 then
    return edges
  end
  for i = 0, n - 1 do
    local id1 = ids:get(i)
    for _ = 1, edges_per_node do
      local idx2 = num.random(n) - 1
      if idx2 == i then
        idx2 = (idx2 + 1) % n
      end
      local id2 = ids:get(idx2)
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
      if prev == nil then
        label_of_id[id] = y
        if not class_to_ids[y] then
          class_to_ids[y] = ivec.create()
          classes:push(y)
        end
        class_to_ids[y]:push(id)
      elseif prev ~= y then
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

return M
