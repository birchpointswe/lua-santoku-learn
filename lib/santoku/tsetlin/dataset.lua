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

M.read_binary_mnist = function (fp, n_features, max)
  local ps = ivec.create()
  local ss = ivec.create()
  local n = 0
  for l in fs.lines(fp) do
    if max and n >= max then
      break
    end
    local f = 0
    for token in str.gmatch(l, "%S+") do
      if f == n_features then
        ss:push(tonumber(token))
        break
      elseif token == "1" then
        ps:push(n * n_features + f)
      end
      f = f + 1
    end
    n = n + 1
  end
  local ids = ivec.create(n)
  ids:fill_indices()
  return {
    ids = ids,
    problems = ps,
    solutions = ss,
    n_labels = 10,
    n_features = n_features,
    n = n,
  }
end

local function _split_binary_mnist (dataset, s, e)
  local ids = ivec.create()
  ids:copy(dataset.ids, s - 1, e, 0)
  local ss = ivec.create()
  ss:copy(dataset.solutions, s - 1, e, 0)
  return {
    ids = ids,
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

M.random_pairs = function (ids, edges_per_node, out, labels)
  edges_per_node = edges_per_node or 3
  local edges = out or pvec.create()
  local n = ids:size()
  if n <= 1 then
    return edges
  end
  local ids_by_label
  if labels then
    err.assert(ids:size() == labels:size(), "ids and labels must align")
    ids_by_label = {}
    for i = 0, n - 1 do
      local id = ids:get(i)
      local label = labels:get(i)
      if label ~= -1 then
        if not ids_by_label[label] then
          ids_by_label[label] = ivec.create()
        end
        ids_by_label[label]:push(id)
      end
    end
  end
  for i = 0, n - 1 do
    local id1 = ids:get(i)
    local label1 = labels and labels:get(i)
    if not labels or label1 ~= -1 then
      for _ = 1, edges_per_node do
        local id2
        if labels then
          local same_label_ids = ids_by_label[label1]
          if same_label_ids and same_label_ids:size() > 1 then
            local idx2 = num.random(same_label_ids:size()) - 1
            id2 = same_label_ids:get(idx2)
            while id2 == id1 do
              idx2 = num.random(same_label_ids:size()) - 1
              id2 = same_label_ids:get(idx2)
            end
          end
        else
          local idx2 = num.random(n) - 1
          if idx2 == i then
            idx2 = (idx2 + 1) % n
          end
          id2 = ids:get(idx2)
        end
        if id2 then
          if id1 < id2 then
            edges:push(id1, id2)
          else
            edges:push(id2, id1)
          end
        end
      end
    end
  end
  return edges
end

M.anchor_pairs = function (ids, n_anchors, out, labels)
  if n_anchors == nil or n_anchors < 1 then
    n_anchors = 1
  end
  local edges = out or pvec.create()
  if labels then
    err.assert(ids:size() == labels:size(), "ids and labels must align")
    local ids_by_label = {}
    for i = 0, ids:size() - 1 do
      local id = ids:get(i)
      local label = labels:get(i)
      if label ~= -1 then
        if not ids_by_label[label] then
          ids_by_label[label] = {}
        end
        table.insert(ids_by_label[label], id)
      end
    end
    for _, label_ids in pairs(ids_by_label) do
      local n_label_anchors = num.min(n_anchors, #label_ids)
      local anchors = {}
      while #anchors < n_label_anchors do
        local a = label_ids[num.random(#label_ids)]
        if not anchors[a] then
          anchors[a] = true
          for _, id in ipairs(label_ids) do
            if id ~= a then
              edges:push(id, a)
            end
          end
        end
      end
    end
  else
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
  local label_of_id = {}
  local n_unique_ids = 0
  for i, y in labels:ieach() do
    local id = ids:get(i)
    if y ~= -1 then
      local prev = label_of_id[id]
      if prev == nil then
        n_unique_ids = n_unique_ids + 1
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
    local n_edges_created = 0
    local n_singleton_classes = 0
    for class in classes:each() do
      local idvec, anchors = class_to_ids[class], class_to_anchors[class]
      if idvec:size() == 1 then
        n_singleton_classes = n_singleton_classes + 1
      end
      for id in idvec:each() do
        for a_id in anchors:each() do
          if id ~= a_id then
            if index and eps_pos then
              if index:distance(id, a_id) <= eps_pos then
                pos:push(canonicalize(id, a_id))
                n_edges_created = n_edges_created + 1
              end
            else
              pos:push(canonicalize(id, a_id))
              n_edges_created = n_edges_created + 1
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

M.star_hoods = function (ids, hoods)
  local out = pvec.create()
  for idx, hood in hoods:ieach() do
    local id = ids:get(idx)
    for idxnbr in hood:each() do
      local nbr = ids:get(idxnbr)
      out:push(id, nbr)
    end
  end
  return out
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
