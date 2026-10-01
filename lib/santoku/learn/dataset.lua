-- SPDX-License-Identifier: MIT
-- SPDX-FileCopyrightText: 2024 Birch Point SWE
local booleanizer = require("santoku.learn.booleanizer")
local csr = require("santoku.csr")
local mtx = require("santoku.mtx")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local fvec = require("santoku.fvec")
local spans = require("santoku.spans")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local lpeg_utils = require("santoku.lpeg")
local rand = require("santoku.random")

local M = {}

local SHUFFLE_SEED = 1

local function sorted_files (dir)
  local out = {}
  for fp in fs.files(dir) do out[#out + 1] = fp end
  arr.sort(out)
  return out
end

local function shuffled_range (n)
  rand.seed(SHUFFLE_SEED)
  return arr.shuffle(arr.range(1, n))
end

local function single_label_csr (cls, n_cols)
  return csr.from_classes(cls, n_cols)
end

local function split_ranges (n, ratio)
  local n_train = num.floor(n * ratio)
  return { 1, n_train, n_train + 1, n }
end

M.read_binary_mnist = function (fp, n_features, max)
  local p_off = ivec.create()
  local p_nbr = ivec.create()
  local ss = ivec.create()
  local n = 0
  p_off:push(0)
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
        p_nbr:push(f)
      end
      f = f + 1
    end
    n = n + 1
    p_off:push(p_nbr:size())
  end
  local ids = ivec.create(n)
  ids:fill_indices()
  return {
    ids = ids,
    problems = csr.create({ offsets = p_off, neighbors = p_nbr, n_cols = n_features }),
    solutions = ss,
    n_labels = 10,
    n_features = n_features,
    n = n,
  }
end

local function _split_binary_mnist (dataset, s, e)
  local ids = ivec.create()
  ids:copy(dataset.ids, s - 1, e, 0)
  local n = e - s + 1
  local cls = ivec.create(n)
  cls:copy(dataset.solutions, s - 1, e, 0)
  return {
    ids = ids,
    labels = single_label_csr(cls, dataset.n_labels),
    n_labels = dataset.n_labels,
    n_features = dataset.n_features,
    n = n,
  }
end

M.split_binary_mnist = function (dataset, ratio)
  if ratio >= 1 then
    return _split_binary_mnist(dataset, 1, dataset.n)
  end
  local r = split_ranges(dataset.n, ratio)
  return
    _split_binary_mnist(dataset, r[1], r[2]),
    _split_binary_mnist(dataset, r[3], r[4])
end

M.read_imdb = function (dir, max)
  local problems = {}
  local solutions = {}
  local n = 0
  for _, fp in ipairs(sorted_files(dir .. "/pos")) do
    if max and n >= max then break end
    solutions[#solutions + 1] = 1
    problems[#problems + 1] = fs.readfile(fp)
    n = n + 1
  end
  n = 0
  for _, fp in ipairs(sorted_files(dir .. "/neg")) do
    if max and n >= max then break end
    solutions[#solutions + 1] = 0
    problems[#problems + 1] = fs.readfile(fp)
    n = n + 1
  end
  local idxs = shuffled_range(#problems)
  return {
    n = #problems,
    problems = arr.lookup(idxs, problems, {}),
    solutions = arr.lookup(idxs, solutions, {})
  }
end

local function _split_imdb (dataset, s, e)
  local ps = arr.copy({}, dataset.problems, s, e)
  local ss = arr.copy({}, dataset.solutions, s, e)
  local n = #ps
  local sol_off = ivec.create()
  local sol_nbr = ivec.create()
  for i = 1, n do
    sol_off:push(sol_nbr:size())
    if ss[i] == 1 then sol_nbr:push(0) end
  end
  sol_off:push(sol_nbr:size())
  return {
    n = n,
    problems = ps,
    labels = csr.create({ offsets = sol_off, neighbors = sol_nbr, n_cols = 1 }),
  }
end

M.split_imdb = function (dataset, ratio)
  local r = split_ranges(#dataset.problems, ratio)
  return
    _split_imdb(dataset, r[1], r[2]),
    _split_imdb(dataset, r[3], r[4])
end

local scrub_tlds = { "edu", "com", "org", "net", "gov" }

local function clean_newsgroup_text (text, remove)
  remove = remove or { headers = true, quotes = true, footers = true, emails = false }
  local lines = {}
  local in_body = not remove.headers
  local sig_start = nil
  for line in str.gmatch(text, "[^\r\n]*") do
    if not in_body then
      if line == "" then
        in_body = true
      end
    else
      local dominated_by_quotes = str.match(line, "^[>|%%:]") or str.match(line, "^[%s]*[>|%%:]")
      if remove.quotes and dominated_by_quotes then -- luacheck: ignore

      elseif remove.footers and line == "--" then
        sig_start = #lines + 1
        lines[#lines + 1] = line
      else
        if remove.emails then
          line = str.gsub(line, "[%w%.%-_]+@[%w%.%-]+%.[%w]+", "")
          for _, tld in ipairs(scrub_tlds) do
            line = str.gsub(line, "[%w%-]+%.[%w%-]+%." .. tld, "")
          end
        end
        lines[#lines + 1] = line
      end
    end
  end
  if sig_start then
    for i = #lines, sig_start, -1 do
      lines[i] = nil
    end
  end
  return arr.concat(lines, "\n")
end

M.read_20newsgroups = function (dir, max_per_class, remove, max)
  local problems = {}
  local solutions = {}
  local categories = {}
  for cat_dir in fs.dirs(dir) do
    categories[#categories + 1] = { name = fs.basename(cat_dir), path = cat_dir }
  end
  arr.sort(categories, function (a, b) return a.name < b.name end)
  for cat_idx, cat in ipairs(categories) do
    categories[cat_idx] = cat.name
    local n = 0
    for _, fp in ipairs(sorted_files(cat.path)) do
      if max_per_class and n >= max_per_class then break end
      solutions[#solutions + 1] = cat_idx - 1
      local raw = fs.readfile(fp)
      problems[#problems + 1] = clean_newsgroup_text(raw, remove)
      n = n + 1
    end
  end
  local idxs = shuffled_range(#problems)
  local shuffled_problems = arr.lookup(idxs, problems, {})
  local shuffled_solutions = arr.lookup(idxs, solutions, {})
  local total = max and num.min(#shuffled_problems, max) or #shuffled_problems
  if total < #shuffled_problems then
    local ps, ss = {}, {}
    for i = 1, total do
      ps[i] = shuffled_problems[i]
      ss[i] = shuffled_solutions[i]
    end
    shuffled_problems = ps
    shuffled_solutions = ss
  end
  local n_cats = #categories
  local cls = ivec.create(total)
  for i = 0, total - 1 do cls:set(i, shuffled_solutions[i + 1]) end
  return {
    n = total,
    n_labels = n_cats,
    categories = categories,
    problems = shuffled_problems,
    labels = single_label_csr(cls, n_cats),
  }
end

M.read_20newsgroups_split = function (train_dir, test_dir, max, remove)
  local all_train = M.read_20newsgroups(train_dir, nil, remove, max)
  local test_raw = M.read_20newsgroups(test_dir, nil, remove, max)
  local test = {
    n = test_raw.n,
    n_labels = test_raw.n_labels,
    categories = test_raw.categories,
    problems = test_raw.problems,
    labels = test_raw.labels,
  }
  return all_train, test
end

M.read_eurlex57k = function (dir, max)
  local label_map = { n_labels = 0 }
  local text_fields = { "title", "header", "recitals", "main_body" }
  local label_fields = { "eurovoc_concepts" }
  local function make_text_iter(fp, n_max)
    local count = 0
    local lines = fs.lines(fp)
    return function ()
      if n_max and count >= n_max then return nil end
      local line = lines()
      if not line then return nil end
      count = count + 1
      local parts = {}
      for s, e in lpeg_utils.json_fields(line, text_fields) do
        parts[#parts + 1] = line:sub(s, e)
      end
      return arr.concat(parts, "\n")
    end
  end
  local function read_file(fname)
    local fp = dir .. "/" .. fname
    local sol_off = ivec.create()
    local sol_nbr = ivec.create()
    local label_counts = ivec.create()
    local n = 0
    sol_off:push(0)
    for line in fs.lines(fp) do
      if max and n >= max then break end
      local doc_labels = {}
      for s, e in lpeg_utils.json_fields(line, label_fields) do
        local lbl = line:sub(s, e)
        local idx = label_map[lbl]
        if not idx then
          idx = label_map.n_labels
          label_map[lbl] = idx
          label_map.n_labels = label_map.n_labels + 1
        end
        doc_labels[#doc_labels + 1] = idx
      end
      arr.sort(doc_labels)
      for _, idx in ipairs(doc_labels) do sol_nbr:push(idx) end
      label_counts:push(#doc_labels)
      n = n + 1
      sol_off:push(sol_nbr:size())
    end
    return {
      n = n,
      text_iter = function () return make_text_iter(fp, max) end,
      sol_offsets = sol_off, sol_neighbors = sol_nbr,
      label_counts = label_counts,
    }
  end
  local train = read_file("train.jsonl")
  local dev = read_file("dev.jsonl")
  local test = read_file("test.jsonl")
  for _, d in ipairs({ train, dev, test }) do
    d.n_labels = label_map.n_labels
    d.labels = csr.create({ offsets = d.sol_offsets, neighbors = d.sol_neighbors, n_cols = d.n_labels })
    d.sol_offsets = nil
    d.sol_neighbors = nil
  end
  return train, dev, test, label_map
end

M.read_california_housing = function (fp, opts)
  opts = opts or {}
  local max = opts.max
  local feature_cols = opts.feature_cols or {
    "longitude", "latitude", "housing_median_age",
    "total_rooms", "total_bedrooms", "population",
    "households", "median_income"
  }
  local categorical_cols = opts.categorical_cols or { "ocean_proximity" }
  local target_col = opts.target_col or "median_house_value"
  local lines = {}
  for line in fs.lines(fp) do
    lines[#lines + 1] = line
  end
  local header = {}
  for col in str.gmatch(lines[1], "[^,]+") do
    header[#header + 1] = col
  end
  local data = {}
  local n = max and num.min(#lines - 1, max) or (#lines - 1)
  for i = 2, n + 1 do
    local row = {}
    local j = 1
    local line = lines[i] .. ","
    for val in str.gmatch(line, "([^,]*),") do
      if val ~= "" then
        row[header[j]] = val
      end
      j = j + 1
    end
    if row[target_col] then
      data[#data + 1] = row
    end
  end
  local bzr = booleanizer.create()
  for _, row in ipairs(data) do
    for _, col in ipairs(categorical_cols) do
      local val = row[col]
      if val then bzr:observe(col, val) end
    end
  end
  bzr:finalize()
  local idxs = shuffled_range(#data)
  local shuffled = {}
  for i, idx in ipairs(idxs) do
    shuffled[i] = data[idx]
  end
  return {
    data = shuffled,
    booleanizer = bzr,
    n = #shuffled,
    n_features = bzr:features(),
    feature_cols = feature_cols,
    categorical_cols = categorical_cols,
    target_col = target_col,
  }
end

local function _encode_housing_split (dataset, rows)
  local bzr = dataset.booleanizer
  local feature_cols = dataset.feature_cols
  local target_col = dataset.target_col
  local n_features = bzr:features()
  local n_cont = #feature_cols
  local targets = dvec.create()
  local continuous = dvec.create()
  for _, row in ipairs(rows) do
    for _, col in ipairs(feature_cols) do
      continuous:push(tonumber(row[col]) or 0)
    end
    targets:push(tonumber(row[target_col]))
  end
  return {
    n = #rows,
    n_features = n_features,
    n_continuous = n_cont,
    bits = bzr:encode({ samples = rows, cols = dataset.categorical_cols }),

    continuous = mtx.create({ data = continuous:to_fvec(), n_rows = #rows, n_cols = n_cont }),
    targets = targets,
  }
end

M.split_california_housing = function (dataset, ttr)
  local n = dataset.n
  local data = dataset.data
  local n_train = num.floor(n * ttr)
  local train_rows, test_rows = {}, {}
  for i = 1, n_train do
    train_rows[#train_rows + 1] = data[i]
  end
  for i = n_train + 1, n do
    test_rows[#test_rows + 1] = data[i]
  end
  return _encode_housing_split(dataset, train_rows), _encode_housing_split(dataset, test_rows)
end

local ADULT_COLS = {
  "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
  "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
  "hours_per_week", "native_country",
}
local ADULT_CONTINUOUS = {
  age = true, fnlwgt = true, education_num = true,
  capital_gain = true, capital_loss = true, hours_per_week = true,
}

local function read_adult_file (fp)
  local rows, cls = {}, ivec.create()
  for line in fs.lines(fp) do
    local f = str.splits(line, ",")
    if #f == #ADULT_COLS + 1 then
      local row = {}
      for j, col in ipairs(ADULT_COLS) do
        local v = str.trim(f[j])
        row[col] = ADULT_CONTINUOUS[col] and tonumber(v) or v
      end
      rows[#rows + 1] = row
      cls:push(str.find(f[#f], ">50K", 1, true) and 1 or 0)
    end
  end
  return rows, cls
end

M.read_adult = function (dir)
  local train_rows, train_cls = read_adult_file(dir .. "/adult.data")
  local test_rows, test_cls = read_adult_file(dir .. "/adult.test")
  local bzr = booleanizer.create()
  for _, row in ipairs(train_rows) do
    for _, col in ipairs(ADULT_COLS) do bzr:observe(col, row[col]) end
  end
  bzr:finalize()
  local function encode (rows, cls)
    local bits, dense = bzr:encode({ samples = rows, cols = ADULT_COLS })
    return { n = #rows, bits = bits, dense = dense, labels = single_label_csr(cls, 2) }
  end
  local n_bits, n_dense = bzr:features()
  return encode(train_rows, train_cls), encode(test_rows, test_cls), {
    booleanizer = bzr, n_bits = n_bits, n_dense = n_dense,
  }
end

local conll_types = { PER = 0, ORG = 1, LOC = 2, MISC = 3 }

local function read_conll_file (fp, max)
  local texts = {}
  local gold = spans.create({ "s", "e", "ty" })
  local n = 0
  local words, ners = {}, {}
  local function flush ()
    if #words == 0 then return end
    if max and n >= max then return "stop" end
    local parts, ends = {}, {}
    local off = 0
    for i = 1, #words do
      if i > 1 then parts[#parts + 1] = " "; off = off + 1 end
      parts[#parts + 1] = words[i]
      off = off + #words[i]
      ends[i] = off
    end
    local ct, cs, ce
    for i = 1, #words do
      local tag = ners[i]
      local bio = tag:sub(1, 1)
      local typ = conll_types[tag:sub(3)]
      local ws, we = ends[i] - #words[i], ends[i]
      if bio == "B" then
        if ct then gold:push(cs, ce, ct) end
        ct, cs, ce = typ, ws, we
      elseif bio == "I" and ct and typ == ct then
        ce = we
      else
        if ct then gold:push(cs, ce, ct); ct = nil end
        if bio == "I" and typ then ct, cs, ce = typ, ws, we end
      end
    end
    if ct then gold:push(cs, ce, ct) end
    gold:doc()
    n = n + 1
    texts[n] = arr.concat(parts)
    words, ners = {}, {}
  end
  for line in fs.lines(fp) do
    if line == "" then
      if flush() == "stop" then break end
    elseif line:sub(1, 9) == "-DOCSTART" then
      words, ners = {}, {}
    else
      local w, _, _, nr = line:match("^(%S+)%s+(%S+)%s+(%S+)%s+(%S+)")
      if w then
        words[#words + 1] = w
        ners[#ners + 1] = nr
      end
    end
  end
  if not (max and n >= max) then flush() end
  return { n = n, texts = texts, gold = gold }
end

M.read_conll2003 = function (dir, max)
  return read_conll_file(dir .. "/train.txt", max),
    read_conll_file(dir .. "/valid.txt", max),
    read_conll_file(dir .. "/test.txt", max)
end

M.merge_conll2003 = function (a, b)
  local m = { n = a.n + b.n, texts = {}, gold = spans.create({ "s", "e", "ty" }) }
  for i = 1, a.n do m.texts[i] = a.texts[i] end
  for i = 1, b.n do m.texts[a.n + i] = b.texts[i] end
  m.gold:append(a.gold)
  m.gold:append(b.gold)
  return m
end

local function json_first (line, field)
  local s, e = lpeg_utils.json_fields(line, { field })()
  return s and str.sub(line, s, e) or ""
end

M.read_beir = function (dir, split)
  local ids, texts, index = {}, {}, {}
  for line in fs.lines(dir .. "/corpus.jsonl") do
    local id = json_first(line, "_id")
    local title, text = json_first(line, "title"), json_first(line, "text")
    local n = #ids
    ids[n + 1] = id
    texts[n + 1] = title == "" and text or (title .. "\n" .. text)
    index[id] = n
  end
  local qtext = {}
  for line in fs.lines(dir .. "/queries.jsonl") do
    qtext[json_first(line, "_id")] = json_first(line, "text")
  end
  local qids, qtexts, qrow = {}, {}, {}
  local rows = {}
  local first = true
  for line in fs.lines(dir .. "/qrels/" .. split .. ".tsv") do
    if first then
      first = false
    elseif line ~= "" then
      local f = str.splits(line, "\t")
      local q, d, g = f[1], f[2], tonumber(f[3])
      local di = index[d]
      if di and qtext[q] then
        local r = qrow[q]
        if not r then
          r = #qids
          qids[r + 1] = q
          qtexts[r + 1] = qtext[q]
          qrow[q] = r
          rows[r + 1] = {}
        end
        local row = rows[r + 1]
        row[#row + 1] = { di, g }
      end
    end
  end
  local off, nbr, val = ivec.create(), ivec.create(), fvec.create()
  off:push(0)
  for r = 1, #rows do
    for _, p in ipairs(rows[r]) do nbr:push(p[1]); val:push(p[2]) end
    off:push(nbr:size())
  end
  return {
    corpus_ids = ids, corpus_texts = texts, n_corpus = #ids,
    query_ids = qids, query_texts = qtexts, n_queries = #qids,
    qrels = csr.create({ offsets = off, neighbors = nbr, values = val, n_cols = #ids }),
  }
end

return M
