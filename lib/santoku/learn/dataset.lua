local serialize = require("santoku.serialize") -- luacheck: ignore
local booleanizer = require("santoku.learn.booleanizer")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local json = require("cjson") -- luacheck: ignore
local lpeg_utils = require("santoku.learn.lpeg")

local M = {}

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
    problem_offsets = p_off,
    problem_neighbors = p_nbr,
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
  local sol_off = ivec.create()
  local sol_nbr = ivec.create()
  for i = 0, n - 1 do
    sol_off:push(i)
    sol_nbr:push(dataset.solutions:get(s - 1 + i))
  end
  sol_off:push(n)
  return {
    ids = ids,
    sol_offsets = sol_off,
    sol_neighbors = sol_nbr,
    n_labels = dataset.n_labels,
    n_features = dataset.n_features,
    n = n,
  }
end


M.split_binary_mnist = function (dataset, ratio, tvr)
  if ratio >= 1 then
    return _split_binary_mnist(dataset, 1, dataset.n)
  else
    local n_train_total = num.floor(dataset.n * ratio)
    if not tvr or tvr <= 0 then
      return
        _split_binary_mnist(dataset, 1, n_train_total),
        _split_binary_mnist(dataset, n_train_total + 1, dataset.n)
    end
    local n_val = num.floor(n_train_total * tvr)
    local n_train = n_train_total - n_val
    return
      _split_binary_mnist(dataset, 1, n_train),
      _split_binary_mnist(dataset, n_train_total + 1, dataset.n),
      _split_binary_mnist(dataset, n_train + 1, n_train_total)
  end
end

M.read_imdb = function (dir, max)
  local problems = {}
  local solutions = {}
  local n = 0
  for fp in fs.files(dir .. "/pos") do
    if max and n >= max then break end
    solutions[#solutions + 1] = 1
    problems[#problems + 1] = fs.readfile(fp)
    n = n + 1
  end
  n = 0
  for fp in fs.files(dir .. "/neg") do
    if max and n >= max then break end
    solutions[#solutions + 1] = 0
    problems[#problems + 1] = fs.readfile(fp)
    n = n + 1
  end
  local idxs = arr.shuffle(arr.range(1, #problems))
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
    sol_off:push(i - 1)
    sol_nbr:push(ss[i])
  end
  sol_off:push(n)
  return {
    n = n,
    problems = ps,
    sol_offsets = sol_off,
    sol_neighbors = sol_nbr,
  }
end

M.split_imdb = function (dataset, ratio, tvr)
  local n_train_total = num.floor(#dataset.problems * ratio)
  if not tvr or tvr <= 0 then
    return
      _split_imdb(dataset, 1, n_train_total),
      _split_imdb(dataset, n_train_total + 1, #dataset.problems)
  end
  local n_val = num.floor(n_train_total * tvr)
  local n_train = n_train_total - n_val
  return
    _split_imdb(dataset, 1, n_train),
    _split_imdb(dataset, n_train_total + 1, #dataset.problems),
    _split_imdb(dataset, n_train + 1, n_train_total)
end

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
        -- nothing
      elseif remove.footers and line == "--" then
        sig_start = #lines + 1
        lines[#lines + 1] = line
      else
        if remove.emails then
          line = str.gsub(line, "[%w%.%-_]+@[%w%.%-]+%.[%w]+", "")
          line = str.gsub(line, "[%w%-]+%.[%w%-]+%.edu", "")
          line = str.gsub(line, "[%w%-]+%.[%w%-]+%.com", "")
          line = str.gsub(line, "[%w%-]+%.[%w%-]+%.org", "")
          line = str.gsub(line, "[%w%-]+%.[%w%-]+%.net", "")
          line = str.gsub(line, "[%w%-]+%.[%w%-]+%.gov", "")
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
  return table.concat(lines, "\n")
end

M.read_20newsgroups = function (dir, max_per_class, remove, max)
  local problems = {}
  local solutions = {}
  local categories = {}
  for cat_dir in fs.dirs(dir) do
    categories[#categories + 1] = { name = fs.basename(cat_dir), path = cat_dir }
  end
  table.sort(categories, function (a, b) return a.name < b.name end)
  for cat_idx, cat in ipairs(categories) do
    categories[cat_idx] = cat.name
    local n = 0
    for fp in fs.files(cat.path) do
      if max_per_class and n >= max_per_class then break end
      solutions[#solutions + 1] = cat_idx - 1
      local raw = fs.readfile(fp)
      problems[#problems + 1] = clean_newsgroup_text(raw, remove)
      n = n + 1
    end
  end
  local idxs = arr.shuffle(arr.range(1, #problems))
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
  local sol_off = ivec.create()
  local sol_nbr = ivec.create()
  for i = 1, total do
    sol_off:push(i - 1)
    sol_nbr:push(shuffled_solutions[i])
  end
  sol_off:push(total)
  return {
    n = total,
    n_labels = n_cats,
    categories = categories,
    problems = shuffled_problems,
    sol_offsets = sol_off,
    sol_neighbors = sol_nbr,
  }
end

M.read_20newsgroups_split = function (train_dir, test_dir, max, remove, tvr)
  local all_train = M.read_20newsgroups(train_dir, nil, remove, max)
  local test_raw = M.read_20newsgroups(test_dir, nil, remove, max)
  local test = {
    n = test_raw.n,
    n_labels = test_raw.n_labels,
    categories = test_raw.categories,
    problems = test_raw.problems,
    sol_offsets = test_raw.sol_offsets,
    sol_neighbors = test_raw.sol_neighbors,
  }
  if not tvr or tvr <= 0 then
    return all_train, test
  end
  local val_n = math.floor(all_train.n * tvr)
  local train_n = all_train.n - val_n
  local train_problems, val_problems = {}, {}
  local train_sol_off, train_sol_nbr = ivec.create(), ivec.create()
  local val_sol_off, val_sol_nbr = ivec.create(), ivec.create()
  for i = 1, train_n do
    train_problems[i] = all_train.problems[i]
    train_sol_off:push(i - 1)
    train_sol_nbr:push(all_train.sol_neighbors:get(i - 1))
  end
  train_sol_off:push(train_n)
  for i = train_n + 1, all_train.n do
    val_problems[i - train_n] = all_train.problems[i]
    val_sol_off:push(i - train_n - 1)
    val_sol_nbr:push(all_train.sol_neighbors:get(i - 1))
  end
  val_sol_off:push(val_n)
  local train = {
    n = train_n,
    n_labels = all_train.n_labels,
    categories = all_train.categories,
    problems = train_problems,
    sol_offsets = train_sol_off,
    sol_neighbors = train_sol_nbr,
  }
  local validate = {
    n = val_n,
    n_labels = all_train.n_labels,
    categories = all_train.categories,
    problems = val_problems,
    sol_offsets = val_sol_off,
    sol_neighbors = val_sol_nbr,
  }
  return train, test, validate
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
      return table.concat(parts, "\n")
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
      table.sort(doc_labels)
      for _, idx in ipairs(doc_labels) do sol_nbr:push(idx) end
      label_counts:push(#doc_labels)
      n = n + 1
      sol_off:push(sol_nbr:size())
    end
    collectgarbage("collect")
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
  train.n_labels = label_map.n_labels
  dev.n_labels = label_map.n_labels
  test.n_labels = label_map.n_labels
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
  local idxs = arr.shuffle(arr.range(1, #data))
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
  local categorical_cols = dataset.categorical_cols
  local target_col = dataset.target_col
  local n_features = bzr:features()
  local n_cont = #feature_cols
  local bit_off = ivec.create()
  local bit_nbr = ivec.create()
  local targets = dvec.create()
  local continuous = dvec.create()
  bit_off:push(0)
  for _, row in ipairs(rows) do
    for _, col in ipairs(feature_cols) do
      continuous:push(tonumber(row[col]) or 0)
    end
    for _, col in ipairs(categorical_cols) do
      local val = row[col]
      if val then
        local bit = bzr:feature(col, val)
        if bit then
          bit_nbr:push(bit)
        end
      end
    end
    bit_off:push(bit_nbr:size())
    targets:push(tonumber(row[target_col]))
  end
  return {
    n = #rows,
    n_features = n_features,
    n_continuous = n_cont,
    bit_offsets = bit_off,
    bit_neighbors = bit_nbr,
    continuous = continuous,
    targets = targets,
  }
end

M.split_california_housing = function (dataset, ttr, tvr)
  local n = dataset.n
  local data = dataset.data
  local n_train_total = num.floor(n * ttr)
  local n_val = tvr and num.floor(n * tvr) or 0
  local n_train = n_train_total
  local train_rows, val_rows, test_rows = {}, {}, {}
  for i = 1, n_train do
    train_rows[#train_rows + 1] = data[i]
  end
  for i = n_train + 1, n_train + n_val do
    val_rows[#val_rows + 1] = data[i]
  end
  for i = n_train + n_val + 1, n do
    test_rows[#test_rows + 1] = data[i]
  end
  local train = _encode_housing_split(dataset, train_rows)
  local test = _encode_housing_split(dataset, test_rows)
  if n_val > 0 then
    local validate = _encode_housing_split(dataset, val_rows)
    return train, test, validate
  end
  return train, test
end

local snli_label_map = { entailment = 0, neutral = 1, contradiction = 2 }

local function _read_snli_file (fp, max)
  local raw_s1, raw_s2 = {}, {}
  local sol_off = ivec.create()
  local sol_nbr = ivec.create()
  local n = 0
  local fields = { "gold_label", "sentence1", "sentence2" }
  sol_off:push(0)
  for line in fs.lines(fp) do
    if max and n >= max then break end
    local vals = {}
    for s, e in lpeg_utils.json_fields(line, fields) do
      vals[#vals + 1] = line:sub(s, e)
    end
    if #vals == 3 then
      local label = vals[1]
      local cls = snli_label_map[label]
      if cls then
        n = n + 1
        raw_s1[n] = vals[2]
        raw_s2[n] = vals[3]
        sol_off:push(n)
        sol_nbr:push(cls)
      end
    end
  end
  local unique_texts = {}
  local text_idx = {}
  local n_unique = 0
  local idx1_v = ivec.create()
  local idx2_v = ivec.create()
  for i = 1, n do
    if not text_idx[raw_s1[i]] then
      n_unique = n_unique + 1
      text_idx[raw_s1[i]] = n_unique
      unique_texts[n_unique] = raw_s1[i]
    end
    if not text_idx[raw_s2[i]] then
      n_unique = n_unique + 1
      text_idx[raw_s2[i]] = n_unique
      unique_texts[n_unique] = raw_s2[i]
    end
    idx1_v:push(text_idx[raw_s1[i]] - 1)
    idx2_v:push(text_idx[raw_s2[i]] - 1)
  end
  return {
    n = n, unique_texts = unique_texts, n_unique = n_unique,
    idx1 = idx1_v, idx2 = idx2_v,
    sol_offsets = sol_off, sol_neighbors = sol_nbr,
    n_labels = 3,
  }
end

M.read_snli = function (dir, max, tvr)
  local train = _read_snli_file(dir .. "/snli_1.0_train.jsonl", max)
  local dev = _read_snli_file(dir .. "/snli_1.0_dev.jsonl", max)
  local test = _read_snli_file(dir .. "/snli_1.0_test.jsonl", max)
  if tvr and tvr > 0 then
    local val_n = num.floor(train.n * tvr)
    local train_n = train.n - val_n
    local t_sol_off, t_sol_nbr = ivec.create(), ivec.create()
    local v_sol_off, v_sol_nbr = ivec.create(), ivec.create()
    local t_idx1, t_idx2 = ivec.create(), ivec.create()
    local v_idx1, v_idx2 = ivec.create(), ivec.create()
    t_sol_off:push(0)
    v_sol_off:push(0)
    for i = 1, train_n do
      t_idx1:push(train.idx1:get(i - 1))
      t_idx2:push(train.idx2:get(i - 1))
      t_sol_off:push(i)
      t_sol_nbr:push(train.sol_neighbors:get(i - 1))
    end
    for i = train_n + 1, train.n do
      local j = i - train_n
      v_idx1:push(train.idx1:get(i - 1))
      v_idx2:push(train.idx2:get(i - 1))
      v_sol_off:push(j)
      v_sol_nbr:push(train.sol_neighbors:get(i - 1))
    end
    local validate = {
      n = val_n, unique_texts = train.unique_texts, n_unique = train.n_unique,
      idx1 = v_idx1, idx2 = v_idx2,
      sol_offsets = v_sol_off, sol_neighbors = v_sol_nbr,
      n_labels = 3,
    }
    train = {
      n = train_n, unique_texts = train.unique_texts, n_unique = train.n_unique,
      idx1 = t_idx1, idx2 = t_idx2,
      sol_offsets = t_sol_off, sol_neighbors = t_sol_nbr,
      n_labels = 3,
    }
    return train, dev, test, validate
  end
  return train, dev, test
end

return M
