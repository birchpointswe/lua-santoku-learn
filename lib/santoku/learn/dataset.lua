local serialize = require("santoku.serialize") -- luacheck: ignore
local booleanizer = require("santoku.learn.booleanizer")
local quantizer = require("santoku.learn.quantizer")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local dvec = require("santoku.dvec")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")
local json = require("cjson")

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
  local bits = ivec.create()
  for i = 0, e - s do
    bits:push(i * dataset.n_labels + dataset.solutions:get(s - 1 + i))
  end
  return {
    ids = ids,
    solutions = bits,
    n_labels = dataset.n_labels,
    n_features = dataset.n_features,
    n = e - s + 1,
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
  local bits = ivec.create()
  for i = 1, #ss do
    bits:push((i - 1) * 2 + ss[i])
  end
  return {
    n = #ps,
    problems = ps,
    solutions = bits
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
  local cat_idx = 0
  for cat_dir in fs.dirs(dir) do
    local cat_name = fs.basename(cat_dir)
    categories[#categories + 1] = cat_name
    local n = 0
    for fp in fs.files(cat_dir) do
      if max_per_class and n >= max_per_class then break end
      solutions[#solutions + 1] = cat_idx
      local raw = fs.readfile(fp)
      problems[#problems + 1] = clean_newsgroup_text(raw, remove)
      n = n + 1
    end
    cat_idx = cat_idx + 1
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
  local bits = ivec.create()
  for i = 1, total do
    bits:push((i - 1) * cat_idx + shuffled_solutions[i])
  end
  return {
    n = total,
    n_labels = cat_idx,
    categories = categories,
    problems = shuffled_problems,
    solutions = bits
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
    solutions = test_raw.solutions
  }
  if not tvr or tvr <= 0 then
    return all_train, test
  end
  local n_labels = all_train.n_labels
  local all_classes = {}
  for j = 0, all_train.solutions:size() - 1 do
    local v = all_train.solutions:get(j)
    local s = math.floor(v / n_labels)
    local c = v % n_labels
    all_classes[s] = c
  end
  local val_n = math.floor(all_train.n * tvr)
  local train_n = all_train.n - val_n
  local train_problems, val_problems = {}, {}
  local train_solutions, val_solutions = ivec.create(), ivec.create()
  for i = 1, train_n do
    train_problems[i] = all_train.problems[i]
    train_solutions:push((i - 1) * n_labels + all_classes[i - 1])
  end
  for i = train_n + 1, all_train.n do
    val_problems[i - train_n] = all_train.problems[i]
    val_solutions:push((i - train_n - 1) * n_labels + all_classes[i - 1])
  end
  local train = {
    n = train_n,
    n_labels = all_train.n_labels,
    categories = all_train.categories,
    problems = train_problems,
    solutions = train_solutions
  }
  local validate = {
    n = val_n,
    n_labels = all_train.n_labels,
    categories = all_train.categories,
    problems = val_problems,
    solutions = val_solutions
  }
  return train, test, validate
end

local function read_eurlex57k_jsonl (fp, max, label_map, label_counts)
  local problems = {}
  local solutions = ivec.create()
  local n = 0
  for line in fs.lines(fp) do
    if max and n >= max then
      break
    end
    local doc = json.decode(line)
    local parts = {}
    if doc.title then parts[#parts + 1] = doc.title end
    if doc.header then parts[#parts + 1] = doc.header end
    if doc.recitals then parts[#parts + 1] = doc.recitals end
    if doc.main_body then
      if type(doc.main_body) == "table" then
        for _, p in ipairs(doc.main_body) do
          parts[#parts + 1] = p
        end
      else
        parts[#parts + 1] = doc.main_body
      end
    end
    problems[#problems + 1] = table.concat(parts, "\n")
    local labels = doc.eurovoc_concepts or {}
    local doc_label_count = 0
    for _, lbl in ipairs(labels) do
      local idx = label_map[lbl]
      if idx then
        solutions:push(n * label_map.n_labels + idx)
        doc_label_count = doc_label_count + 1
      end
    end
    label_counts:push(doc_label_count)
    n = n + 1
  end
  solutions:asc()
  return problems, solutions, n
end

M.read_eurlex57k = function (dir, max)
  local label_map = { n_labels = 0 }
  for _, fname in ipairs({ "train.jsonl", "dev.jsonl", "test.jsonl" }) do
    local fp = dir .. "/" .. fname
    if fs.exists(fp) then
      for line in fs.lines(fp) do
        local doc = json.decode(line)
        for _, lbl in ipairs(doc.eurovoc_concepts or {}) do
          if not label_map[lbl] then
            label_map[lbl] = label_map.n_labels
            label_map.n_labels = label_map.n_labels + 1
          end
        end
      end
    end
  end
  local train_label_counts = ivec.create()
  local dev_label_counts = ivec.create()
  local test_label_counts = ivec.create()
  local train_problems, train_solutions, train_n =
    read_eurlex57k_jsonl(dir .. "/train.jsonl", max, label_map, train_label_counts)
  local dev_problems, dev_solutions, dev_n =
    read_eurlex57k_jsonl(dir .. "/dev.jsonl", max, label_map, dev_label_counts)
  local test_problems, test_solutions, test_n =
    read_eurlex57k_jsonl(dir .. "/test.jsonl", max, label_map, test_label_counts)
  local train = {
    n = train_n,
    n_labels = label_map.n_labels,
    problems = train_problems,
    solutions = train_solutions,
    label_counts = train_label_counts,
  }
  local dev = {
    n = dev_n,
    n_labels = label_map.n_labels,
    problems = dev_problems,
    solutions = dev_solutions,
    label_counts = dev_label_counts,
  }
  local test = {
    n = test_n,
    n_labels = label_map.n_labels,
    problems = test_problems,
    solutions = test_solutions,
    label_counts = test_label_counts,
  }
  return train, dev, test, label_map
end

M.read_california_housing = function (fp, opts)
  opts = opts or {}
  local n_bins = opts.n_bins or opts.n_thresholds or 0
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
  local cont = dvec.create()
  for _, row in ipairs(data) do
    for _, col in ipairs(feature_cols) do
      cont:push(tonumber(row[col]) or 0)
    end
  end
  local encoder = quantizer.create({
    raw_codes = cont,
    n_samples = #data,
    mode = "thermometer",
    n_bins = n_bins,
  })
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
    encoder = encoder,
    booleanizer = bzr,
    n = #shuffled,
    n_features = encoder:n_bits() + bzr:features(),
    feature_cols = feature_cols,
    categorical_cols = categorical_cols,
    target_col = target_col,
  }
end

local function _encode_housing_split (dataset, rows)
  local encoder = dataset.encoder
  local bzr = dataset.booleanizer
  local feature_cols = dataset.feature_cols
  local categorical_cols = dataset.categorical_cols
  local target_col = dataset.target_col
  local n_thermo = encoder:n_bits()
  local n_features = n_thermo + bzr:features()
  local dims = encoder:dims()
  local threshs = encoder:thresholds()
  local bits = ivec.create()
  local targets = dvec.create()
  for i, row in ipairs(rows) do
    local base = (i - 1) * n_features
    local vals = {}
    for j, col in ipairs(feature_cols) do
      vals[j] = tonumber(row[col]) or 0
    end
    for k = 0, n_thermo - 1 do
      if vals[dims:get(k) + 1] > threshs:get(k) then
        bits:push(base + k)
      end
    end
    for _, col in ipairs(categorical_cols) do
      local val = row[col]
      if val then
        local bit = bzr:feature(col, val)
        if bit then
          bits:push(base + n_thermo + bit)
        end
      end
    end
    targets:push(tonumber(row[target_col]))
  end
  return {
    n = #rows,
    n_features = n_features,
    bits = bits,
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

return M
