local serialize = require("santoku.serialize") -- luacheck: ignore
local inv = require("santoku.tsetlin.inv")
local ivec = require("santoku.ivec")
local fs = require("santoku.fs")
local str = require("santoku.string")
local arr = require("santoku.array")
local num = require("santoku.num")

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
  ss = ivec.create(ss)
  return {
    n = #ps,
    problems = ps,
    solutions = ss
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
  remove = remove or { headers = true, quotes = true, footers = true, emails = true }
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

M.read_20newsgroups = function (dir, max_per_class, remove)
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
  return {
    n = #problems,
    n_labels = cat_idx,
    categories = categories,
    problems = arr.lookup(idxs, problems, {}),
    solutions = ivec.create(arr.lookup(idxs, solutions, {}))
  }
end

M.read_20newsgroups_split = function (train_dir, test_dir, max_per_class, remove, tvr)
  local all_train = M.read_20newsgroups(train_dir, max_per_class, remove)
  local test_raw = M.read_20newsgroups(test_dir, max_per_class, remove)
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
  local val_n = math.floor(all_train.n * tvr)
  local train_n = all_train.n - val_n
  local train_problems, val_problems = {}, {}
  local train_solutions, val_solutions = ivec.create(), ivec.create()
  for i = 1, train_n do
    train_problems[i] = all_train.problems[i]
    train_solutions:push(all_train.solutions:get(i - 1))
  end
  for i = train_n + 1, all_train.n do
    val_problems[i - train_n] = all_train.problems[i]
    val_solutions:push(all_train.solutions:get(i - 1))
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

return M
