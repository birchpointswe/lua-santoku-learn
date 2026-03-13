local arr = require("santoku.array")
local csr = require("santoku.learn.csr")

local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil, ttr = 0.5, tvr = 0.1 },
  tok = { ngram = 5 },
  emb = { n_landmarks = 8192, trace_tol = 0.01, kernel = "cosine" },
  ridge = {
    lambda = { def = 6.6967e-03 },
    propensity_a = { def = 3.0326 },
    propensity_b = { def = 4.2558 },
    classes = 2,
    search_trials = 0,
    k = 1,
  },
}

local class_names = { "negative", "positive" }

test("imdb csr+kernel", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors
  str.printf("[Data] train=%d val=%d test=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_classes, sw())

  local ngram_map, offsets, tokens, values, n_tokens = csr.tokenize({
    texts = train.problems, hdc_ngram = cfg.tok.ngram, n_samples = train.n,
  })
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    label_off, label_nbr, n_tokens, n_classes)
  str.printf("[Tokenize] ngram=%d tokens=%d %s\n",
    cfg.tok.ngram, n_tokens, sw())

  str.printf("[Spectral] Cholesky trace_tol=%s kernel=%s\n",
    tostring(cfg.emb.trace_tol), cfg.emb.kernel)
  local train_codes, sp_enc, gram = spectral.encode({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
  })
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  collectgarbage("collect")
  local emb_d = sp_enc:dims()
  str.printf("[Spectral] emb_d=%d %s\n", emb_d, sw())

  local function encode_texts(texts, n)
    local _, off, tok, val = csr.tokenize({
      texts = texts, hdc_ngram = cfg.tok.ngram,
      n_samples = n, ngram_map = ngram_map,
    })
    csr.apply_bns(off, tok, val, bns_scores)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  local val_codes = encode_texts(validate.problems, validate.n)
  validate.problems = nil
  collectgarbage("collect")

  str.printf("[Ridge] Training\n")
  local ridge_obj, best_params = optimize.ridge({
    gram = gram,
    val_codes = val_codes, val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  gram = nil
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best_params.lambda, best_params.propensity_a, best_params.propensity_b, sw())

  str.printf("[Eval] Labeling splits\n")
  local _, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  val_codes = nil
  local test_codes = encode_texts(test_set.problems, test_set.n)
  test_set.problems = nil
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
  test_codes = nil
  str.printf("[Eval] Labels done %s\n", sw())

  local _, train_labels = ridge_obj:label(train_codes, train.n, 1)
  local train_stats = eval.class_accuracy(train_labels, train.sol_offsets, train.sol_neighbors, train.n, n_classes)
  local val_stats = eval.class_accuracy(val_labels, validate.sol_offsets, validate.sol_neighbors, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.sol_offsets, test_set.sol_neighbors, test_set.n, n_classes)
  str.printf("[Class] F1: train=%.2f val=%.2f test=%.2f %s\n",
    train_stats.f1, val_stats.f1, test_stats.f1, sw())

  str.printf("\n[Per-class Test Accuracy]\n")
  local class_order = arr.range(1, n_classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    local cat = class_names[c] or ("class_" .. (c - 1))
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
  end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
