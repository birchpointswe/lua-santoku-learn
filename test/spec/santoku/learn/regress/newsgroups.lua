local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil, tvr = 0.1 },
  tok = { ngram_min = 5, ngram_max = 5 },
  emb = { n_landmarks = 1024*32, trace_tol = 0.01, kernel = { "cosine", "nngp", "ntk", "expcos", "geolaplace" } },
  ridge = {
    lambda = { def = 1.43e-01 },
    propensity_a = { def = 0.09 },
    propensity_b = { def = 4.97 },
    classes = 20,
    search_trials = 800,
    k = 1,
  },
}

test("newsgroups classifier", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, test_set, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max, nil, cfg.data.tvr)
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors
  str.printf("[Data] train=%d val=%d test=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_classes, sw())

  local ngram_map, offsets, tokens, values, n_tokens = csr.tokenize({
    texts = train.problems, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max, n_samples = train.n,
  })
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    label_off, label_nbr, n_tokens, n_classes)
  str.printf("[Tokenize] ngram_min=%d ngram_max=%d tokens=%d %s\n",
    cfg.tok.ngram_min, cfg.tok.ngram_max, n_tokens, sw())

  local _, val_off, val_tok, val_val = csr.tokenize({
    texts = validate.problems, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    n_samples = validate.n, ngram_map = ngram_map,
  })
  csr.apply_bns(val_off, val_tok, val_val, bns_scores)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, val_codes, best_params = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    val_offsets = val_off, val_tokens = val_tok, val_values = val_val,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  validate.problems = nil
  collectgarbage("collect")
  local emb_d = sp_enc:dims()
  str.printf("[KRR] emb_d=%d kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    emb_d, best_params.kernel, best_params.lambda,
    best_params.propensity_a, best_params.propensity_b, sw())

  local function encode_texts(texts, n)
    local _, off, tok, val = csr.tokenize({
      texts = texts, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_samples = n, ngram_map = ngram_map,
    })
    csr.apply_bns(off, tok, val, bns_scores)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  str.printf("[Eval] Labeling splits\n")
  local _, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode_texts(test_set.problems, test_set.n)
  test_set.problems = nil
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  local val_stats = eval.class_accuracy(val_labels, validate.sol_offsets, validate.sol_neighbors, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.sol_offsets, test_set.sol_neighbors, test_set.n, n_classes)
  str.printf("[Class] F1: val=%.2f test=%.2f %s\n",
    val_stats.f1, test_stats.f1, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
