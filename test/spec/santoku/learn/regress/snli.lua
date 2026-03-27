local env = require("santoku.env")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local snli_dir = env.var("SNLI_DIR", nil)
if not snli_dir then
  print("SNLI_DIR not set. Skipping.")
  return
end

local cfg = {
  data = { max = nil, tvr = 0.1 },
  tok = { ngram_min = 7, ngram_max = 7 },
  -- n_landmarks = 1024*24 for max (high memory)
  emb = { n_landmarks = 1024*8, trace_tol = 0.01, kernel = "cosine" --[[{ "cosine", "nngp", "ntk", "expcos", "geolaplace" }]] },
  ridge = {
    lambda = { def = 1.9316e-02 },
    propensity_a = { def = 3.5412 },
    propensity_b = { def = 6.6018 },
    search_trials = 0,
    k = 1,
  },
}

test("snli classifier", function ()

  local stopwatch = utc.stopwatch()
  local n_classes = 3
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set, validate = ds.read_snli(snli_dir, cfg.data.max, cfg.data.tvr)
  str.printf("[Data] train=%d val=%d dev=%d test=%d %s\n",
    train.n, validate.n, dev.n, test_set.n, sw())

  local ngram_map, u_off, u_tok, u_val, n_tokens = csr.tokenize({
    texts = train.unique_texts, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    n_samples = train.n_unique,
  })
  local a_off, a_tok, a_val = csr.gather_rows(u_off, u_tok, u_val, train.idx1)
  local b_off, b_tok, b_val = csr.gather_rows(u_off, u_tok, u_val, train.idx2)
  local offsets, tokens, values = csr.merge(a_off, a_tok, a_val, b_off, b_tok, b_val, n_tokens)
  local n_pair_tokens = n_tokens * 2
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    train.sol_offsets, train.sol_neighbors, n_pair_tokens, n_classes)
  str.printf("[Tokenize] ngram_min=%d ngram_max=%d tokens=%d %s\n",
    cfg.tok.ngram_min, cfg.tok.ngram_max, n_tokens, sw())

  local function tokenize_pairs (split)
    local _, off, tok, val = csr.tokenize({
      texts = split.unique_texts, ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
      n_samples = split.n_unique, ngram_map = ngram_map,
    })
    csr.apply_bns(off, tok, val, bns_scores)
    local ao, at, av = csr.gather_rows(off, tok, val, split.idx1)
    local bo, bt, bv = csr.gather_rows(off, tok, val, split.idx2)
    return csr.merge(ao, at, av, bo, bt, bv, n_tokens)
  end

  local val_off, val_tok, val_val = tokenize_pairs(validate)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, val_codes, best_params = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_pair_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = train.sol_offsets, label_neighbors = train.sol_neighbors, n_labels = n_classes,
    val_offsets = val_off, val_tokens = val_tok, val_values = val_val,
    val_n_samples = validate.n,
    val_expected_offsets = validate.sol_offsets, val_expected_neighbors = validate.sol_neighbors,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Eval] Labeling splits\n")
  local val_off, val_nbr = ridge_obj:label(val_codes, validate.n, 1)
  val_codes = nil -- luacheck: ignore

  local function eval_split (name, split)
    local pair_off, pair_tok, pair_val = tokenize_pairs(split)
    local codes = sp_enc:encode({
      offsets = pair_off, tokens = pair_tok, values = pair_val, n_samples = split.n,
    })
    local off, nbr = ridge_obj:label(codes, split.n, 1)
    local _, stats = eval.label_accuracy({
      pred_offsets = off, pred_neighbors = nbr,
      expected_offsets = split.sol_offsets, expected_neighbors = split.sol_neighbors,
      ks = 1,
    })
    str.printf("[%s] F1=%.4f precision=%.4f recall=%.4f %s\n",
      name, stats.micro_f1, stats.micro_precision, stats.micro_recall, sw())
    return stats
  end

  local _, val_stats = eval.label_accuracy({
    pred_offsets = val_off, pred_neighbors = val_nbr,
    expected_offsets = validate.sol_offsets, expected_neighbors = validate.sol_neighbors,
    ks = 1,
  })
  str.printf("[Val] F1=%.4f precision=%.4f recall=%.4f %s\n",
    val_stats.micro_f1, val_stats.micro_precision, val_stats.micro_recall, sw())

  eval_split("Dev", dev)
  eval_split("Test", test_set)

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
