local csr = require("santoku.learn.csr")
local tokenizer = require("santoku.learn.tokenizer")
local function tokenize (iter, n, ng, tok)
  local texts, x = {}, iter()
  while x do texts[#texts + 1] = x; x = iter() end
  local grow = tok == nil
  if grow then tok = tokenizer.create({ ngram_min = ng, ngram_max = ng }) end
  local o, t, v = tok:tokenize({ texts = texts, n_samples = n, grow = grow })
  return tok, o, t, v, tok:n_tokens()
end
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local fvec = require("santoku.fvec")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = {
    max = nil
  },
  tok = {
    ngram = 6
  },
  emb = {
    n_landmarks = 1024 * 8,
    trace_tol = 0.01,
    kernel = { "rbf", "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1" },
    gamma = { def = 0.3163 },
    k = 256
  },
  ridge = {
    lambda = { def = 2.3652e-04 },
    propensity_a = { def = 0.0536 },
    propensity_b = { def = 13.1522 },
    search_trials = 0
  },
}

test("eurlex classifier", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  local fmt_metrics = util.fmt_metrics

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local k = cfg.emb.k or n_labels
  str.printf("[Data] train=%d dev=%d test=%d labels=%d %s\n", train.n, dev.n, test_set.n, n_labels, sw())

  local train_label_off, train_label_nbr = train.sol_offsets, train.sol_neighbors
  local dev_label_off, dev_label_nbr = dev.sol_offsets, dev.sol_neighbors
  local test_label_off, test_label_nbr = test_set.sol_offsets, test_set.sol_neighbors

  local ngram_map, offsets, tokens, values, n_tokens =
    tokenize(train.text_iter(), train.n, cfg.tok.ngram)
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    train_label_off, train_label_nbr, n_tokens, n_labels)
  csr.normalize(offsets, values)
  str.printf("[Tokenize] ngram=%d tokens=%d %s\n", cfg.tok.ngram, n_tokens, sw())

  local _, val_off, val_tok, val_val =
    tokenize(dev.text_iter(), dev.n, cfg.tok.ngram, ngram_map)
  csr.apply_bns(val_off, val_tok, val_val, bns_scores)
  csr.normalize(val_off, val_val)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local chol_path = "test/res/eurlex57k/chol_tmp"
  local w_path = "test/res/eurlex57k/w_tmp"
  local chol_buf = fvec.mmap_create(chol_path, cfg.emb.n_landmarks * train.n)
  local w_buf = fvec.mmap_create(w_path, cfg.emb.n_landmarks * n_labels)
  local pqty_path = "test/res/eurlex57k/pqty_tmp"
  local pqty_buf = cfg.ridge.search_trials > 0
    and function (kname) return fvec.mmap_create(pqty_path .. "_" .. kname, cfg.emb.n_landmarks * n_labels) end
    or nil
  local sp_enc, ridge_obj, dev_codes, best_params, decider, dec_metrics = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel, rbf_gamma = cfg.emb.gamma,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = train_label_off, label_neighbors = train_label_nbr, n_labels = n_labels,
    val_offsets = val_off, val_tokens = val_tok, val_values = val_val,
    val_n_samples = dev.n,
    val_expected_offsets = dev_label_off, val_expected_neighbors = dev_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = k, search_trials = cfg.ridge.search_trials, tile_labels = 1024,
    chol_buf = chol_buf, w_buf = w_buf, pqty_buf = pqty_buf,
    each = util.make_ridge_log(stopwatch),
  })
  do
    local pth = os.tmpname()
    sp_enc:persist(pth)
    local enc2 = spectral.load(pth)
    os.remove(pth)
    local vc2 = enc2:encode({ offsets = val_off, tokens = val_tok, values = val_val, n_samples = dev.n })
    local nchk = dev_codes:size()
    if nchk > 100000 then nchk = 100000 end
    for i = 0, nchk - 1 do
      assert(dev_codes:get(i) == vc2:get(i), "persist/load parity mismatch at " .. i)
    end
    str.printf("[Persist] load parity OK (%d codes)\n", nchk)
  end
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  chol_buf = nil; w_buf = nil; pqty_buf = nil -- luacheck: ignore
  collectgarbage("collect")
  os.remove(chol_path)
  os.remove(w_path)
  local kernels = type(cfg.emb.kernel) == "table" and cfg.emb.kernel or { cfg.emb.kernel }
  for _, kn in ipairs(kernels) do os.remove(pqty_path .. "_" .. kn) end
  local function encode_texts(text_iter_fn, n)
    local _, off, tok, val =
      tokenize(text_iter_fn(), n, cfg.tok.ngram, ngram_map)
    csr.apply_bns(off, tok, val, bns_scores)
    csr.normalize(off, val)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  local dv_off, dv_nbr = ridge_obj:label(dev_codes, dev.n, k)
  local _, dv_oracle = eval.oracle_f1({
    pred_offsets = dv_off, pred_neighbors = dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  str.printf("[Oracle] dev %s %s\n", fmt_metrics(dv_oracle), sw())

  dev_codes = nil -- luacheck: ignore
  collectgarbage("collect")

  local test_codes = encode_texts(test_set.text_iter, test_set.n)
  local ts_off, ts_nbr, ts_sco = ridge_obj:label(test_codes, test_set.n, k)
  test_codes = nil; ridge_obj:shrink(); sp_enc:shrink() -- luacheck: ignore
  collectgarbage("collect")

  local _, ts_oracle = eval.oracle_f1({
    pred_offsets = ts_off, pred_neighbors = ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
  })
  str.printf("[Oracle] test %s %s\n", fmt_metrics(ts_oracle), sw())

  local _, ts_pred_m = decider:score({
    offsets = ts_off, neighbors = ts_nbr, scores = ts_sco,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    n_samples = test_set.n,
  })
  str.printf("[Decide] val %s | test %s %s\n",
    fmt_metrics(dec_metrics), fmt_metrics(ts_pred_m), sw())

  str.printf("\nSummary\n")
  str.printf("  %-12s lambda=%.4e pa=%.4f pb=%.4f\n",
    "Params", best_params.lambda, best_params.propensity_a, best_params.propensity_b)
  str.printf("  %-12s dev %s | test %s\n", "Oracle", fmt_metrics(dv_oracle), fmt_metrics(ts_oracle))
  str.printf("  %-12s val %s | test %s\n", "Decide", fmt_metrics(dec_metrics), fmt_metrics(ts_pred_m))
  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
