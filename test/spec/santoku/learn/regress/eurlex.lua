local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local fvec = require("santoku.fvec")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil },
  tok = { ngram = 6 },
  -- n_landmarks = 1024*24 for max (high memory)
  emb = { n_landmarks = 1024*8, trace_tol = 0.01, kernel = { "cosine", "nngp", "ntk", "expcos", "geolaplace" }, k = 256 },
  ridge = {
    lambda = { def = 2.73e-02 },
    propensity_a = { def = 0.04 },
    propensity_b = { def = 6.31 },
    search_trials = 0,
  },
}

test("eurlex classifier", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  local function fmt_metrics(m)
    return str.format("miP=%.4f miR=%.4f miF1=%.4f",
      m.micro_precision, m.micro_recall, m.micro_f1)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local k = cfg.emb.k or n_labels
  str.printf("[Data] train=%d dev=%d test=%d labels=%d %s\n", train.n, dev.n, test_set.n, n_labels, sw())

  local train_label_off, train_label_nbr = train.sol_offsets, train.sol_neighbors
  local dev_label_off, dev_label_nbr = dev.sol_offsets, dev.sol_neighbors
  local test_label_off, test_label_nbr = test_set.sol_offsets, test_set.sol_neighbors

  local ngram_map, offsets, tokens, values, n_tokens = csr.tokenize({
    texts = train.text_iter(), ngram = cfg.tok.ngram, n_samples = train.n,
  })
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    train_label_off, train_label_nbr, n_tokens, n_labels)
  str.printf("[Tokenize] ngram=%d tokens=%d %s\n", cfg.tok.ngram, n_tokens, sw())

  local _, val_off, val_tok, val_val = csr.tokenize({
    texts = dev.text_iter(), ngram = cfg.tok.ngram, n_samples = dev.n, ngram_map = ngram_map,
  })
  csr.apply_bns(val_off, val_tok, val_val, bns_scores)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local chol_path = "test/res/eurlex57k/chol_tmp"
  local w_path = "test/res/eurlex57k/w_tmp"
  local chol_buf = fvec.mmap_create(chol_path, cfg.emb.n_landmarks * train.n)
  local w_buf = fvec.mmap_create(w_path, cfg.emb.n_landmarks * n_labels)
  local pqty_path = "test/res/eurlex57k/pqty_tmp"
  local pqty_buf = cfg.ridge.search_trials > 0
    and fvec.mmap_create(pqty_path, cfg.emb.n_landmarks * n_labels) or nil
  local sp_enc, ridge_obj, dev_codes, best_params = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
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
    trial_fn = function (gram, params)
      local f1, p, r = gram:label_accuracy(params.lambda, k,
        params.propensity_a, params.propensity_b,
        dev_label_off, dev_label_nbr, "gfm")
      return f1, { f1 = f1, precision = p, recall = r }
    end,
  })
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  chol_buf = nil; w_buf = nil; pqty_buf = nil -- luacheck: ignore
  collectgarbage("collect")
  os.remove(chol_path)
  os.remove(w_path)
  os.remove(pqty_path)
  local emb_d = sp_enc:dims()
  str.printf("[KRR] emb_d=%d kernel=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    emb_d, best_params.kernel, best_params.lambda,
    best_params.propensity_a, best_params.propensity_b, sw())

  local function encode_texts(text_iter_fn, n)
    local _, off, tok, val = csr.tokenize({
      texts = text_iter_fn(), ngram = cfg.tok.ngram, n_samples = n, ngram_map = ngram_map,
    })
    csr.apply_bns(off, tok, val, bns_scores)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  local dv_off, dv_nbr, dv_sco = ridge_obj:label(dev_codes, dev.n, k)
  local _, dv_oracle = eval.label_accuracy({
    pred_offsets = dv_off, pred_neighbors = dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  str.printf("[Dv Oracle] %s %s\n", fmt_metrics(dv_oracle), sw())

  local gfm_obj = optimize.gfm({
    n_labels = n_labels,
    val_offsets = dv_off, val_neighbors = dv_nbr, val_scores = dv_sco,
    val_n_samples = dev.n,
    val_expected_offsets = dev_label_off, val_expected_neighbors = dev_label_nbr,
  })

  dev_codes = nil -- luacheck: ignore
  collectgarbage("collect")

  local test_codes = encode_texts(test_set.text_iter, test_set.n)
  local ts_off, ts_nbr, ts_sco = ridge_obj:label(test_codes, test_set.n, k)
  test_codes = nil; ridge_obj:shrink(); sp_enc:shrink() -- luacheck: ignore
  collectgarbage("collect")

  local _, ts_oracle = eval.label_accuracy({
    pred_offsets = ts_off, pred_neighbors = ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
  })
  str.printf("[Ts Oracle] %s %s\n", fmt_metrics(ts_oracle), sw())

  local ts_ks = gfm_obj:predict({ offsets = ts_off, neighbors = ts_nbr, scores = ts_sco, n_samples = test_set.n })
  local _, ts_pred_m = eval.label_accuracy({
    pred_offsets = ts_off, pred_neighbors = ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    ks = ts_ks,
  })
  str.printf("[Ts Pred] %s %s\n", fmt_metrics(ts_pred_m), sw())

  str.printf("\nSummary\n")
  str.printf("  %-10s lambda=%.4e pa=%.4f pb=%.4f\n",
    "Params", best_params.lambda, best_params.propensity_a, best_params.propensity_b)
  str.printf("  %-10s %s\n", "Dv Oracle", fmt_metrics(dv_oracle))
  str.printf("  %-10s %s\n", "Ts Oracle", fmt_metrics(ts_oracle))
  str.printf("  %-10s %s\n", "Ts Pred", fmt_metrics(ts_pred_m))
  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
