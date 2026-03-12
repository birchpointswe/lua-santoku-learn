local ann = require("santoku.learn.ann")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local fvec = require("santoku.fvec")
local ivec = require("santoku.ivec")
local gfm = require("santoku.learn.gfm")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil },
  tok = { ngram = 7 },
  emb = {
    n_landmarks = 8192*2,
    trace_tol = 0.01,
    kernel = "arccos1",
    k = 256,
  },
  ridge = {
    lambda = { def = 6.9568e-05 },
    propensity_a = { def = 0.10 },
    propensity_b = { def = 0.88 },
    search_trials = nil,
  },
  gfm = true,
  -- ann = true,
}

test("eurlex", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local k = cfg.emb.k or n_labels
  str.printf("[Data] train=%d dev=%d test=%d labels=%d %s\n", train.n, dev.n, test_set.n, n_labels, sw())

  local train_label_off, train_label_nbr = train.sol_offsets, train.sol_neighbors
  local dev_label_off, dev_label_nbr = dev.sol_offsets, dev.sol_neighbors
  local test_label_off, test_label_nbr = test_set.sol_offsets, test_set.sol_neighbors

  local function eval_oracle(pred_off, pred_nbr, exp_off, exp_nbr, ks)
    local _, m = eval.retrieval_ks({
      pred_offsets = pred_off, pred_neighbors = pred_nbr,
      expected_offsets = exp_off, expected_neighbors = exp_nbr,
      ks = ks,
    })
    return m
  end

  local function fmt_metrics(m)
    return str.format("miP=%.4f miR=%.4f miF1=%.4f saP=%.4f saR=%.4f saF1=%.4f",
      m.micro_precision, m.micro_recall, m.micro_f1,
      m.sample_precision, m.sample_recall, m.sample_f1)
  end

  local function csr_topk(off, nbr, sco, tk)
    if not tk then return off, nbr, sco end
    return csr.truncate(off, nbr, sco, tk)
  end

  local function run_gfm(tag, gfm_k,
    tr_off, tr_nbr, tr_sco,
    ts_off, ts_nbr, ts_sco,
    dv_off, dv_nbr, dv_sco)
    tr_off, tr_nbr, tr_sco = csr_topk(tr_off, tr_nbr, tr_sco, gfm_k)
    ts_off, ts_nbr, ts_sco = csr_topk(ts_off, ts_nbr, ts_sco, gfm_k)
    local gfm_obj = gfm.create({
      expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
      n_samples = train.n, n_labels = n_labels,
    })
    gfm_obj:fit({
      pred_offsets = tr_off, pred_neighbors = tr_nbr, pred_scores = tr_sco,
    })
    str.printf("[%s] fit %s\n", tag, sw())
    local trk = gfm_obj:predict({
      offsets = tr_off, neighbors = tr_nbr, scores = tr_sco,
      n_samples = train.n,
    })
    local trm = eval_oracle(tr_off, tr_nbr, train_label_off, train_label_nbr, trk)
    str.printf("[Tr %s] %s %s\n", tag, fmt_metrics(trm), sw())
    local dm
    if dv_off then
      dv_off, dv_nbr, dv_sco = csr_topk(dv_off, dv_nbr, dv_sco, gfm_k)
      local dk = gfm_obj:predict({
        offsets = dv_off, neighbors = dv_nbr, scores = dv_sco,
        n_samples = dev.n,
      })
      dm = eval_oracle(dv_off, dv_nbr, dev_label_off, dev_label_nbr, dk)
      str.printf("[Dv %s] %s %s\n", tag, fmt_metrics(dm), sw())
    end
    local tk = gfm_obj:predict({
      offsets = ts_off, neighbors = ts_nbr, scores = ts_sco,
      n_samples = test_set.n,
    })
    local tm = eval_oracle(ts_off, ts_nbr, test_label_off, test_label_nbr, tk)
    str.printf("[Ts %s] %s %s\n", tag, fmt_metrics(tm), sw())
    return trm, dm, tm, gfm_obj
  end

  local ngram_map, offsets, tokens, values, n_tokens = csr.tokenize({
    texts = train.text_iter(), hdc_ngram = cfg.tok.ngram, n_samples = train.n,
  })
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    train_label_off, train_label_nbr, n_tokens, n_labels)
  str.printf("[Tokenize] ngram=%d tokens=%d %s\n",
    cfg.tok.ngram, n_tokens, sw())

  str.printf("[Spectral] Cholesky trace_tol=%s kernel=%s\n",
    tostring(cfg.emb.trace_tol), cfg.emb.kernel)
  local train_codes, sp_enc = spectral.encode({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
  })
  offsets = nil; tokens = nil; values = nil
  collectgarbage("collect")
  local emb_d = sp_enc:dims()
  str.printf("[Spectral] emb_d=%d %s\n", emb_d, sw())

  local function encode_texts(text_iter_fn, n)
    local _, off, tok, val = csr.tokenize({
      texts = text_iter_fn(), hdc_ngram = cfg.tok.ngram,
      n_samples = n, ngram_map = ngram_map,
    })
    csr.apply_bns(off, tok, val, bns_scores)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  local dev_codes = encode_texts(dev.text_iter, dev.n)
  local test_codes = encode_texts(test_set.text_iter, test_set.n)

  local dv_short_off, dv_short_nbr, ts_short_off, ts_short_nbr, tr_short_off, tr_short_nbr
  if cfg.ann then
    local train_sign, n_ann_bits = train_codes:mtx_sign(emb_d)
    local doc_ids = ivec.create(train.n):fill_indices()
    local mih = ann.create({ data = train_sign, features = n_ann_bits, codes = train_codes, n_dims = emb_d })
    str.printf("[ANN] indexed %s\n", sw())
    local K = cfg.emb.k
    local function shortlist(codes, n)
      local sign_cvec = codes:mtx_sign(emb_d)
      local a_off, a_nbr = mih:neighborhoods_by_vecs(sign_cvec, K, codes)
      local sl_off, sl_nbr = csr.label_union(a_off, a_nbr, doc_ids, train_label_off, train_label_nbr, n_labels)
      return sl_off, sl_nbr
    end
    dv_short_off, dv_short_nbr = shortlist(dev_codes, dev.n)
    ts_short_off, ts_short_nbr = shortlist(test_codes, test_set.n)
    tr_short_off, tr_short_nbr = shortlist(train_codes, train.n)
    train_sign = nil
    collectgarbage("collect")
    str.printf("[Shortlist] built %s\n", sw())
  end

  sp_enc:shrink()
  collectgarbage("collect")

  str.printf("\n[Ridge]\n")
  local ridge_obj, best_params = optimize.ridge({
    train_codes = train_codes, n_samples = train.n, n_dims = emb_d,
    label_offsets = train_label_off, label_neighbors = train_label_nbr, n_labels = n_labels,
    val_codes = dev_codes, val_n_samples = dev.n,
    val_expected_offsets = dev_label_off, val_expected_neighbors = dev_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a, propensity_b = cfg.ridge.propensity_b,
    k = k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("[Ridge] lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best_params.lambda, best_params.propensity_a, best_params.propensity_b, sw())

  local tr_off, tr_nbr, tr_sco = ridge_obj:label(train_codes, train.n, k, tr_short_off, tr_short_nbr)
  local dv_off, dv_nbr, dv_sco = ridge_obj:label(dev_codes, dev.n, k, dv_short_off, dv_short_nbr)
  local ts_off, ts_nbr, ts_sco = ridge_obj:label(test_codes, test_set.n, k, ts_short_off, ts_short_nbr)
  train_codes = nil
  collectgarbage("collect")

  local dv_oracle = eval_oracle(dv_off, dv_nbr, dev_label_off, dev_label_nbr)
  local ts_oracle = eval_oracle(ts_off, ts_nbr, test_label_off, test_label_nbr)
  str.printf("[Dv Orc] %s\n", fmt_metrics(dv_oracle))
  str.printf("[Ts Orc] %s %s\n", fmt_metrics(ts_oracle), sw())

  local gfm_trm, gfm_dvm, gfm_tsm
  if cfg.gfm then
    gfm_trm, gfm_dvm, gfm_tsm = run_gfm("GFM", k,
      tr_off, tr_nbr, tr_sco,
      ts_off, ts_nbr, ts_sco,
      dv_off, dv_nbr, dv_sco)
  end

  str.printf("\nSummary\n")
  str.printf("  %-10s lambda=%.4e pa=%.4f pb=%.4f\n",
    "Ridge", best_params.lambda, best_params.propensity_a, best_params.propensity_b)
  str.printf("  %-10s %s\n", "Dv Orc", fmt_metrics(dv_oracle))
  str.printf("  %-10s %s\n", "Ts Orc", fmt_metrics(ts_oracle))
  if gfm_trm then str.printf("  %-10s %s\n", "Tr GFM", fmt_metrics(gfm_trm)) end
  if gfm_dvm then str.printf("  %-10s %s\n", "Dv GFM", fmt_metrics(gfm_dvm)) end
  if gfm_tsm then str.printf("  %-10s %s\n", "Ts GFM", fmt_metrics(gfm_tsm)) end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
