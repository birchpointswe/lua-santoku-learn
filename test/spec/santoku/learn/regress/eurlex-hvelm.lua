local ann = require("santoku.learn.ann")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local hdc = require("santoku.learn.hdc")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local quantizer = require("santoku.learn.quantizer")
local ridge = require("santoku.learn.ridge")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local SCALE = 2

local cfg = {
  data = {
    max = nil
  },
  hdc = {
    d = 8192*SCALE,
    ngram = 7
  },
  ridge = {
    lambda = 3.8814e-05, --{ def = 3.8814e-05 },
    propensity_a = 0.41, --{ def = 0.41 },
    propensity_b = 0.18, --{ def = 0.18 },
    search_trials = 0,
  },
  emb = {
    k = 256,
    n_dims = 1024,
    binarize = "sign",
  },
  gfm1 = { beta = { def = 1.0 }, search_trials = 400, k = 256 },
  gfm2 = { beta = { def = 1.0 }, search_trials = 400, k = 256 },
  oof = { n_folds = 20 },
}

test("eurlex-hvelm", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local k1 = cfg.gfm1.k or n_labels
  str.printf("[Data] train=%d dev=%d test=%d labels=%d %s\n", train.n, dev.n, test_set.n, n_labels, sw())

  local train_label_off, train_label_nbr = train.solutions:bits_to_csr(train.n, n_labels)
  local dev_label_off, dev_label_nbr = dev.solutions:bits_to_csr(dev.n, n_labels)
  local test_label_off, test_label_nbr = test_set.solutions:bits_to_csr(test_set.n, n_labels)

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

  local function csr_topk(off, nbr, sco, k)
    if not k then return off, nbr, sco end
    local ns = off:size() - 1
    local new_off = ivec.create(ns + 1)
    local pos = 0
    for s = 0, ns - 1 do
      new_off:set(s, pos)
      local rlen = off:get(s + 1) - off:get(s)
      if rlen > k then rlen = k end
      pos = pos + rlen
    end
    new_off:set(ns, pos)
    local new_nbr = ivec.create(pos)
    local new_sco = dvec.create(pos)
    local di = 0
    for s = 0, ns - 1 do
      local si = off:get(s)
      local rlen = new_off:get(s + 1) - new_off:get(s)
      for j = 0, rlen - 1 do
        new_nbr:set(di, nbr:get(si + j))
        new_sco:set(di, sco:get(si + j))
        di = di + 1
      end
    end
    return new_off, new_nbr, new_sco
  end

  local function run_gfm(tag, gfm_cfg,
    tr_off, tr_nbr, tr_sco,
    dv_off, dv_nbr, dv_sco,
    ts_off, ts_nbr, ts_sco)
    local k = gfm_cfg.k
    tr_off, tr_nbr, tr_sco = csr_topk(tr_off, tr_nbr, tr_sco, k)
    dv_off, dv_nbr, dv_sco = csr_topk(dv_off, dv_nbr, dv_sco, k)
    ts_off, ts_nbr, ts_sco = csr_topk(ts_off, ts_nbr, ts_sco, k)
    local gfm_obj, gfm_p = optimize.gfm({
      pred_offsets = tr_off, pred_neighbors = tr_nbr,
      pred_scores = tr_sco, n_samples = train.n, n_labels = n_labels,
      expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
      val_offsets = dv_off, val_neighbors = dv_nbr,
      val_scores = dv_sco, val_n_samples = dev.n,
      val_expected_offsets = dev_label_off, val_expected_neighbors = dev_label_nbr,
      beta = gfm_cfg.beta,
      search_trials = gfm_cfg.search_trials,
      each = function (ev)
        local best = ev.global_best_score > -math.huge
          and str.format(" (best=%.4f%s)", ev.global_best_score, ev.score > ev.global_best_score + 1e-6 and " ++" or "")
          or ""
        str.printf("[%s %d/%d] beta=%.3f miF1=%.4f%s %s\n",
          tag, ev.trial, ev.trials, ev.params.beta,
          ev.metrics.micro_f1, best, sw())
      end,
    })
    str.printf("[%s] beta=%.3f %s\n", tag, gfm_p.beta, sw())
    local dk = gfm_obj:predict({
      offsets = dv_off, neighbors = dv_nbr, scores = dv_sco,
      n_samples = dev.n, beta = gfm_p.beta,
    })
    local dm = eval_oracle(dv_off, dv_nbr, dev_label_off, dev_label_nbr, dk)
    local tk = gfm_obj:predict({
      offsets = ts_off, neighbors = ts_nbr, scores = ts_sco,
      n_samples = test_set.n, beta = gfm_p.beta,
    })
    local tm = eval_oracle(ts_off, ts_nbr, test_label_off, test_label_nbr, tk)
    str.printf("[Dv %s] %s %s\n", tag, fmt_metrics(dm), sw())
    str.printf("[Ts %s] %s %s\n", tag, fmt_metrics(tm), sw())
    return dm, tm, gfm_p
  end

  local ngram_map, train_tok, n_hdc_tokens = hdc.tokenize({
    texts = train.problems, hdc_ngram = cfg.hdc.ngram, n_samples = train.n,
  })
  str.printf("[HDC] d=%d ngram=%d tokens=%d %s\n", cfg.hdc.d, cfg.hdc.ngram, n_hdc_tokens, sw())
  local bns_ids, bns_scores = train_tok:bits_top_bns(
    train.solutions, train.n, n_hdc_tokens, n_labels)
  train_tok = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[BNS] selected=%d %s\n", bns_ids:size(), sw())
  local hdc_enc, train_h = hdc.create({
    texts = train.problems, n_samples = train.n,
    d = cfg.hdc.d, hdc_ngram = cfg.hdc.ngram,
    weight_map = ngram_map,
    weight_ids = bns_ids,
    weights = bns_scores,
  })
  ngram_map = nil; bns_ids = nil; bns_scores = nil -- luacheck: ignore
  local dev_h = hdc_enc:encode({ texts = dev.problems, n_samples = dev.n })
  local test_h = hdc_enc:encode({ texts = test_set.problems, n_samples = test_set.n })
  local hdc_out_d = hdc_enc:out_d()
  str.printf("[HDC] encoded %s\n", sw())
  hdc_enc = nil -- luacheck: ignore

  local dv_short_off, dv_short_nbr, ts_short_off, ts_short_nbr, tr_short_off, tr_short_nbr

  if cfg.emb then

  str.printf("\n[Truncate]\n")
  local trunc_d = cfg.emb.n_dims
  local keep = ivec.create(trunc_d):fill_indices()
  local doc_embs = train_h:mtx_select(keep, nil, hdc_out_d)
  local dev_embs = dev_h:mtx_select(keep, nil, hdc_out_d)
  local test_embs = test_h:mtx_select(keep, nil, hdc_out_d)
  str.printf("[Truncate] dims=%d %s\n", trunc_d, sw())

  local post_n_bits, post_train_bin, binarize
  if cfg.emb.binarize == "sign" then
    post_train_bin, post_n_bits = doc_embs:mtx_sign(trunc_d)
    binarize = function (embs) return (embs:mtx_sign(trunc_d)) end
    str.printf("[Sign] bits=%d %s\n", post_n_bits, sw())
  else
    local post_itq = quantizer.create({
      mode = "itq", raw_codes = doc_embs, n_samples = train.n,
    })
    post_n_bits = post_itq:n_bits()
    post_train_bin = post_itq:encode(doc_embs)
    binarize = function (embs) return post_itq:encode(embs) end
    str.printf("[ITQ] bits=%d %s\n", post_n_bits, sw())
  end
  local doc_ann = ann.create({ features = post_n_bits })
  local doc_ids = ivec.create(train.n):fill_indices()
  doc_ann:add(post_train_bin, doc_ids)
  post_train_bin = nil -- luacheck: ignore
  str.printf("[ANN] indexed %s\n", sw())

  local K = cfg.emb.k
  local function shortlist(embs, n)
    local bin = binarize(embs)
    local hood_ids, ann_hoods = doc_ann:neighborhoods_by_vecs(bin, K)
    local a_off, a_nbr = ann_hoods:to_csr(post_n_bits)
    local union_bits = csr.label_union(
      a_off, a_nbr, hood_ids, train_label_off, train_label_nbr, n_labels)
    local sl_off, sl_nbr = union_bits:bits_to_csr(n, n_labels)
    return sl_off, sl_nbr
  end
  dv_short_off, dv_short_nbr = shortlist(dev_embs, dev.n)
  ts_short_off, ts_short_nbr = shortlist(test_embs, test_set.n)
  tr_short_off, tr_short_nbr = shortlist(doc_embs, train.n)
  doc_embs = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Shortlist] built %s\n", sw())

  local function shortlist_recall(tag, short_off, short_nbr, gt_off, gt_nbr, n)
    local ks = ivec.create(n)
    local total = 0
    for i = 0, n - 1 do
      local k = short_off:get(i + 1) - short_off:get(i)
      ks:set(i, k)
      total = total + k
    end
    local m = eval_oracle(short_off, short_nbr, gt_off, gt_nbr, ks)
    str.printf("[Shortlist %s] %s avg=%.1f %s\n", tag, fmt_metrics(m), total / n, sw())
  end
  shortlist_recall("Dev", dv_short_off, dv_short_nbr, dev_label_off, dev_label_nbr, dev.n)
  shortlist_recall("Test", ts_short_off, ts_short_nbr, test_label_off, test_label_nbr, test_set.n)

  end -- cfg.emb

  str.printf("\n[#1] BNS-HDC + OVA Ridge + Topk GFM\n")
  local r1_enc, ridge_obj, elm_params, _, r1_th, r1_dvh = optimize.ridge({
    elm = cfg.ridge.mode, n_hidden = cfg.ridge.n_hidden,
    codes = train_h, n_input_dims = hdc_out_d,
    n_samples = train.n,
    n_labels = n_labels,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    val_codes = dev_h, val_n_samples = dev.n,
    val_expected_offsets = dev_label_off, val_expected_neighbors = dev_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a, propensity_b = cfg.ridge.propensity_b,
    k = k1, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  local r1_tsh = r1_enc and r1_enc:encode({ codes = test_h, n_samples = test_set.n }) or test_h
  local r1_n_dims = r1_enc and r1_enc:out_d() or hdc_out_d
  train_h = nil; dev_h = nil; test_h = nil; r1_enc = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Ridge] mode=%s lambda=%.4e pa=%.4f pb=%.4f %s\n",
    elm_params.elm or "none", elm_params.lambda, elm_params.propensity_a, elm_params.propensity_b, sw())
  local oof_off, oof_nbr, oof_sco, oof_transform = ridge.oof({
    codes = r1_th, n_samples = train.n, n_dims = r1_n_dims,
    n_labels = n_labels,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    lambda = elm_params.lambda, propensity_a = elm_params.propensity_a,
    propensity_b = elm_params.propensity_b,
    k = k1, n_folds = cfg.oof.n_folds, transform = cfg.emb and true,
  })
  str.printf("[OOF] %d-fold %s\n", cfg.oof.n_folds, sw())
  local d1_tr_off, d1_tr_nbr = ridge_obj:label(r1_th, train.n, k1)
  local d1_dv_off, d1_dv_nbr, d1_dv_sco = ridge_obj:label(r1_dvh, dev.n, k1)
  local d1_ts_off, d1_ts_nbr, d1_ts_sco = ridge_obj:label(r1_tsh, test_set.n, k1)

  local d1_tr_oracle = eval_oracle(d1_tr_off, d1_tr_nbr, train_label_off, train_label_nbr)
  local d1_oof_oracle = eval_oracle(oof_off, oof_nbr, train_label_off, train_label_nbr)
  local d1_dv_oracle = eval_oracle(d1_dv_off, d1_dv_nbr, dev_label_off, dev_label_nbr)
  local d1_ts_oracle = eval_oracle(d1_ts_off, d1_ts_nbr, test_label_off, test_label_nbr)
  str.printf("[Tr Orc]  %s\n", fmt_metrics(d1_tr_oracle))
  str.printf("[Oo Orc] %s\n", fmt_metrics(d1_oof_oracle))
  str.printf("[Dv Orc]  %s\n", fmt_metrics(d1_dv_oracle))
  str.printf("[Ts Orc]  %s %s\n", fmt_metrics(d1_ts_oracle), sw())

  local gfm1_dm, gfm1_tm, gfm1_p = run_gfm("GFM", cfg.gfm1,
    oof_off, oof_nbr, oof_sco,
    d1_dv_off, d1_dv_nbr, d1_dv_sco,
    d1_ts_off, d1_ts_nbr, d1_ts_sco)

  oof_off = nil; oof_nbr = nil; oof_sco = nil -- luacheck: ignore
  d1_dv_off = nil; d1_dv_nbr = nil; d1_dv_sco = nil -- luacheck: ignore
  d1_ts_off = nil; d1_ts_nbr = nil; d1_ts_sco = nil -- luacheck: ignore
  collectgarbage("collect")

  local d2_tr_oracle, d2_oof_oracle, d2_dv_oracle, d2_ts_oracle
  local gfm2_dm, gfm2_tm, gfm2_p

  if cfg.emb then

  str.printf("\n[#2] BNS-HDC + Truncate + ITQ + ANN Shortlist + OVA Ridge + Topk GFM\n")
  str.printf("[OVA] Scoring shortlists\n")
  local dv_short_scores = ridge_obj:regress(r1_dvh, dev.n, dv_short_off, dv_short_nbr)
  r1_dvh = nil -- luacheck: ignore
  local dv_sorted_nbr, dv_sorted_sc = csr.sort_csr_desc(dv_short_off, dv_short_nbr, dv_short_scores)
  local ts_short_scores = ridge_obj:regress(r1_tsh, test_set.n, ts_short_off, ts_short_nbr)
  r1_tsh = nil -- luacheck: ignore
  local ts_sorted_nbr, ts_sorted_sc = csr.sort_csr_desc(ts_short_off, ts_short_nbr, ts_short_scores)
  local tr_short_scores_is = ridge_obj:regress(r1_th, train.n, tr_short_off, tr_short_nbr)
  local tr_sorted_nbr_is = csr.sort_csr_desc(tr_short_off, tr_short_nbr, tr_short_scores_is)
  local d2_tr_oracle = eval_oracle(tr_short_off, tr_sorted_nbr_is, train_label_off, train_label_nbr)
  tr_short_scores_is = nil; tr_sorted_nbr_is = nil -- luacheck: ignore
  local tr_short_scores = eval.gather_dense({
    scores = oof_transform, n_labels = n_labels,
    offsets = tr_short_off, neighbors = tr_short_nbr,
  })
  oof_transform = nil -- luacheck: ignore
  r1_th = nil; ridge_obj = nil -- luacheck: ignore
  collectgarbage("collect")
  local tr_sorted_nbr, tr_sorted_sc = csr.sort_csr_desc(tr_short_off, tr_short_nbr, tr_short_scores)
  str.printf("[OVA] scored %s\n", sw())

  local d2_oof_oracle = eval_oracle(tr_short_off, tr_sorted_nbr, train_label_off, train_label_nbr)
  local d2_dv_oracle = eval_oracle(dv_short_off, dv_sorted_nbr, dev_label_off, dev_label_nbr)
  local d2_ts_oracle = eval_oracle(ts_short_off, ts_sorted_nbr, test_label_off, test_label_nbr)
  str.printf("[Tr Orc]  %s\n", fmt_metrics(d2_tr_oracle))
  str.printf("[Oo Orc] %s\n", fmt_metrics(d2_oof_oracle))
  str.printf("[Dv Orc]  %s\n", fmt_metrics(d2_dv_oracle))
  str.printf("[Ts Orc]  %s %s\n", fmt_metrics(d2_ts_oracle), sw())

  local gfm2_dm, gfm2_tm, gfm2_p = run_gfm("GFM", cfg.gfm2,
    tr_short_off, tr_sorted_nbr, tr_sorted_sc,
    dv_short_off, dv_sorted_nbr, dv_sorted_sc,
    ts_short_off, ts_sorted_nbr, ts_sorted_sc)

  else
  oof_transform = nil; r1_dvh = nil; r1_tsh = nil -- luacheck: ignore
  r1_th = nil; ridge_obj = nil -- luacheck: ignore
  collectgarbage("collect")
  end -- cfg.emb

  str.printf("\n#1: BNS-HDC + OVA Ridge + Topk GFM\n")
  if elm_params.elm then
    str.printf("  %-10s mode=%s input=%d hidden=%d\n", "ELM", elm_params.elm, hdc_out_d, cfg.ridge.n_hidden)
  end
  str.printf("  %-10s lambda=%.4e pa=%.4f pb=%.4f\n",
    "Ridge", elm_params.lambda, elm_params.propensity_a, elm_params.propensity_b)
  str.printf("  %-10s beta=%.3f\n", "GFM", gfm1_p.beta)
  str.printf("  %-10s %s\n", "Tr Orc", fmt_metrics(d1_tr_oracle))
  str.printf("  %-10s %s\n", "OOF Orc", fmt_metrics(d1_oof_oracle))
  str.printf("  %-10s %s\n", "Dv Orc", fmt_metrics(d1_dv_oracle))
  str.printf("  %-10s %s\n", "Ts Orc", fmt_metrics(d1_ts_oracle))
  str.printf("  %-10s %s\n", "Dv GFM", fmt_metrics(gfm1_dm))
  str.printf("  %-10s %s\n", "Ts GFM", fmt_metrics(gfm1_tm))

  if cfg.emb then
  str.printf("\n#2: BNS-HDC + Truncate + ITQ + ANN Shortlist + OVA Ridge + Topk GFM\n")
  if elm_params.elm then
    str.printf("  %-10s mode=%s input=%d hidden=%d\n", "ELM", elm_params.elm, hdc_out_d, cfg.ridge.n_hidden)
  end
  str.printf("  %-10s lambda=%.4e pa=%.4f pb=%.4f\n",
    "Ridge", elm_params.lambda, elm_params.propensity_a, elm_params.propensity_b)
  str.printf("  %-10s beta=%.3f\n", "GFM", gfm2_p.beta)
  str.printf("  %-10s %s\n", "Tr Orc", fmt_metrics(d2_tr_oracle))
  str.printf("  %-10s %s\n", "OOF Orc", fmt_metrics(d2_oof_oracle))
  str.printf("  %-10s %s\n", "Dv Orc", fmt_metrics(d2_dv_oracle))
  str.printf("  %-10s %s\n", "Ts Orc", fmt_metrics(d2_ts_oracle))
  str.printf("  %-10s %s\n", "Dv GFM", fmt_metrics(gfm2_dm))
  str.printf("  %-10s %s\n", "Ts GFM", fmt_metrics(gfm2_tm))
  end -- cfg.emb

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
