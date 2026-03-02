local ann = require("santoku.learn.ann")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local gfm = require("santoku.learn.gfm")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local quantizer = require("santoku.learn.quantizer")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil, toks_per_class = 1024, toks_overall = nil },
  tokenizer = {
    max_len = 20, min_len = 1, max_run = 2,
    ngrams = 2, cgrams_min = 0, cgrams_max = 0,
    cgrams_cross = false, skips = 1,
  },
  elm_direct = {
    norm = "l2",
    mode = "sigmoid",
    n_hidden = 8192,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 200, k = 32, n_folds = 10,
  },
  emb = {
    k = 32, n_folds = 10, n_landmarks = 4096, n_dims = 256,
    distill = {
      norm = "none",
      mode = "sin",
      n_hidden = 8192,
      lambda = { def = 1.0 }, search_trials = 200,
    },
  },
  gfm = {
    beta = { min = 0.5, max = 3.0 },
    gamma = { min = 0.3, max = 3.0 },
    search_trials = 200,
  },
}

test("eurlex-aligned", function ()

  local stopwatch = utc.stopwatch()
  local _, dv_gfm, ts_gfm
  local dv_ann_orc, dv_ann_gfm, ts_ann_gfm

  print("Loading data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  str.printf("  Train: %d  Dev: %d  Test: %d  Labels: %d\n", train.n, dev.n, test_set.n, n_labels)

  print("\nBuilding label CSR")
  local train_label_off, train_label_nbr = train.solutions:bits_to_csr(train.n, n_labels)
  local dev_label_off, dev_label_nbr = dev.solutions:bits_to_csr(dev.n, n_labels)
  local test_label_off, test_label_nbr = test_set.solutions:bits_to_csr(test_set.n, n_labels)
  local function eval_oracle(pred_off, pred_nbr, exp_off, exp_nbr, ks)
    local _, oracle = eval.retrieval_ks({
      pred_offsets = pred_off, pred_neighbors = pred_nbr,
      expected_offsets = exp_off, expected_neighbors = exp_nbr,
      ks = ks,
    })
    return oracle
  end

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens_raw = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  dev.problems = nil
  test_set.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens_raw)

  print("\nFeature selection for text->labels")
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens_raw, n_labels, cfg.data.toks_per_class, cfg.data.toks_overall, "max")
  local n_bns_tokens = bns_ids:size()
  str.printf("  Selected features: %d\n", n_bns_tokens)

  local function make_bns_csc(tokens, n)
    local bits = ivec.create():copy(tokens)
    bits:bits_select(bns_ids, nil, n_tokens_raw)
    return csr.to_csc(bits, n, n_bns_tokens)
  end
  local train_bns_off, train_bns_idx = make_bns_csc(train.tokens, train.n)
  local dev_bns_off, dev_bns_idx = make_bns_csc(dev.tokens, dev.n)
  local test_bns_off, test_bns_idx = make_bns_csc(test_set.tokens, test_set.n)

  local function print_result(label, m)
    str.printf("  %-10s miP=%.4f miR=%.4f miF1=%.4f maP=%.4f maR=%.4f maF1=%.4f\n",
      label, m.micro_precision, m.micro_recall, m.micro_f1,
      m.macro_precision, m.macro_recall, m.macro_f1)
  end

  local at_ks = { 1, 2, 3, 5, 8, 10, 15, 20, 25, 32 }
  local function compute_at_ks(scores, n, nl, exp_off, exp_nbr)
    local diag_ks = ivec.create()
    for _, k in ipairs(at_ks) do diag_ks:push(k) end
    return eval.scores_at_ks({
      scores = scores, n_samples = n, n_labels = nl,
      expected_offsets = exp_off, expected_neighbors = exp_nbr, ks = diag_ks,
    })
  end
  local function print_at_ks_table(diag)
    for _, k in ipairs(at_ks) do
      local m = diag[k]
      str.printf("  @%-3d miP=%.3f miR=%.3f miF1=%.3f maF1=%.3f NDCG=%.3f\n",
        k, m.micro_precision, m.micro_recall, m.micro_f1, m.macro_f1, m.ndcg)
    end
  end
  local function print_k_err(label, pred_ks, oracle_ks, n)
    local err_sum, abs_sum = 0, 0
    local hi, lo, eq = 0, 0, 0
    for i = 0, n - 1 do
      local e = pred_ks:get(i) - oracle_ks:get(i)
      err_sum = err_sum + e
      abs_sum = abs_sum + math.abs(e)
      if e > 0 then hi = hi + 1
      elseif e < 0 then lo = lo + 1
      else eq = eq + 1 end
    end
    str.printf("  %-10s k MAE=%.2f bias=%.2f exact=%d high=%d low=%d\n",
      label, abs_sum / n, err_sum / n, eq, hi, lo)
  end

  local function print_k_buckets(label, pred_ks, oracle_ks, n)
    local buckets = {}
    local bounds = { 1, 2, 3, 4, 5, 8, 12, 20, 999 }
    local bnames = { "1", "2", "3", "4", "5", "6-8", "9-12", "13-20", "21+" }
    for b = 1, #bounds do
      buckets[b] = { cnt = 0, sum_err = 0, sum_abs = 0, sum_sq = 0, hi = 0, lo = 0, eq = 0 }
    end
    for i = 0, n - 1 do
      local ok = oracle_ks:get(i)
      local e = pred_ks:get(i) - ok
      local bi = #bounds
      for b = 1, #bounds do
        if ok <= bounds[b] then bi = b; break end
      end
      local bk = buckets[bi]
      bk.cnt = bk.cnt + 1
      bk.sum_err = bk.sum_err + e
      bk.sum_abs = bk.sum_abs + math.abs(e)
      bk.sum_sq = bk.sum_sq + e * e
      if e > 0 then bk.hi = bk.hi + 1
      elseif e < 0 then bk.lo = bk.lo + 1
      else bk.eq = bk.eq + 1 end
    end
    str.printf("  %s k-error by oracle-k bucket:\n", label)
    str.printf("  %-6s %6s %7s %7s %7s %5s %5s %5s\n",
      "k*", "n", "bias", "MAE", "RMSE", "hi", "lo", "eq")
    for b = 1, #bounds do
      local bk = buckets[b]
      if bk.cnt > 0 then
        str.printf("  %-6s %6d %+7.2f %7.2f %7.2f %5d %5d %5d\n",
          bnames[b], bk.cnt, bk.sum_err / bk.cnt, bk.sum_abs / bk.cnt,
          math.sqrt(bk.sum_sq / bk.cnt), bk.hi, bk.lo, bk.eq)
      end
    end
  end

  local function print_mu_by_k(label, mu, oracle_ks, n)
    local bounds = { 1, 2, 3, 4, 5, 8, 12, 20, 999 }
    local bnames = { "1", "2", "3", "4", "5", "6-8", "9-12", "13-20", "21+" }
    local bk = {}
    for b = 1, #bounds do bk[b] = {} end
    for i = 0, n - 1 do
      local ok = oracle_ks:get(i)
      local m = mu:get(i)
      for b = 1, #bounds do
        if ok <= bounds[b] then bk[b][#bk[b] + 1] = m; break end
      end
    end
    str.printf("  %s raw mu_hat by oracle-k*:\n", label)
    str.printf("  %-6s %6s %7s %7s %7s %7s %7s\n", "k*", "n", "mean", "std", "p10", "p50", "p90")
    for b = 1, #bounds do
      local t = bk[b]
      if #t > 0 then
        table.sort(t)
        local s, s2 = 0, 0
        for j = 1, #t do s = s + t[j]; s2 = s2 + t[j] * t[j] end
        local mn = s / #t
        local sd = math.sqrt(math.max(0, s2 / #t - mn * mn))
        str.printf("  %-6s %6d %7.2f %7.2f %7.2f %7.2f %7.2f\n",
          bnames[b], #t, mn, sd,
          t[math.max(1, math.floor(#t * 0.1))],
          t[math.max(1, math.floor(#t * 0.5))],
          t[math.max(1, math.floor(#t * 0.9))])
      end
    end
  end

  local function eval_elm(obj, train_h, k, dense_dv, dense_ts)
    local dv_off, dv_nbr = obj:label(dev_bns_off, dev_bns_idx, dev.n, k, dense_dv)
    local dv_oracle = eval_oracle(dv_off, dv_nbr, dev_label_off, dev_label_nbr)
    print_result("Oracle", dv_oracle)
    return dv_oracle
  end

  print("\n#1: ELM text->labels (oracle + GFM)")
  local elm_direct_obj, elm_direct_params, _, d1_train_h, d1_oof = optimize.elm({
    mode = cfg.elm_direct.mode,
    norm = cfg.elm_direct.norm,
    n_samples = train.n,
    n_tokens = n_bns_tokens,
    n_hidden = cfg.elm_direct.n_hidden,
    csc_offsets = train_bns_off,
    csc_indices = train_bns_idx,
    feature_weights = bns_scores,
    n_labels = n_labels,
    label_offsets = train_label_off,
    label_neighbors = train_label_nbr,
    expected_offsets = train_label_off,
    expected_neighbors = train_label_nbr,
    val_csc_offsets = dev_bns_off,
    val_csc_indices = dev_bns_idx,
    val_n_samples = dev.n,
    val_expected_offsets = dev_label_off,
    val_expected_neighbors = dev_label_nbr,
    lambda = cfg.elm_direct.lambda,
    propensity_a = cfg.elm_direct.propensity_a,
    propensity_b = cfg.elm_direct.propensity_b,
    k = cfg.elm_direct.k,
    search_trials = cfg.elm_direct.search_trials,
    each = util.make_elm_log(stopwatch),
    n_folds = cfg.elm_direct.n_folds, transform = true,
  })
  str.printf("  lambda=%.4e\n", elm_direct_params.lambda)
  local dv_oracle = eval_elm(elm_direct_obj, d1_train_h, cfg.elm_direct.k, nil, nil)

  print("\n  OOF + GFM calibration")
  local kp_k = cfg.elm_direct.k
  local d1_dv_off, d1_dv_nbr, d1_dv_scores = elm_direct_obj:label(dev_bns_off, dev_bns_idx, dev.n, kp_k)
  local d1_ts_off, d1_ts_nbr, d1_ts_scores = elm_direct_obj:label(test_bns_off, test_bns_idx, test_set.n, kp_k)
  local kp_dv_scores = elm_direct_obj:transform(dev_bns_off, dev_bns_idx, dev.n)
  local kp_ts_scores = elm_direct_obj:transform(test_bns_off, test_bns_idx, test_set.n)

  print("\n  GFM k-selection")
  local gfm_obj = gfm.create({
    pred_offsets = d1_oof.offsets, pred_neighbors = d1_oof.neighbors, pred_scores = d1_oof.scores,
    n_samples = train.n, n_labels = n_labels,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  local d1_train_mu = gfm_obj:calibrate_sums(d1_oof.transform, train.n)
  local d1_train_oracle_ks = eval.retrieval_ks({
    pred_offsets = d1_oof.offsets, pred_neighbors = d1_oof.neighbors,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  print_mu_by_k("#1", d1_train_mu, d1_train_oracle_ks, train.n)
  gfm_obj:fit_mu(d1_train_mu, d1_train_oracle_ks, train.n)
  local d1_gfm_p = select(2, optimize.gfm({
    gfm = gfm_obj,
    offsets = d1_dv_off, neighbors = d1_dv_nbr, scores = d1_dv_scores,
    n_samples = dev.n,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    beta = cfg.gfm.beta, gamma = cfg.gfm.gamma,
    search_trials = cfg.gfm.search_trials,
    each = util.make_gfm_log(stopwatch),
  }))
  str.printf("  beta=%.4f gamma=%.4f\n", d1_gfm_p.beta, d1_gfm_p.gamma)
  local d1_dv_mu = gfm_obj:calibrate_sums(kp_dv_scores, dev.n)
  local d1_ts_mu = gfm_obj:calibrate_sums(kp_ts_scores, test_set.n)
  local dv_ks = gfm_obj:predict({
    offsets = d1_dv_off, neighbors = d1_dv_nbr, scores = d1_dv_scores,
    n_samples = dev.n, beta = d1_gfm_p.beta, gamma = d1_gfm_p.gamma,
  })
  local ts_ks = gfm_obj:predict({
    offsets = d1_ts_off, neighbors = d1_ts_nbr, scores = d1_ts_scores,
    n_samples = test_set.n, beta = d1_gfm_p.beta, gamma = d1_gfm_p.gamma,
  })
  _, dv_gfm = eval.retrieval_ks({
    pred_offsets = d1_dv_off, pred_neighbors = d1_dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    ks = dv_ks,
  })
  _, ts_gfm = eval.retrieval_ks({
    pred_offsets = d1_ts_off, pred_neighbors = d1_ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    ks = ts_ks,
  })
  print_result("GFM", dv_gfm)

  local dv_oracle_ks = eval.retrieval_ks({
    pred_offsets = d1_dv_off, pred_neighbors = d1_dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  local d1_dv_atk = compute_at_ks(kp_dv_scores, dev.n, n_labels, dev_label_off, dev_label_nbr)
  local d1_ts_atk = compute_at_ks(kp_ts_scores, test_set.n, n_labels, test_label_off, test_label_nbr)
  d1_train_h = nil; d1_oof = nil -- luacheck: ignore
  d1_dv_off = nil; d1_dv_nbr = nil; d1_dv_scores = nil -- luacheck: ignore
  d1_ts_off = nil; d1_ts_nbr = nil; d1_ts_scores = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\nANN retrieval pipeline")

  print("\nLabel spectral embedding (Nystrom)")
  local label_bits = ivec.create():copy(train.solutions)
  local label_idf_ids, label_idf_scores = label_bits:bits_top_idf(train.n, n_labels)
  label_bits:bits_select(label_idf_ids, nil, n_labels)
  local label_inv = inv.create({ features = label_idf_scores })
  label_inv:add(label_bits, 0, train.n)
  local doc_embs, doc_ids, _, doc_eigs = spectral.encode({
    inv = label_inv, n_landmarks = cfg.emb.n_landmarks, n_dims = cfg.emb.n_dims,
  })
  label_inv = nil; label_bits = nil -- luacheck: ignore
  local doc_dims = doc_eigs:size()
  str.printf("  Embedded: %d  Dims: %d\n", doc_ids:size(), doc_dims)

  print("\nPhase B: Distill text -> label embedding")
  local distill_ids, distill_scores = train.tokens:bits_top_reg_auc(
    doc_embs, train.n, n_tokens_raw, doc_dims,
    cfg.data.toks_per_class, cfg.data.toks_overall, "max")
  local n_distill_tokens = distill_ids:size()
  str.printf("  Distill features: %d\n", n_distill_tokens)
  local function make_distill_csc(tokens, n)
    local bits = ivec.create():copy(tokens)
    bits:bits_select(distill_ids, nil, n_tokens_raw)
    return csr.to_csc(bits, n, n_distill_tokens)
  end
  local tr_d_off, tr_d_idx = make_distill_csc(train.tokens, train.n)
  local dv_d_off, dv_d_idx = make_distill_csc(dev.tokens, dev.n)
  local ts_d_off, ts_d_idx = make_distill_csc(test_set.tokens, test_set.n)
  train.tokens = nil; dev.tokens = nil; test_set.tokens = nil -- luacheck: ignore
  collectgarbage("collect")
  local distill_obj, distill_params, _, distill_train_h, distill_oof = optimize.elm({
    mode = cfg.emb.distill.mode,
    norm = cfg.emb.distill.norm,
    n_samples = train.n,
    n_tokens = n_distill_tokens,
    n_hidden = cfg.emb.distill.n_hidden,
    csc_offsets = tr_d_off, csc_indices = tr_d_idx,
    feature_weights = distill_scores,
    targets = doc_embs,
    n_targets = doc_dims,
    lambda = cfg.emb.distill.lambda,
    search_trials = cfg.emb.distill.search_trials,
    each = util.make_elm_log(stopwatch),
    n_folds = cfg.emb.n_folds,
  })
  str.printf("  Distill lambda=%.4e\n", distill_params.lambda)

  print("\nPhase C: OOF + ITQ + ANN")
  local itq = quantizer.create({
    mode = "itq", raw_codes = doc_embs, n_samples = train.n,
  })
  local train_bin = itq:encode(doc_embs)
  doc_embs = nil; distill_oof = nil -- luacheck: ignore
  local doc_ann = ann.create({ features = doc_dims })
  doc_ann:add(train_bin, doc_ids)
  train_bin = nil -- luacheck: ignore

  print("\nPhase D: Shortlisting (ANN -> neighbor docs -> label union)")
  local K = cfg.emb.k
  local function shortlist(d_off, d_idx, n)
    local pred = distill_obj:transform(d_off, d_idx, n)
    local bin = itq:encode(pred)
    local hood_ids, ann_hoods = doc_ann:neighborhoods_by_vecs(bin, K)
    local a_off, a_nbr = ann_hoods:to_csr(doc_dims)
    local union_bits = csr.label_union(
      a_off, a_nbr, hood_ids, train_label_off, train_label_nbr, n_labels)
    return union_bits:bits_to_csr(n, n_labels)
  end
  local dv_short_off, dv_short_nbr = shortlist(dv_d_off, dv_d_idx, dev.n)
  local ts_short_off, ts_short_nbr = shortlist(ts_d_off, ts_d_idx, test_set.n)
  collectgarbage("collect")

  local function csr_recall(short_off, short_nbr, gt_off, gt_nbr, n)
    local found, total, rsum, nv = 0, 0, 0, 0
    for i = 0, n - 1 do
      local ss, se = short_off:get(i), short_off:get(i + 1)
      local gs, ge = gt_off:get(i), gt_off:get(i + 1)
      local nt = ge - gs
      local nf = 0
      local si, gi = ss, gs
      while si < se and gi < ge do
        local sv, gv = short_nbr:get(si), gt_nbr:get(gi)
        if sv == gv then nf = nf + 1; si = si + 1; gi = gi + 1
        elseif sv < gv then si = si + 1
        else gi = gi + 1 end
      end
      found = found + nf
      total = total + nt
      if nt > 0 then rsum = rsum + nf / nt; nv = nv + 1 end
    end
    return found / total, rsum / nv
  end
  do
    local mir, mar = csr_recall(dv_short_off, dv_short_nbr, dev_label_off, dev_label_nbr, dev.n)
    str.printf("  Dev retrieval recall: miR=%.4f maR=%.4f avg_short=%.1f\n",
      mir, mar, (dv_short_off:get(dev.n) - dv_short_off:get(0)) / dev.n)
  end
  do
    local mir, mar = csr_recall(ts_short_off, ts_short_nbr, test_label_off, test_label_nbr, test_set.n)
    str.printf("  Test retrieval recall: miR=%.4f maR=%.4f avg_short=%.1f\n",
      mir, mar, (ts_short_off:get(test_set.n) - ts_short_off:get(0)) / test_set.n)
  end

  print("\n#2: ANN shortlist + ELM scores + GFM")
  local dv_short_scores = eval.gather_label_scores({
    scores = kp_dv_scores, pred_offsets = dv_short_off,
    pred_neighbors = dv_short_nbr, n_labels = n_labels,
  })
  local dv_sorted_nbr, dv_sorted_sc = csr.sort_csr_desc(dv_short_off, dv_short_nbr, dv_short_scores)
  local ts_short_scores = eval.gather_label_scores({
    scores = kp_ts_scores, pred_offsets = ts_short_off,
    pred_neighbors = ts_short_nbr, n_labels = n_labels,
  })
  local ts_sorted_nbr, ts_sorted_sc = csr.sort_csr_desc(ts_short_off, ts_short_nbr, ts_short_scores)

  _, dv_ann_orc = eval.retrieval_ks({
    pred_offsets = dv_short_off, pred_neighbors = dv_sorted_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  print_result("Orc D", dv_ann_orc)
  local dv_ks4 = gfm_obj:predict({
    offsets = dv_short_off, neighbors = dv_sorted_nbr, scores = dv_sorted_sc,
    n_samples = dev.n, beta = d1_gfm_p.beta, gamma = d1_gfm_p.gamma,
  })
  _, dv_ann_gfm = eval.retrieval_ks({
    pred_offsets = dv_short_off, pred_neighbors = dv_sorted_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr, ks = dv_ks4,
  })
  print_result("GFM D", dv_ann_gfm)
  local ts_ks4 = gfm_obj:predict({
    offsets = ts_short_off, neighbors = ts_sorted_nbr, scores = ts_sorted_sc,
    n_samples = test_set.n, beta = d1_gfm_p.beta, gamma = d1_gfm_p.gamma,
  })
  _, ts_ann_gfm = eval.retrieval_ks({
    pred_offsets = ts_short_off, pred_neighbors = ts_sorted_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr, ks = ts_ks4,
  })
  print_result("GFM T", ts_ann_gfm)
  elm_direct_obj = nil; gfm_obj = nil -- luacheck: ignore
  kp_dv_scores = nil; kp_ts_scores = nil -- luacheck: ignore
  distill_obj = nil; distill_train_h = nil -- luacheck: ignore

  str.printf("\n  Time: %.1fs\n", stopwatch())

  print("\n========== Summary ==========")
  print("#1 ELM text->labels (oracle + GFM)")
  print_result("  Dev Orc", dv_oracle)
  print_result("  Dev GFM", dv_gfm)
  print_result("  Tst GFM", ts_gfm)
  print_k_err("GFM", dv_ks, dv_oracle_ks, dev.n)
  print_k_buckets("GFM", dv_ks, dv_oracle_ks, dev.n)
  print("  Dev @k:")
  print_at_ks_table(d1_dv_atk)
  print("  Tst @k:")
  print_at_ks_table(d1_ts_atk)
  print("#2 ANN shortlist + ELM scores + GFM")
  print_result("  Dev Orc", dv_ann_orc)
  print_result("  Dev GFM", dv_ann_gfm)
  print_result("  Tst GFM", ts_ann_gfm)

end)
