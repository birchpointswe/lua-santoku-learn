local ann = require("santoku.learn.ann")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local gfm = require("santoku.learn.gfm")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local quantizer = require("santoku.learn.quantizer")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  kernel = "cosine",
  data = { max = nil, toks_per_class = nil, toks_overall = nil },
  tokenizer = {
    max_len = 20, min_len = 1, max_run = 2,
    ngrams = 2, cgrams_min = 3, cgrams_max = 5,
    cgrams_cross = true, skips = 1,
  },
  elm_direct = {
    n_hidden = 1024, seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 80, k = 32, n_folds = 5,
  },
  elm_stacked = {
    n_hidden = 1024,
    seed = 42,
    lambda = { def = 1.0 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    search_trials = 80, k = 32, n_folds = 5,
  },
  emb = {
    k = 256, n_folds = 5,
    n_landmarks = 4096, n_dims = 64, decay = 0,
    distill = {
      n_hidden = 1024, seed = 42,
      lambda = { def = 1.0 }, search_trials = 80,
    },
  },
}

test("eurlex-aligned", function ()

  local stopwatch = utc.stopwatch()

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
  local bns_ids, bns_scores = train.tokens:bits_top_chi2(
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

  local function eval_elm(obj, train_h, k, dense_dv, dense_ts)
    local dv_off, dv_nbr = obj:label(dev_bns_off, dev_bns_idx, dev.n, k, dense_dv)
    local dv_oracle = eval_oracle(dv_off, dv_nbr, dev_label_off, dev_label_nbr)
    print_result("Oracle", dv_oracle)
    return dv_oracle
  end

  print("\n#1: Direct ELM text->labels (baseline)")
  local elm_direct_obj, elm_direct_params, _, d1_train_h = optimize.elm({
    n_samples = train.n,
    n_tokens = n_bns_tokens,
    n_hidden = cfg.elm_direct.n_hidden,
    seed = cfg.elm_direct.seed,
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
  })
  str.printf("  lambda=%.4e\n", elm_direct_params.lambda)
  local dv_oracle = eval_elm(elm_direct_obj, d1_train_h, cfg.elm_direct.k, nil, nil)

  print("\n#2: GFM (OOF scores->k)")
  local kp_k = cfg.elm_direct.k
  local elm_dims = elm_direct_obj.ridge:n_dims()
  local oof_off, oof_nbr, oof_scores, oof_transform = optimize.elm_oof({
    n_samples = train.n, train_h = d1_train_h, dims = elm_dims,
    n_labels = n_labels,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    lambda = elm_direct_params.lambda,
    propensity_a = elm_direct_params.propensity_a,
    propensity_b = elm_direct_params.propensity_b,
    k = kp_k, n_folds = cfg.elm_direct.n_folds, transform = true,
  })
  local d1_dv_off, d1_dv_nbr, d1_dv_scores = elm_direct_obj:label(dev_bns_off, dev_bns_idx, dev.n, kp_k)
  local d1_ts_off, d1_ts_nbr, d1_ts_scores = elm_direct_obj:label(test_bns_off, test_bns_idx, test_set.n, kp_k)
  local gfm_obj = gfm.create({
    pred_offsets = oof_off, pred_neighbors = oof_nbr, pred_scores = oof_scores,
    n_samples = train.n, n_labels = n_labels,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  local kp_dv_scores = elm_direct_obj:transform(dev_bns_off, dev_bns_idx, dev.n)
  local kp_ts_scores = elm_direct_obj:transform(test_bns_off, test_bns_idx, test_set.n)
  local dv_ks = gfm_obj:predict({
    offsets = d1_dv_off, neighbors = d1_dv_nbr, scores = d1_dv_scores,
    n_samples = dev.n,
  })
  local ts_ks = gfm_obj:predict({
    offsets = d1_ts_off, neighbors = d1_ts_nbr, scores = d1_ts_scores,
    n_samples = test_set.n,
  })
  local _, dv_gfm = eval.retrieval_ks({
    pred_offsets = d1_dv_off, pred_neighbors = d1_dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    ks = dv_ks,
  })
  local _, ts_gfm = eval.retrieval_ks({
    pred_offsets = d1_ts_off, pred_neighbors = d1_ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    ks = ts_ks,
  })
  print_result("GFM", dv_gfm)

  print("\n#3: Stacked ELM (OOF scores->labels) + GFM")
  local elm2_obj, elm2_params, _, elm2_train_h = optimize.elm({
    n_samples = train.n,
    dense_features = oof_transform,
    n_dense = n_labels,
    n_hidden = cfg.elm_stacked.n_hidden,
    seed = cfg.elm_stacked.seed,
    n_labels = n_labels,
    label_offsets = train_label_off,
    label_neighbors = train_label_nbr,
    expected_offsets = train_label_off,
    expected_neighbors = train_label_nbr,
    val_dense_features = kp_dv_scores,
    val_n_samples = dev.n,
    val_expected_offsets = dev_label_off,
    val_expected_neighbors = dev_label_nbr,
    lambda = cfg.elm_stacked.lambda,
    propensity_a = cfg.elm_stacked.propensity_a,
    propensity_b = cfg.elm_stacked.propensity_b,
    search_trials = cfg.elm_stacked.search_trials,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  lambda=%.4e\n", elm2_params.lambda)
  local kp_k2 = cfg.elm_stacked.k
  local elm2_dims = elm2_obj.ridge:n_dims()
  local e2_oof_off, e2_oof_nbr, e2_oof_scores = optimize.elm_oof({
    n_samples = train.n, train_h = elm2_train_h, dims = elm2_dims,
    n_labels = n_labels,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    lambda = elm2_params.lambda,
    propensity_a = elm2_params.propensity_a,
    propensity_b = elm2_params.propensity_b,
    k = kp_k2, n_folds = cfg.elm_stacked.n_folds,
  })
  local e2_dv_off, e2_dv_nbr, e2_dv_scores = elm2_obj:label(nil, nil, dev.n, kp_k2, kp_dv_scores)
  local e2_ts_off, e2_ts_nbr, e2_ts_scores = elm2_obj:label(nil, nil, test_set.n, kp_k2, kp_ts_scores)
  local e2_dv_oracle = eval_oracle(e2_dv_off, e2_dv_nbr, dev_label_off, dev_label_nbr)
  local e2_ts_oracle = eval_oracle(e2_ts_off, e2_ts_nbr, test_label_off, test_label_nbr)
  print_result("Stk Orc", e2_dv_oracle)
  local gfm2_obj = gfm.create({
    pred_offsets = e2_oof_off, pred_neighbors = e2_oof_nbr, pred_scores = e2_oof_scores,
    n_samples = train.n, n_labels = n_labels,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
  })
  local stk_dv_scores = elm2_obj:transform(nil, nil, dev.n, kp_dv_scores)
  local stk_ts_scores = elm2_obj:transform(nil, nil, test_set.n, kp_ts_scores)
  local e2_dv_ks = gfm2_obj:predict({
    offsets = e2_dv_off, neighbors = e2_dv_nbr, scores = e2_dv_scores,
    n_samples = dev.n,
  })
  local e2_ts_ks = gfm2_obj:predict({
    offsets = e2_ts_off, neighbors = e2_ts_nbr, scores = e2_ts_scores,
    n_samples = test_set.n,
  })
  local _, e2_dv_gfm = eval.retrieval_ks({
    pred_offsets = e2_dv_off, pred_neighbors = e2_dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr, ks = e2_dv_ks,
  })
  local _, e2_ts_gfm = eval.retrieval_ks({
    pred_offsets = e2_ts_off, pred_neighbors = e2_ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr, ks = e2_ts_ks,
  })
  print_result("Stk GFM", e2_dv_gfm)

  print("\n  --- Diagnostics (dev) ---")
  local dv_oracle_ks = eval.retrieval_ks({
    pred_offsets = d1_dv_off, pred_neighbors = d1_dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  d1_dv_atk = compute_at_ks(kp_dv_scores, dev.n, n_labels, dev_label_off, dev_label_nbr)
  d1_ts_atk = compute_at_ks(kp_ts_scores, test_set.n, n_labels, test_label_off, test_label_nbr)
  print("\n  #1 scores@k (dev):")
  print_at_ks_table(d1_dv_atk)
  print_k_err("GFM", dv_ks, dv_oracle_ks, dev.n)
  print_k_buckets("GFM", dv_ks, dv_oracle_ks, dev.n)
  e2_dv_atk = compute_at_ks(stk_dv_scores, dev.n, n_labels, dev_label_off, dev_label_nbr)
  e2_ts_atk = compute_at_ks(stk_ts_scores, test_set.n, n_labels, test_label_off, test_label_nbr)
  print("\n  #3 scores@k (dev):")
  print_at_ks_table(e2_dv_atk)
  print_k_err("Stk GFM", e2_dv_ks, dv_oracle_ks, dev.n)
  print_k_buckets("Stk GFM", e2_dv_ks, dv_oracle_ks, dev.n)
  print("  --- End diagnostics ---")

  d1_train_h = nil; oof_transform = nil -- luacheck: ignore
  elm2_obj = nil; elm2_train_h = nil; gfm2_obj = nil -- luacheck: ignore
  oof_off = nil; oof_nbr = nil; oof_scores = nil -- luacheck: ignore
  d1_dv_off = nil; d1_dv_nbr = nil; d1_dv_scores = nil -- luacheck: ignore
  d1_ts_off = nil; d1_ts_nbr = nil; d1_ts_scores = nil -- luacheck: ignore
  stk_dv_scores = nil; stk_ts_scores = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\n#4: Doc-embedding retrieval pipeline")

  print("\nPhase A: Doc spectral embedding")
  local doc_idf_ids, doc_idf_scores = train.solutions:bits_top_idf(train.n, n_labels)
  local doc_bits = ivec.create():copy(train.solutions)
  doc_bits:bits_select(doc_idf_ids, nil, n_labels)
  local n_doc_feats = doc_idf_ids:size()
  str.printf("  IDF label-features: %d\n", n_doc_feats)
  local doc_ids = ivec.create(train.n):fill_indices()
  local doc_index = inv.create({ features = doc_idf_scores, kernel = cfg.kernel })
  doc_index:add(doc_bits, doc_ids)
  doc_bits = nil -- luacheck: ignore
  local doc_model = optimize.spectral({
    index = doc_index,
    n_landmarks = cfg.emb.n_landmarks,
    n_dims = cfg.emb.n_dims,
    decay = cfg.emb.decay,
    each = util.make_spectral_log(stopwatch),
  })
  local doc_dims = doc_model.dims
  str.printf("  Doc spectral: %d dims, %d embedded\n", doc_dims, doc_model.ids:size())
  local all_doc_ids = ivec.create(train.n):fill_indices()
  local doc_embs = dvec.create():mtx_extend(
    doc_model.raw_codes, all_doc_ids, doc_model.ids, 0, doc_dims, true)
  doc_index = nil; doc_model = nil -- luacheck: ignore

  print("\nPhase B: Distill text -> doc embedding")
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
  local distill_obj, distill_params, _, distill_train_h = optimize.elm({
    n_samples = train.n,
    n_tokens = n_distill_tokens,
    n_hidden = cfg.emb.distill.n_hidden,
    seed = cfg.emb.distill.seed,
    csc_offsets = tr_d_off, csc_indices = tr_d_idx,
    feature_weights = distill_scores,
    targets = doc_embs,
    n_targets = doc_dims,
    lambda = cfg.emb.distill.lambda,
    search_trials = cfg.emb.distill.search_trials,
    each = util.make_elm_log(stopwatch),
  })
  str.printf("  Distill lambda=%.4e\n", distill_params.lambda)

  print("\nPhase C: OOF + ITQ + ANN")
  local oof_doc_embs = optimize.elm_oof_dense({
    n_samples = train.n,
    train_h = distill_train_h,
    dims = cfg.emb.distill.n_hidden,
    targets = doc_embs,
    n_targets = doc_dims,
    lambda = distill_params.lambda,
    n_folds = cfg.emb.n_folds,
  })
  doc_embs = nil -- luacheck: ignore
  local itq = quantizer.create({
    mode = "itq", raw_codes = oof_doc_embs, n_samples = train.n,
  })
  local train_bin = itq:encode(oof_doc_embs)
  oof_doc_embs = nil -- luacheck: ignore
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
  distill_obj = nil; distill_train_h = nil -- luacheck: ignore
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

  print("\nPhase E: OVA re-ranking + GFM")
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

  local _, dv_emb_orc = eval.retrieval_ks({
    pred_offsets = dv_short_off, pred_neighbors = dv_sorted_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  print_result("Orc D", dv_emb_orc)
  local dv_ks4 = gfm_obj:predict({
    offsets = dv_short_off, neighbors = dv_sorted_nbr, scores = dv_sorted_sc,
    n_samples = dev.n,
  })
  local _, dv_emb_gfm = eval.retrieval_ks({
    pred_offsets = dv_short_off, pred_neighbors = dv_sorted_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr, ks = dv_ks4,
  })
  print_result("GFM D", dv_emb_gfm)
  local ts_ks4 = gfm_obj:predict({
    offsets = ts_short_off, neighbors = ts_sorted_nbr, scores = ts_sorted_sc,
    n_samples = test_set.n,
  })
  local _, ts_emb_gfm = eval.retrieval_ks({
    pred_offsets = ts_short_off, pred_neighbors = ts_sorted_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr, ks = ts_ks4,
  })
  print_result("GFM T", ts_emb_gfm)
  elm_direct_obj = nil; gfm_obj = nil -- luacheck: ignore

  str.printf("\n  Time: %.1fs\n", stopwatch())

  print("\n========== Summary ==========")
  print("#1 Direct ELM: text->labels, oracle-k")
  print_result("  Dev", dv_oracle)
  print("#2 Direct ELM + GFM: OOF calibration")
  print_result("  Dev GFM", dv_gfm)
  print_result("  Tst GFM", ts_gfm)
  print_k_err("GFM", dv_ks, dv_oracle_ks, dev.n)
  print_k_buckets("GFM", dv_ks, dv_oracle_ks, dev.n)
  print("  Dev @k:")
  print_at_ks_table(d1_dv_atk)
  print("  Tst @k:")
  print_at_ks_table(d1_ts_atk)
  print("#3 Stacked ELM + GFM: OOF scores->labels")
  print_result("  Dev Orc", e2_dv_oracle)
  print_result("  Tst Orc", e2_ts_oracle)
  print_result("  Dev GFM", e2_dv_gfm)
  print_result("  Tst GFM", e2_ts_gfm)
  print_k_err("Stk GFM", e2_dv_ks, dv_oracle_ks, dev.n)
  print_k_buckets("Stk GFM", e2_dv_ks, dv_oracle_ks, dev.n)
  print("  Dev @k:")
  print_at_ks_table(e2_dv_atk)
  print("  Tst @k:")
  print_at_ks_table(e2_ts_atk)
  print("#4 Doc-embedding retrieval + OVA re-ranking + GFM")
  print_result("  Dev Orc", dv_emb_orc)
  print_result("  Dev GFM", dv_emb_gfm)
  print_result("  Tst GFM", ts_emb_gfm)

end)
