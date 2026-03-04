local ann = require("santoku.learn.ann")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local elm = require("santoku.learn.elm")
local eval = require("santoku.learn.evaluator")
local hdc = require("santoku.learn.hdc")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local quantizer = require("santoku.learn.quantizer")
local ridge = require("santoku.learn.ridge")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { max = nil },
  hdc = { d = 8192, ngram = 5 },
  elm = {
    lambda = { def = 3.8814e-05 },
    propensity_a = { def = 0.41 },
    propensity_b = { def = 0.18 },
    search_trials = 200,
    n_folds = 10,
    k = 256,
  },
  emb = {
    k = 256,
    n_folds = 10,
    n_landmarks = 4096,
    n_dims = 256,
    distill = {
      lambda = { def = 1.1741e-04 },
      search_trials = 200,
    },
  },
  kpred = {
    mode = "rff",
    n_hidden = 8192,
    lambda = { def = 2.5792e-06 },
    propensity_a = { def = 3.11 },
    propensity_b = { def = 1.56 },
    search_trials = 200,
  },
}

test("eurlex-hvelm", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local kp_k = cfg.elm.k
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

  local function print_result(label, m)
    str.printf("  %-10s miP=%.4f miR=%.4f miF1=%.4f maP=%.4f maR=%.4f maF1=%.4f\n",
      label, m.micro_precision, m.micro_recall, m.micro_f1,
      m.macro_precision, m.macro_recall, m.macro_f1)
  end

  local function make_kpred_ctx(oof_off, oof_nbr, dv_off, dv_nbr, ts_off, ts_nbr)
    local tr_ks = eval.retrieval_ks({
      pred_offsets = oof_off, pred_neighbors = oof_nbr,
      expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    })
    local dv_ks = eval.retrieval_ks({
      pred_offsets = dv_off, pred_neighbors = dv_nbr,
      expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    })
    local n_classes = math.max((tr_ks:max()), (dv_ks:max()))
    str.printf("  n_classes (max k): %d\n", n_classes)
    return {
      dv_pred_off = dv_off, dv_pred_nbr = dv_nbr,
      ts_pred_off = ts_off, ts_pred_nbr = ts_nbr,
      n_classes = n_classes,
      tr_off = ivec.create(train.n + 1):fill_indices(),
      tr_nbr = ivec.create():copy(tr_ks):add(-1),
      dv_koff = ivec.create(dev.n + 1):fill_indices(),
      dv_knbr = ivec.create():copy(dv_ks):add(-1),
    }
  end

  local function run_kpred(label, tr_sco, dv_sco, ts_sco, ctx)
    local enc, th = elm.create({
      codes = tr_sco, n_input_dims = kp_k,
      n_samples = train.n, n_hidden = cfg.kpred.n_hidden, mode = cfg.kpred.mode,
    })
    local dh = enc:encode({ codes = dv_sco, n_samples = dev.n })
    local kp_off_buf, kp_nbr_buf, kp_sco_buf
    local function kp_score_fn(_, r)
      kp_off_buf, kp_nbr_buf, kp_sco_buf =
        r:label(dh, dev.n, 1, kp_off_buf, kp_nbr_buf, kp_sco_buf)
      kp_nbr_buf:add(1)
      local _, m = eval.retrieval_ks({
        pred_offsets = ctx.dv_pred_off, pred_neighbors = ctx.dv_pred_nbr,
        expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr, ks = kp_nbr_buf,
      })
      kp_nbr_buf:add(-1)
      return m.macro_f1
    end
    local r, p = optimize.ridge({
      n_samples = train.n, n_dims = cfg.kpred.n_hidden, codes = th,
      n_labels = ctx.n_classes,
      label_offsets = ctx.tr_off, label_neighbors = ctx.tr_nbr,
      expected_offsets = ctx.tr_off, expected_neighbors = ctx.tr_nbr,
      val_codes = dh, val_n_samples = dev.n,
      val_expected_offsets = ctx.dv_koff, val_expected_neighbors = ctx.dv_knbr,
      lambda = cfg.kpred.lambda, propensity_a = cfg.kpred.propensity_a,
      propensity_b = cfg.kpred.propensity_b,
      k = 1, search_trials = cfg.kpred.search_trials,
      score_fn = kp_score_fn,
      each = util.make_ridge_log(stopwatch),
    })
    str.printf("  %s: lambda=%.4e propA=%.4f propB=%.4f\n",
      label, p.lambda, p.propensity_a, p.propensity_b)
    local _, dn = r:label(dh, dev.n, 1)
    local dk = ivec.create():copy(dn):add(1)
    local tsh = enc:encode({ codes = ts_sco, n_samples = test_set.n })
    local _, tn = r:label(tsh, test_set.n, 1)
    local tk = ivec.create():copy(tn):add(1)
    local _, dm = eval.retrieval_ks({
      pred_offsets = ctx.dv_pred_off, pred_neighbors = ctx.dv_pred_nbr,
      expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr, ks = dk,
    })
    local _, tm = eval.retrieval_ks({
      pred_offsets = ctx.ts_pred_off, pred_neighbors = ctx.ts_pred_nbr,
      expected_offsets = test_label_off, expected_neighbors = test_label_nbr, ks = tk,
    })
    print_result(label .. " D", dm)
    print_result(label .. " T", tm)
    return dm, tm
  end

  local function print_pipeline_summary(label, oracle, dm, tm)
    print("\n========== " .. label .. " ==========")
    print_result("  Dev Orc", oracle)
    print_result("  D", dm)
    print_result("  T", tm)
    str.printf("  Time: %.1fs\n", stopwatch())
  end

  -- =============================================
  -- Pipeline 1: text -> labels
  -- =============================================
  print("\n#1: HDC text->labels")
  local ngram_map, train_tok, n_hdc_tokens = hdc.tokenize({
    texts = train.problems, hdc_ngram = cfg.hdc.ngram, n_samples = train.n,
  })
  str.printf("  HDC tokens: %d\n", n_hdc_tokens)
  local bns_ids, bns_scores = train_tok:bits_top_bns(
    train.solutions, train.n, n_hdc_tokens, n_labels)
  str.printf("  BNS selected: %d\n", bns_ids:size())
  local hdc_enc, train_h = hdc.create({
    texts = train.problems, n_samples = train.n,
    d = cfg.hdc.d, hdc_ngram = cfg.hdc.ngram,
    weight_map = ngram_map, weight_ids = bns_ids, weights = bns_scores,
  })
  local dev_h = hdc_enc:encode({ texts = dev.problems, n_samples = dev.n })
  local test_h = hdc_enc:encode({ texts = test_set.problems, n_samples = test_set.n })
  local hdc_out_d = hdc_enc:out_d()

  local ridge_obj, elm_params = optimize.ridge({
    n_samples = train.n, n_dims = hdc_out_d, codes = train_h,
    n_labels = n_labels,
    label_offsets = train_label_off, label_neighbors = train_label_nbr,
    expected_offsets = train_label_off, expected_neighbors = train_label_nbr,
    val_codes = dev_h, val_n_samples = dev.n,
    val_expected_offsets = dev_label_off, val_expected_neighbors = dev_label_nbr,
    lambda = cfg.elm.lambda, propensity_a = cfg.elm.propensity_a, propensity_b = cfg.elm.propensity_b,
    k = kp_k, search_trials = cfg.elm.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("  lambda=%.4e propA=%.4f propB=%.4f\n",
    elm_params.lambda, elm_params.propensity_a, elm_params.propensity_b)

  print("\n  OOF predictions")
  local oof_off, oof_nbr, oof_sco, oof_transform = ridge.solve_oof({
    n_samples = train.n, n_dims = hdc_out_d, codes = train_h,
    lambda = elm_params.lambda, n_folds = cfg.elm.n_folds,
    n_labels = n_labels, label_offsets = train_label_off, label_neighbors = train_label_nbr,
    propensity_a = elm_params.propensity_a, propensity_b = elm_params.propensity_b,
    k = kp_k, transform = true,
  })
  local d1_dv_off, d1_dv_nbr, d1_dv_sco = ridge_obj:label(dev_h, dev.n, kp_k)
  local d1_ts_off, d1_ts_nbr, d1_ts_sco = ridge_obj:label(test_h, test_set.n, kp_k)

  print("\n  kpred1")
  local d1_oracle = eval_oracle(d1_dv_off, d1_dv_nbr, dev_label_off, dev_label_nbr)
  print_result("Oracle", d1_oracle)
  local kp1_ctx = make_kpred_ctx(
    oof_off, oof_nbr, d1_dv_off, d1_dv_nbr, d1_ts_off, d1_ts_nbr)
  local kp1_dm, kp1_tm = run_kpred("kpred1", oof_sco, d1_dv_sco, d1_ts_sco, kp1_ctx)

  print_pipeline_summary("#1", d1_oracle, kp1_dm, kp1_tm)

  oof_off = nil; oof_nbr = nil; oof_sco = nil -- luacheck: ignore
  d1_dv_off = nil; d1_dv_nbr = nil; d1_dv_sco = nil -- luacheck: ignore
  d1_ts_off = nil; d1_ts_nbr = nil; d1_ts_sco = nil -- luacheck: ignore
  collectgarbage("collect")

  print("\nScore spectral embedding (Nystrom)")
  local doc_embs, doc_ids, _, doc_eigs = spectral.encode({
    codes = oof_transform, n_samples = train.n,
    n_landmarks = cfg.emb.n_landmarks, n_dims = cfg.emb.n_dims,
  })
  local doc_dims = doc_eigs:size()
  str.printf("  Embedded: %d  Dims: %d\n", doc_ids:size(), doc_dims)

  print("\nDistillation: text -> score embedding")
  local auc_ids, auc_scores = train_tok:bits_top_reg_auc(
    doc_embs, train.n, n_hdc_tokens, doc_dims)
  str.printf("  AUC selected: %d\n", auc_ids:size())
  hdc_enc:reweight(ngram_map, auc_ids, auc_scores)
  local distill_train_h = hdc_enc:encode({
    texts = train.problems, n_samples = train.n, out = train_h,
  })
  train_h = nil -- luacheck: ignore
  local distill_out_d = hdc_enc:out_d()
  local distill_ridge, distill_params = optimize.ridge({
    n_samples = train.n, n_dims = distill_out_d, codes = distill_train_h,
    targets = doc_embs, n_targets = doc_dims,
    lambda = cfg.emb.distill.lambda,
    search_trials = cfg.emb.distill.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("  Distill lambda=%.4e\n", distill_params.lambda)

  print("\n  Distill OOF (train embeddings)")
  local train_pred = ridge.solve_oof({
    n_samples = train.n, n_dims = distill_out_d, codes = distill_train_h,
    targets = doc_embs, n_targets = doc_dims,
    lambda = distill_params.lambda, n_folds = cfg.emb.n_folds,
  })
  doc_embs = nil; distill_train_h = nil -- luacheck: ignore

  local dev_distill_h = hdc_enc:encode({ texts = dev.problems, n_samples = dev.n })
  local test_distill_h = hdc_enc:encode({ texts = test_set.problems, n_samples = test_set.n })
  hdc_enc = nil -- luacheck: ignore
  local dev_embs = distill_ridge:transform(dev_distill_h, dev.n)
  dev_distill_h = nil -- luacheck: ignore
  local test_embs = distill_ridge:transform(test_distill_h, test_set.n)
  test_distill_h = nil -- luacheck: ignore

  print("\n#2: ANN shortlist + OVA scoring")

  print("\n  ITQ + ANN")
  local post_itq = quantizer.create({
    mode = "itq", raw_codes = train_pred, n_samples = train.n,
  })
  local post_n_bits = post_itq:n_bits()
  str.printf("  ITQ bits: %d\n", post_n_bits)
  local post_train_bin = post_itq:encode(train_pred)
  local doc_ann = ann.create({ features = post_n_bits })
  doc_ann:add(post_train_bin, doc_ids)
  post_train_bin = nil -- luacheck: ignore

  print("\n  Shortlisting")
  local K = cfg.emb.k
  local function shortlist(embs, n)
    local bin = post_itq:encode(embs)
    local hood_ids, ann_hoods = doc_ann:neighborhoods_by_vecs(bin, K)
    local a_off, a_nbr = ann_hoods:to_csr(post_n_bits)
    local union_bits = csr.label_union(
      a_off, a_nbr, hood_ids, train_label_off, train_label_nbr, n_labels)
    local sl_off, sl_nbr = union_bits:bits_to_csr(n, n_labels)
    return sl_off, sl_nbr
  end
  local dv_short_off, dv_short_nbr = shortlist(dev_embs, dev.n)
  local ts_short_off, ts_short_nbr = shortlist(test_embs, test_set.n)
  local tr_short_off, tr_short_nbr = shortlist(train_pred, train.n)
  train_pred = nil -- luacheck: ignore
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

  print("\n  Scoring shortlists via OVA ridge")
  local dv_short_scores = eval.gather_label_scores({
    scores = ridge_obj:transform(dev_h, dev.n),
    pred_offsets = dv_short_off, pred_neighbors = dv_short_nbr, n_labels = n_labels,
  })
  dev_h = nil -- luacheck: ignore
  local dv_sorted_nbr, dv_sorted_sc = csr.sort_csr_desc(dv_short_off, dv_short_nbr, dv_short_scores)
  local ts_short_scores = eval.gather_label_scores({
    scores = ridge_obj:transform(test_h, test_set.n),
    pred_offsets = ts_short_off, pred_neighbors = ts_short_nbr, n_labels = n_labels,
  })
  test_h = nil; ridge_obj = nil -- luacheck: ignore
  local ts_sorted_nbr, ts_sorted_sc = csr.sort_csr_desc(ts_short_off, ts_short_nbr, ts_short_scores)
  local tr_short_scores = eval.gather_label_scores({
    scores = oof_transform, pred_offsets = tr_short_off,
    pred_neighbors = tr_short_nbr, n_labels = n_labels,
  })
  oof_transform = nil -- luacheck: ignore
  collectgarbage("collect")
  local tr_sorted_nbr, tr_sorted_sc = csr.sort_csr_desc(tr_short_off, tr_short_nbr, tr_short_scores)

  print("\n  kpred2")
  local ann_oracle = eval_oracle(dv_short_off, dv_sorted_nbr, dev_label_off, dev_label_nbr)
  print_result("Ann Orc", ann_oracle)
  local tr_ann_sco = eval.csr_topk_dense({
    offsets = tr_short_off, values = tr_sorted_sc, n_samples = train.n, k = kp_k,
  })
  local dv_ann_sco = eval.csr_topk_dense({
    offsets = dv_short_off, values = dv_sorted_sc, n_samples = dev.n, k = kp_k,
  })
  local ts_ann_sco = eval.csr_topk_dense({
    offsets = ts_short_off, values = ts_sorted_sc, n_samples = test_set.n, k = kp_k,
  })
  local kp2_ctx = make_kpred_ctx(
    tr_short_off, tr_sorted_nbr, dv_short_off, dv_sorted_nbr, ts_short_off, ts_sorted_nbr)
  local kp2_dm, kp2_tm = run_kpred("kpred2", tr_ann_sco, dv_ann_sco, ts_ann_sco, kp2_ctx)
  print_pipeline_summary("#2", ann_oracle, kp2_dm, kp2_tm)

  post_itq = nil; distill_ridge = nil -- luacheck: ignore

  str.printf("\n  Total time: %.1fs\n", stopwatch())

end)
