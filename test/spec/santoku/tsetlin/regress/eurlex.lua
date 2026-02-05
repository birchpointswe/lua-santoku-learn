local ann = require("santoku.tsetlin.ann")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local cvec = require("santoku.cvec")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local hlth = require("santoku.tsetlin.hlth")
local inv = require("santoku.tsetlin.inv")
local ivec = require("santoku.ivec")
local pvec = require("santoku.pvec")
local rvec = require("santoku.rvec")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.tsetlin.util")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max = nil,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 0,
    cgrams_max = 0,
    cgrams_cross = false,
    skips = 1,
  },
  feature_selection = {
    min_df = 2,
    max_df = 0.5,
    bns_tokens = 65536,
    per_class = 8192,
  },
  regressor = {
    clauses = 256,
    clause_tolerance = { def = 869, min = 8, max = 1024, int = true },
    clause_maximum = { def = 765, min = 8, max = 1024, int = true },
    target = { def = 119, min = 8, max = 1024, int = true },
    specificity = { def = 1182, min = 2, max = 2000 },
    search_rounds = 0,
    search_trials = 20,
    search_iterations = 40,
    final_patience = 20,
    final_batch = 40,
    final_iterations = 400,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 256,
    decay = 0.1,
    bandwidth = -1,
    rounds = 0,
    samples = 20,
  },
  rp = {
    max_bits = 256,
    tolerance = 0.001
  },
  eval = {
    knn = 16,
    ranking = "ndcg",
    retrieval_k = 32,
    retrieval_radius = 5,
  },
}

test("eurlex-docs", function()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local train_lc = train.label_counts:to_dvec()
  train_lc:asc()
  str.printf("  Train: %d  Dev: %d  Test: %d  Labels: %d  Labels/doc: %.1f median\n",
    train.n, dev.n, test_set.n, n_labels, train_lc:get(train_lc:size() / 2))

  print("\nCreating IDs")
  train.ids = ivec.create(train.n):fill_indices()
  dev.ids = ivec.create(dev.n):fill_indices():add(train.n)
  test_set.ids = ivec.create(test_set.n):fill_indices():add(train.n + dev.n)

  print("\nBuilding label CSR for lookups")
  local train_label_offsets, train_label_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }
  local dev_label_offsets, dev_label_neighbors = dev.solutions:bits_to_csr(dev.n, n_labels)
  dev.label_csr = { offsets = dev_label_offsets, neighbors = dev_label_neighbors }
  local test_label_offsets, test_label_neighbors = test_set.solutions:bits_to_csr(test_set.n, n_labels)
  test_set.label_csr = { offsets = test_label_offsets, neighbors = test_label_neighbors }

  print("\nBuilding bipartite graph")
  local n_total_nodes = train.n + n_labels
  local all_ids = ivec.create(n_total_nodes):fill_indices()
  local label_node_ids = ivec.create(n_labels):fill_indices():add(train.n)

  local all_features, n_graph_features = train.solutions:bits_bipartite(train.n, n_labels, "adjacency")
  local graph_ranks = ivec.create(n_graph_features):fill(0, 0, train.n):fill(1, train.n, n_graph_features)
  local idf_ids, idf_scores = all_features:bits_top_df(n_total_nodes, n_graph_features)
  all_features:bits_select(idf_ids, nil, n_graph_features)
  n_graph_features = idf_ids:size()
  local ranks = ivec.create():copy(graph_ranks, idf_ids)
  train.graph_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.graph_index:add(all_features, all_ids)
  local label_features = ivec.create()
  all_features:bits_select(nil, label_node_ids, n_graph_features, label_features)
  train.labels_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.labels_index:add(label_features, label_node_ids)
  train.all_features = all_features
  train.all_ids = all_ids
  train.label_node_ids = label_node_ids
  str.printf("  Adjacency index: %d idf features, %d nodes, %d label nodes\n", idf_ids:size(), n_total_nodes, n_labels)

  print("\nSpot-checking graph index")
  local hood = rvec.create()
  local doc_labels = {}
  for i = 0, train.solutions:size() - 1 do
    local bit = train.solutions:get(i)
    local doc = math.floor(bit / n_labels)
    local label = bit % n_labels
    if not doc_labels[doc] then doc_labels[doc] = {} end
    doc_labels[doc][#doc_labels[doc] + 1] = label
  end

  for _, d in ipairs({ 0, 100, 1000 }) do
    local labels = doc_labels[d] or {}
    str.printf("  Doc %d: %d GT labels\n", d, #labels)
    train.graph_index:neighbors(d, 10, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    str.printf("    Top 10 neighbors:\n")
    for i = 0, hood:size() - 1 do
      local nid, dist = hood:get(i)
      local kind = nid >= train.n and "label" or "doc"
      local lid = nid >= train.n and (nid - train.n) or -1
      local is_gt = ""
      if kind == "label" then
        for _, gl in ipairs(labels) do
          if gl == lid then is_gt = " *GT*"; break end
        end
      end
      str.printf("      [%d] nid=%d (%s %d) dist=%.4f%s\n", i, nid, kind, lid, dist, is_gt)
    end

    if #labels > 0 then
      local l0 = labels[1]
      str.printf("    doc-%d <-> GT label %d: %.6f\n", d, l0, train.graph_index:similarity(d, train.n + l0, cfg.nystrom.decay, cfg.nystrom.bandwidth))
      if #labels > 1 then
        local l1 = labels[2]
        str.printf("    doc-%d <-> GT label %d: %.6f\n", d, l1, train.graph_index:similarity(d, train.n + l1, cfg.nystrom.decay, cfg.nystrom.bandwidth))
      end
      local neg_l = (l0 + 1) % n_labels
      for _, gl in ipairs(labels) do
        if gl == neg_l then neg_l = (l0 + 2) % n_labels; break end
      end
      str.printf("    doc-%d <-> NEG label %d: %.6f\n", d, neg_l, train.graph_index:similarity(d, train.n + neg_l, cfg.nystrom.decay, cfg.nystrom.bandwidth))

      local found_sharing = false
      for d2 = 0, train.n - 1 do
        if found_sharing then break end
        if d2 ~= d and doc_labels[d2] then
          for _, gl in ipairs(doc_labels[d2]) do
            if gl == l0 then
              str.printf("    doc-%d <-> doc-%d (share label %d): %.6f\n", d, d2, l0, train.graph_index:similarity(d, d2, cfg.nystrom.decay, cfg.nystrom.bandwidth))
              found_sharing = true
              break
            end
          end
        end
      end

      for d3 = 0, train.n - 1 do
        if doc_labels[d3] then
          local has = false
          for _, gl in ipairs(doc_labels[d3]) do
            for _, ml in ipairs(labels) do
              if gl == ml then has = true; break end
            end
            if has then break end
          end
          if not has then
            str.printf("    doc-%d <-> doc-%d (no shared labels): %.6f\n", d, d3, train.graph_index:similarity(d, d3, cfg.nystrom.decay, cfg.nystrom.bandwidth))
            break
          end
        end
      end

      str.printf("    label-%d <-> label-%d: %.6f\n", l0, (l0 + 1) % n_labels,
        train.graph_index:similarity(train.n + l0, train.n + ((l0 + 1) % n_labels), cfg.nystrom.decay, cfg.nystrom.bandwidth))
    end
  end

  print("\n  Doc->label lookup (labels-only index):")
  local doc_feats = ivec.create()
  for _, d in ipairs({ 0, 100, 1000 }) do
    local labels = doc_labels[d] or {}
    train.graph_index:get(d, doc_feats)
    train.labels_index:neighbors(doc_feats, 10, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    str.printf("    Doc %d (%d GT labels): top 10 label neighbors:\n", d, #labels)
    for i = 0, hood:size() - 1 do
      local nid, dist = hood:get(i)
      local lid = nid - train.n
      local is_gt = ""
      for _, gl in ipairs(labels) do
        if gl == lid then is_gt = " *GT*"; break end
      end
      str.printf("      [%d] label=%d dist=%.4f%s\n", i, lid, dist, is_gt)
    end
  end

  print("\n  Label neighbors (sample):")
  for _, l in ipairs({ 0, 100 }) do
    train.graph_index:neighbors(train.n + l, 5, hood, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    str.printf("    Label %d: top 5:\n", l)
    for i = 0, hood:size() - 1 do
      local nid, dist = hood:get(i)
      local kind = nid >= train.n and "label" or "doc"
      str.printf("      [%d] nid=%d (%s) dist=%.4f\n", i, nid, kind, dist)
    end
  end

  print("\nBuilding full evaluation adjacency (bipartite graph)")
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      knn_index = train.graph_index,
      knn = cfg.eval.knn,
      knn_cache = cfg.eval.knn,
      knn_decay = cfg.nystrom.decay,
      knn_bandwidth = cfg.nystrom.bandwidth,
      weight_index = train.graph_index,
      weight_decay = cfg.nystrom.decay,
      weight_bandwidth = cfg.nystrom.bandwidth,
      bridge = "none",
    })

  print("\nRunning spectral embedding (Nystrom)")
  local spectral_metrics
  local model = optimize.spectral({
    index = train.graph_index,
    train_tokens = train.all_features,
    train_ids = train.all_ids,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    bandwidth = cfg.nystrom.bandwidth,
    rounds = cfg.nystrom.rounds,
    samples = cfg.nystrom.samples,
    expected = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    eval = { ranking = cfg.eval.ranking },
    each = function(ev)
      util.spectral_log(ev)
      if ev.event == "eval" or ev.event == "done" then
        spectral_metrics = ev.metrics or ev.best_metrics
      end
    end,
  })
  local spectral_dims = model.dims
  local all_raw_codes = model.raw_codes
  local embedded_ids = model.ids
  local n_embedded = embedded_ids:size()
  str.printf("  Spectral dims: %d, embedded: %d, expected: %d (docs=%d, labels=%d)\n",
    spectral_dims, n_embedded, n_total_nodes, train.n, n_labels)
  str.printf("  embedded_ids: size=%d, min=%d, max=%d, train.n=%d\n",
    embedded_ids:size(), embedded_ids:min(), embedded_ids:max(), train.n)

  print("\nAnalyzing landmark selection:")
  local n_doc_landmarks = 0
  local n_label_landmarks = 0
  for i = 0, model.landmark_ids:size() - 1 do
    local lid = model.landmark_ids:get(i)
    if lid < train.n then n_doc_landmarks = n_doc_landmarks + 1
    else n_label_landmarks = n_label_landmarks + 1 end
  end
  str.printf("  Landmarks: %d docs, %d labels (%.1f%% labels)\n",
    n_doc_landmarks, n_label_landmarks, 100 * n_label_landmarks / model.landmark_ids:size())
  str.printf("  Expected if uniform: %.1f%% labels\n", 100 * n_labels / n_total_nodes)

  print("\nOptimizing SignRP configuration")
  local rp_ranks, rp_scores, rp_dims, rp_bits = optimize.rp({
    raw_codes = all_raw_codes,
    ids = embedded_ids,
    n_samples = n_embedded,
    n_dims = spectral_dims,
    max_bits = cfg.rp.max_bits,
    tolerance = cfg.rp.tolerance,
    eval = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    ranking = cfg.eval.ranking,
  })
  local best_idx, _ = rp_ranks:get(0)
  local best_score = rp_scores:get(best_idx)
  local best_dims = rp_dims:get(best_idx)
  local best_bits = rp_bits:get(best_idx)
  str.printf("  Best: dims=%d bits=%d score=%.4f\n", best_dims, best_bits, best_score)
  str.printf("  Scores (by preference):\n")
  for i = 0, math.min(10, rp_ranks:size() - 1) do
    local idx, rank = rp_ranks:get(i)
    local marker = (rank == 0) and " *" or ""
    str.printf("    dims=%d bits=%d: %.4f%s\n", rp_dims:get(idx), rp_bits:get(idx), rp_scores:get(idx), marker)
  end

  print("\nTruncating spectral codes to best dims")
  str.printf("  spectral_dims=%d, best_dims=%d, n_embedded=%d\n", spectral_dims, best_dims, n_embedded)
  str.printf("  all_raw_codes:size() before truncation = %d (expected %d)\n",
    all_raw_codes:size(), n_embedded * spectral_dims)
  if best_dims < spectral_dims then
    local selected_cols = ivec.create(best_dims):fill_indices()
    local truncated_all = dvec.create()
    all_raw_codes:mtx_select(selected_cols, nil, spectral_dims, truncated_all)
    all_raw_codes = truncated_all
    str.printf("  all_raw_codes:size() after truncation = %d (expected %d)\n",
      all_raw_codes:size(), n_embedded * best_dims)
  end
  train.dims = best_dims
  train.rp_bits = best_bits

  print("\nCreating normalizer and SignRP encoder from spectral codes")
  str.printf("  codes size=%d, n_dims=%d, n_samples=%d, expected=%d\n",
    all_raw_codes:size(), train.dims, n_embedded, n_embedded * train.dims)
  local normalize = hlth.normalizer({
    codes = all_raw_codes,
    n_dims = train.dims,
    n_samples = n_embedded,
  })
  local rp_encode, rp_n_bits = hlth.rp_encoder({
    n_dims = train.dims,
    rp_dims = train.rp_bits,
  })

  local train_wanted = ivec.create(train.n):fill_indices()
  str.printf("  train_wanted: size=%d, min=%d, max=%d\n",
    train_wanted:size(), train_wanted:min(), train_wanted:max())
  local train_embedded_ids = embedded_ids:set_intersect(train_wanted)
  str.printf("  train_embedded_ids: size=%d\n", train_embedded_ids:size())
  local train_raw_codes = dvec.create():mtx_extend(all_raw_codes, train_embedded_ids, embedded_ids, 0, best_dims, true)

  local label_wanted = ivec.create(n_labels):fill_indices():add(train.n)
  str.printf("  label_wanted: size=%d, min=%d, max=%d\n",
    label_wanted:size(), label_wanted:min(), label_wanted:max())
  local embedded_label_ids = embedded_ids:set_intersect(label_wanted)
  str.printf("  embedded_label_ids: size=%d\n", embedded_label_ids:size())
  local n_embedded_labels = embedded_label_ids:size()
  local label_raw_codes = dvec.create():mtx_extend(all_raw_codes, embedded_label_ids, embedded_ids, 0, best_dims, true)

  print("\nPre-computing label normalized and RP codes")
  local label_norm_codes = normalize(label_raw_codes)
  local label_codes_rp = rp_encode(label_norm_codes)
  str.printf("  Label codes: %d/%d labels embedded, %d dims normalized, %d bits RP\n",
    n_embedded_labels, n_labels, train.dims, rp_n_bits)
  train.embedded_label_ids = embedded_label_ids

  print("\nEvaluating spectral train codes (pre-regressor)")
  local train_norm_codes = normalize(train_raw_codes)
  local train_codes_rp = rp_encode(train_norm_codes)
  local sp_combined_norm = dvec.create()
  sp_combined_norm:copy(train_norm_codes)
  sp_combined_norm:copy(label_norm_codes)
  local sp_combined_rp = cvec.create()
  sp_combined_rp:copy(train_codes_rp)
  sp_combined_rp:copy(label_codes_rp)
  local sp_combined_ids = ivec.create(train.n):fill_indices()
  sp_combined_ids:copy(train.embedded_label_ids)
  local sp_raw_stats = eval.ranking_accuracy({
    raw_codes = sp_combined_norm, ids = sp_combined_ids, n_dims = train.dims,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })
  local sp_rp_stats = eval.ranking_accuracy({
    codes = sp_combined_rp, ids = sp_combined_ids, n_dims = rp_n_bits,
    eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
  })
  local sp_entropy = eval.entropy_stats(train_codes_rp, train.n, rp_n_bits)
  str.printf("  Raw ranking:  %.4f\n", sp_raw_stats.score)
  str.printf("  RP ranking:   %.4f\n", sp_rp_stats.score)
  str.printf("  RP entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    sp_entropy.mean, sp_entropy.min, sp_entropy.max, sp_entropy.std)
  local sp_label_ann = ann.create({ features = rp_n_bits })
  sp_label_ann:add(label_codes_rp, train.embedded_label_ids)
  local sp_hood_ids, sp_hoods = sp_label_ann:neighborhoods_by_vecs(train_codes_rp, cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
  local sp_expected_neighbors = ivec.create():copy(train.label_csr.neighbors):add(train.n)
  local sp_ks, sp_ret = eval.retrieval_ks({
    hoods = sp_hoods,
    hood_ids = sp_hood_ids,
    expected_offsets = train.label_csr.offsets,
    expected_neighbors = sp_expected_neighbors,
  })
  str.printf("  Retrieval micro: P=%.4f R=%.4f F1=%.4f\n",
    sp_ret.micro_precision, sp_ret.micro_recall, sp_ret.micro_f1)
  str.printf("  Retrieval macro: P=%.4f R=%.4f F1=%.4f\n",
    sp_ret.macro_precision, sp_ret.macro_recall, sp_ret.macro_f1)

  print("\nSpot-checking retrieval hoods")
  for _, doc_id in ipairs({0, 100, 1000}) do
    local hood = sp_hoods:get(doc_id)
    local gt_start = train.label_csr.offsets:get(doc_id)
    local gt_end = train.label_csr.offsets:get(doc_id + 1)
    local n_gt = gt_end - gt_start
    local gt_set = {}
    for j = gt_start, gt_end - 1 do
      gt_set[train.label_csr.neighbors:get(j) + train.n] = true
    end
    str.printf("  Doc %d: %d GT labels, hood size=%d\n", doc_id, n_gt, hood:size())
    for j = 0, math.min(9, hood:size() - 1) do
      local idx, dist = hood:get(j)
      local uid = sp_hood_ids:get(idx)
      local hit = gt_set[uid] and " HIT" or ""
      str.printf("    [%d] uid=%d (label=%d) dist=%d%s\n", j, uid, uid - train.n, dist, hit)
    end
    local n_hits = 0
    for j = 0, hood:size() - 1 do
      local idx, _ = hood:get(j)
      if gt_set[sp_hood_ids:get(idx)] then n_hits = n_hits + 1 end
    end
    str.printf("    Total hits: %d/%d (of %d GT)\n", n_hits, hood:size(), n_gt)
  end

  print("\nSpot-checking graph vs spectral distances")
  local spectral_index = ann.create({ features = rp_n_bits })
  local all_rp = cvec.create()
  all_rp:copy(train_codes_rp)
  all_rp:copy(label_codes_rp)
  local all_rp_ids = ivec.create(train.n):fill_indices()
  all_rp_ids:copy(train.embedded_label_ids)
  spectral_index:add(all_rp, all_rp_ids)
  for _, doc_id in ipairs({0, 100, 1000}) do
    local gt_start = train.label_csr.offsets:get(doc_id)
    local gt_end = train.label_csr.offsets:get(doc_id + 1)
    local n_gt = gt_end - gt_start
    str.printf("  Doc %d: %d ground-truth labels\n", doc_id, n_gt)
    for j = gt_start, math.min(gt_start + 2, gt_end - 1) do
      local label_idx = train.label_csr.neighbors:get(j)
      local label_nid = label_idx + train.n
      local g_sim = train.graph_index:similarity(doc_id, label_nid, cfg.nystrom.decay, cfg.nystrom.bandwidth)
      local s_dist = spectral_index:distance(doc_id, label_nid)
      str.printf("    GT label %d (nid=%d): graph_sim=%.4f spectral_dist=%.4f\n",
        label_idx, label_nid, g_sim, s_dist)
    end
    local neg_nid = train.n
    if train.label_csr.neighbors:get(gt_start) == 0 then neg_nid = train.n + 1 end
    local g_sim = train.graph_index:similarity(doc_id, neg_nid, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    local s_dist = spectral_index:distance(doc_id, neg_nid)
    str.printf("    NEG label (nid=%d): graph_sim=%.4f spectral_dist=%.4f\n",
      neg_nid, g_sim, s_dist)
  end

  if true then
    return
  end

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  local token_index = tok:index()
  tok = nil
  train.problems = nil
  dev.problems = nil
  test_set.problems = nil

  print("\nFeature selection (DF filter)")
  local min_df = cfg.feature_selection.min_df
  local max_df = cfg.feature_selection.max_df
  local df_ids, idf_weights = train.tokens:bits_top_df(train.n, n_tokens, nil, -min_df, max_df)
  str.printf("  DF filter (min=%d, max=%.0f%%): %d -> %d\n", min_df, max_df * 100, n_tokens, df_ids:size())
  train.tokens:bits_select(df_ids, nil, n_tokens)
  dev.tokens:bits_select(df_ids, nil, n_tokens)
  test_set.tokens:bits_select(df_ids, nil, n_tokens)
  n_tokens = df_ids:size()

  print("\nBNS feature selection")
  local bns_ids, bns_weights = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_labels, nil, cfg.feature_selection.bns_tokens, "max")
  local n_bns_tokens = bns_ids:size()
  str.printf("  BNS selection: %d -> %d tokens\n", n_tokens, n_bns_tokens)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  dev.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  n_tokens = n_bns_tokens

  print("\nFeature selection for regressor (F-score)")
  local union_feat_ids, _, class_offsets, class_feat_ids, class_scores = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, train.dims, cfg.feature_selection.per_class)
  str.printf("  Features: %d union, %d grouped (%d per dim x %d dims)\n",
    union_feat_ids:size(), class_feat_ids:size(), cfg.feature_selection.per_class, train.dims)
  str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
    class_scores:min(), class_scores:max(), class_scores:sum() / class_scores:size())

  local make_regressor_sentences = function(tokens, n_samples)
    return tokens:bits_to_cvec(n_samples, n_tokens, class_offsets, class_feat_ids, true)
  end

  local train_regressor_sentences, max_k = make_regressor_sentences(train.tokens, train.n)
  local n_regressor_features = max_k
  str.printf("  Regressor features: %d\n", n_regressor_features)

  print("\nTraining TM regressor")
  local predicted_buf = dvec.create()
  train.tm = optimize.regressor({
    outputs = train.dims,
    samples = train.n,
    problems = train_regressor_sentences,
    features = n_regressor_features,
    grouped = true,
    clauses = cfg.regressor.clauses,
    clause_tolerance = cfg.regressor.clause_tolerance,
    clause_maximum = cfg.regressor.clause_maximum,
    target = cfg.regressor.target,
    specificity = cfg.regressor.specificity,
    search_rounds = cfg.regressor.search_rounds,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    targets = train_raw_codes,
    search_metric = function (t)
      local predicted = t:regress(train_regressor_sentences, train.n, true, predicted_buf)
      local stats = eval.regression_accuracy(predicted, train_raw_codes)
      return -stats.mean, stats
    end,
    each = util.make_regressor_log(stopwatch),
  })

  print("\nBuilding labels-only ANN for retrieval evaluation")
  local label_ann = ann.create({ features = rp_n_bits })
  label_ann:add(label_codes_rp, train.embedded_label_ids)
  str.printf("  Label ANN: %d labels indexed, %d bits\n", train.embedded_label_ids:size(), rp_n_bits)

  local train_eval_adj = {
    ids = train_eval_ids, offsets = train_eval_offsets,
    neighbors = train_eval_neighbors, weights = train_eval_weights,
  }

  local function evaluate_split(split, name, eval_adj)
    print("\nEvaluating " .. name)
    local sentences = make_regressor_sentences(split.tokens, split.n)
    local predicted_raw = train.tm:regress(sentences, split.n, true)
    local predicted_norm = normalize(predicted_raw)
    local predicted_rp = rp_encode(predicted_norm)
    local combined_norm = dvec.create()
    combined_norm:copy(predicted_norm)
    combined_norm:copy(label_norm_codes)
    local combined_rp = cvec.create()
    combined_rp:copy(predicted_rp)
    combined_rp:copy(label_codes_rp)
    local combined_ids = ivec.create(split.n):fill_indices()
    combined_ids:copy(train.embedded_label_ids)
    local raw_stats = eval.ranking_accuracy({
      raw_codes = combined_norm, ids = combined_ids, n_dims = train.dims,
      eval_ids = eval_adj.ids, eval_offsets = eval_adj.offsets,
      eval_neighbors = eval_adj.neighbors, eval_weights = eval_adj.weights,
      ranking = cfg.eval.ranking,
    })
    local rp_stats = eval.ranking_accuracy({
      codes = combined_rp, ids = combined_ids, n_dims = rp_n_bits,
      eval_ids = eval_adj.ids, eval_offsets = eval_adj.offsets,
      eval_neighbors = eval_adj.neighbors, eval_weights = eval_adj.weights,
      ranking = cfg.eval.ranking,
    })
    local rp_entropy = eval.entropy_stats(predicted_rp, split.n, rp_n_bits)
    str.printf("  Raw ranking:  %.4f\n", raw_stats.score)
    str.printf("  RP ranking:   %.4f\n", rp_stats.score)
    str.printf("  RP entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
      rp_entropy.mean, rp_entropy.min, rp_entropy.max, rp_entropy.std)
    local expected_neighbors = ivec.create():copy(split.label_csr.neighbors):add(train.n)
    local hood_ids, hoods = label_ann:neighborhoods_by_vecs(predicted_rp, cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
    local ks, ret = eval.retrieval_ks({
      hoods = hoods,
      hood_ids = hood_ids,
      expected_offsets = split.label_csr.offsets,
      expected_neighbors = expected_neighbors,
    })
    str.printf("  Retrieval micro: P=%.4f R=%.4f F1=%.4f\n",
      ret.micro_precision, ret.micro_recall, ret.micro_f1)
    str.printf("  Retrieval macro: P=%.4f R=%.4f F1=%.4f\n",
      ret.macro_precision, ret.macro_recall, ret.macro_f1)
    return raw_stats.score, rp_stats.score, ret
  end

  local train_raw, train_rp, train_ret = evaluate_split(train, "train", train_eval_adj)
  local dev_raw, dev_rp, dev_ret = evaluate_split(dev, "dev", train_eval_adj)
  local test_raw, test_rp, test_ret = evaluate_split(test_set, "test", train_eval_adj)

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d  RP bits: %d\n", train.dims, rp_n_bits)
  str.printf("\n  Ranking Accuracy (%s):\n", cfg.eval.ranking)
  str.printf("                          Train      Dev      Test\n")
  str.printf("    Spectral kernel:     %.4f\n", spectral_metrics.kernel_score)
  str.printf("    Spectral raw:        %.4f\n", spectral_metrics.raw_score)
  str.printf("    Spectral SignRP:     %.4f\n", best_score)
  str.printf("    Spectral train raw:  %.4f\n", sp_raw_stats.score)
  str.printf("    Spectral train RP:   %.4f\n", sp_rp_stats.score)
  str.printf("    Predicted raw:       %.4f   %.4f   %.4f\n", train_raw, dev_raw, test_raw)
  str.printf("    Predicted SignRP:    %.4f   %.4f   %.4f\n", train_rp, dev_rp, test_rp)
  str.printf("\n  Entropy:\n")
  str.printf("    Spectral train RP:   mean=%.4f min=%.4f max=%.4f std=%.4f\n",
    sp_entropy.mean, sp_entropy.min, sp_entropy.max, sp_entropy.std)
  str.printf("\n  Retrieval (labels-only ANN, k=%d, radius=%d):\n", cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
  str.printf("                          micro P    micro R  micro F1   macro P    macro R  macro F1\n")
  str.printf("    Spectral train:      %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    sp_ret.micro_precision, sp_ret.micro_recall, sp_ret.micro_f1,
    sp_ret.macro_precision, sp_ret.macro_recall, sp_ret.macro_f1)
  str.printf("    Predicted train:     %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    train_ret.micro_precision, train_ret.micro_recall, train_ret.micro_f1,
    train_ret.macro_precision, train_ret.macro_recall, train_ret.macro_f1)
  str.printf("    Predicted dev:       %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    dev_ret.micro_precision, dev_ret.micro_recall, dev_ret.micro_f1,
    dev_ret.macro_precision, dev_ret.macro_recall, dev_ret.macro_f1)
  str.printf("    Predicted test:      %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
    test_ret.micro_precision, test_ret.micro_recall, test_ret.micro_f1,
    test_ret.macro_precision, test_ret.macro_recall, test_ret.macro_f1)
  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
