require("santoku.pvec")
local ann = require("santoku.learn.ann")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local cvec = require("santoku.cvec")
local eval = require("santoku.learn.evaluator")
local csr = require("santoku.learn.csr")
local quantizer = require("santoku.learn.quantizer")
local inv = require("santoku.learn.inv")
local ivec = require("santoku.ivec")
local rvec = require("santoku.rvec")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = {
    max = nil,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 1,
    cgrams_min = 0,
    cgrams_max = 0,
    cgrams_cross = false,
    skips = 0,
  },
  feature_selection = {
    n_bns = 8192,
    n_selected = 65536,
  },
  regressor = {
    features = { def = 4096, min = 512, max = 8192, pow2 = true },
    absorb_interval = 1,
    absorb_threshold = { def = 58, min = 0, max = 256, int = true },
    absorb_maximum = { def = 120, min = 1, max = 256, int = true },
    absorb_insert_offset = { def = 17, min = 1, max = 256, int = true },
    clauses = { def = 5, min = 1, max = 32, int = true },
    clause_maximum = { def = 102, min = 8, max = 512, int = true },
    clause_tolerance_fraction = { def = 0.76, min = 0.01, max = 1.0 },
    target_fraction = { def = 0.32, min = 0.01, max = 2.0 },
    specificity = { def = 512, min = 2, max = 2000 },
    alpha_tolerance = { def = -0.3, min = -3, max = 3 },
    alpha_maximum = { def = 0.7, min = -3, max = 3 },
    alpha_target = { def = 1.2, min = -3, max = 3 },
    alpha_specificity = { def = 2.1, min = -3, max = 3 },
    search_trials = 0,
    search_iterations = 20,
    search_subsample_targets = 32,
    final_patience = 2,
    final_batch = 20,
    final_iterations = 300,
  },
  nystrom = {
    n_landmarks = 1024,
    n_dims = 32,
    decay = 0,
    bandwidth = -1,
  },
  sfbs = {
    enable_pre = false,
    pre = {
      tolerance = 1e-6,
      max_dims = 256,
    },
    post = {
      tolerance = 1e-6,
    },
    n_neg = 48,
  },
  eval = {
    knn = 16,
    random_pairs = 16,
    retrieval_k = 32,
    retrieval_radius = 3,
  },
}

test("eurlex", function()

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
  dev.ids = ivec.create(dev.n):fill_indices():add(train.n + n_labels)
  test_set.ids = ivec.create(test_set.n):fill_indices():add(train.n + n_labels + dev.n)

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

  local doc_node_ids = ivec.create(train.n):fill_indices()
  local features, n_feat = train.solutions:bits_bipartite(train.n, n_labels, "adjacency")
  local graph_ranks = ivec.create(n_feat):fill(1, 0, train.n):fill(0, train.n, n_feat)
  local idf_ids, idf_scores = features:bits_top_df(n_total_nodes, n_feat)
  features:bits_select(idf_ids, nil, n_feat)
  local n_features = idf_ids:size()
  local ranks = ivec.create():copy(graph_ranks, idf_ids)
  train.graph_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.graph_index:add(features, all_ids)
  local label_features = ivec.create()
  features:bits_select(nil, label_node_ids, n_features, label_features)
  train.labels_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.labels_index:add(label_features, label_node_ids)
  local doc_features = ivec.create()
  features:bits_select(nil, doc_node_ids, n_features, doc_features)
  train.docs_index = inv.create({ features = idf_scores, ranks = ranks, n_ranks = 2 })
  train.docs_index:add(doc_features, doc_node_ids)
  train.all_features = features
  train.all_ids = all_ids
  train.label_node_ids = label_node_ids
  str.printf("  Adjacency index: %d idf features, %d nodes, %d label nodes\n",
    n_features, n_total_nodes, n_labels)

  print("\nBuilding evaluation adjacency (bipartite neg)")
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights = csr.bipartite_neg(
    train.label_csr.offsets, train.label_csr.neighbors,
    ivec.create(train.n):fill_indices(), label_node_ids,
    cfg.eval.random_pairs)
  str.printf("  %d nodes, %d edges\n", train_eval_ids:size(), train_eval_neighbors:size())
  do
    local kern = eval.ranking_accuracy({
      kernel_index = train.graph_index,
      kernel_decay = cfg.nystrom.decay,
      kernel_bandwidth = cfg.nystrom.bandwidth,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    })
    str.printf("  Kernel NDCG: %.4f\n", kern.score)
  end

  print("\nRunning spectral embedding (Nystrom)")
  local landmarks_index = train.labels_index
  local model = optimize.spectral({
    index = train.graph_index,
    landmarks_index = landmarks_index,
    train_tokens = train.all_features,
    train_ids = train.all_ids,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    bandwidth = cfg.nystrom.bandwidth,
    expected_ids = train_eval_ids,
    expected_offsets = train_eval_offsets,
    expected_neighbors = train_eval_neighbors,
    expected_weights = train_eval_weights,
    each = function(ev)
      util.spectral_log(ev)
    end,
  })
  local spectral_dims = model.dims
  local all_raw_codes = model.raw_codes
  local embedded_ids = model.ids
  local n_embedded = embedded_ids:size()
  local spectral_metrics = {
    kernel_score = eval.ranking_accuracy({
      kernel_index = train.graph_index,
      kernel_decay = cfg.nystrom.decay,
      kernel_bandwidth = cfg.nystrom.bandwidth,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    }).score,
    raw_score = eval.ranking_accuracy({
      raw_codes = all_raw_codes, ids = embedded_ids, n_dims = spectral_dims,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    }).score,
  }
  str.printf("  Spectral dims: %d, embedded: %d\n", spectral_dims, n_embedded)

  train.dims = spectral_dims

  local train_wanted = ivec.create(train.n):fill_indices()
  local train_embedded_ids = embedded_ids:set_intersect(train_wanted)
  local train_raw_codes = dvec.create():mtx_extend(all_raw_codes, train_embedded_ids, embedded_ids, 0, spectral_dims, true)

  local label_wanted = ivec.create(n_labels):fill_indices():add(train.n)
  local embedded_label_ids = embedded_ids:set_intersect(label_wanted)
  local n_embedded_labels = embedded_label_ids:size()
  local label_raw_codes = dvec.create():mtx_extend(all_raw_codes, embedded_label_ids, embedded_ids, 0, spectral_dims, true)
  train.embedded_label_ids = embedded_label_ids

  local encoder, bin_n_bits, label_codes_bin
  local sp_raw_stats, sp_bin_stats, sp_entropy, sp_ret

  if cfg.sfbs.enable_pre then

    print("\nQuantizing (spectral)")
    local sfbs_ids, sfbs_offsets, sfbs_neighbors, sfbs_weights = csr.bipartite_neg(
      train.label_csr.offsets, train.label_csr.neighbors,
      ivec.create(train.n):fill_indices(), label_node_ids,
      cfg.sfbs.n_neg)
    encoder = quantizer.create({
      raw_codes = all_raw_codes,
      ids = embedded_ids,
      n_samples = n_embedded,
      n_dims = spectral_dims,
      tolerance = cfg.sfbs.pre.tolerance,
      max_dims = cfg.sfbs.pre.max_dims,
      expected_ids = sfbs_ids,
      expected_offsets = sfbs_offsets,
      expected_neighbors = sfbs_neighbors,
      expected_weights = sfbs_weights,
      each = function(bit_idx, dim, threshold, gain, score, action)
        str.printf("  %s bit %d: dim=%d thresh=%.6f gain=%.6f score=%.4f\n",
          action, bit_idx, dim, threshold, gain, score)
      end,
    })
    bin_n_bits = encoder:n_bits()
    str.printf("  Selected %d bits (n_dims=%d)\n", bin_n_bits, encoder:n_dims())
    train.bin_bits = bin_n_bits

    print("\nPre-computing label binary codes")
    label_codes_bin = encoder:encode(label_raw_codes)
    str.printf("  Label codes: %d/%d labels embedded, %d dims, %d binary bits\n",
      n_embedded_labels, n_labels, train.dims, bin_n_bits)

    print("\n" .. string.rep("=", 60))
    print("PRE-REGRESSOR DIAGNOSTICS (all similarity [0,1], higher=closer)")
    print(string.rep("=", 60))

    local train_codes_bin = encoder:encode(train_raw_codes)

    local sp_label_ann = ann.create({ features = bin_n_bits })
    sp_label_ann:add(label_codes_bin, train.embedded_label_ids)

    local spectral_index = ann.create({ features = bin_n_bits })
    local all_bin = cvec.create()
    all_bin:copy(train_codes_bin)
    all_bin:copy(label_codes_bin)
    local all_bin_ids = ivec.create(train.n):fill_indices()
    all_bin_ids:copy(train.embedded_label_ids)
    spectral_index:add(all_bin, all_bin_ids)

    str.printf("\nRanking accuracy (doc->label, GT + %d random neg)\n", cfg.eval.random_pairs)
    local sp_combined_norm = dvec.create()
    sp_combined_norm:copy(train_raw_codes)
    sp_combined_norm:copy(label_raw_codes)
    local sp_combined_bin = cvec.create()
    sp_combined_bin:copy(train_codes_bin)
    sp_combined_bin:copy(label_codes_bin)
    local sp_combined_ids = ivec.create(train.n):fill_indices()
    sp_combined_ids:copy(train.embedded_label_ids)
    sp_raw_stats = eval.ranking_accuracy({
      raw_codes = sp_combined_norm, ids = sp_combined_ids, n_dims = train.dims,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    })
    sp_bin_stats = eval.ranking_accuracy({
      codes = sp_combined_bin, ids = sp_combined_ids, n_dims = bin_n_bits,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    })
    str.printf("  Graph kernel:  %.4f\n", spectral_metrics.kernel_score)
    str.printf("  Raw cosine:    %.4f\n", sp_raw_stats.score)
    str.printf("  Hamming:       %.4f\n", sp_bin_stats.score)

    str.printf("\nBinary bit entropy\n")
    sp_entropy = eval.entropy_stats(train_codes_bin, train.n, bin_n_bits)
    str.printf("  mean=%.4f min=%.4f max=%.4f std=%.4f\n",
      sp_entropy.mean, sp_entropy.min, sp_entropy.max, sp_entropy.std)

    str.printf("\nLabel retrieval (doc->label, Hamming ANN, k=%d, radius=%d)\n",
      cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
    local sp_expected_neighbors = ivec.create():copy(train.label_csr.neighbors):add(train.n)
    local sp_hood_ids, sp_hoods = sp_label_ann:neighborhoods_by_vecs(
      train_codes_bin, cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
    local sp_ks
    sp_ks, sp_ret = eval.retrieval_ks({
      hoods = sp_hoods,
      hood_ids = sp_hood_ids,
      expected_offsets = train.label_csr.offsets,
      expected_neighbors = sp_expected_neighbors,
    })
    str.printf("  micro: P=%.4f R=%.4f F1=%.4f\n",
      sp_ret.micro_precision, sp_ret.micro_recall, sp_ret.micro_f1)
    str.printf("  macro: P=%.4f R=%.4f F1=%.4f\n",
      sp_ret.macro_precision, sp_ret.macro_recall, sp_ret.macro_f1)
    str.printf("  optimal k: min=%d max=%d mean=%.1f\n",
      sp_ks:min(), sp_ks:max(), sp_ks:sum() / sp_ks:size())

    str.printf("\nSeparation analysis (doc->label, sampled)\n")
    do
      local n_samples = math.min(200, train.n)
      local step = math.max(1, math.floor(train.n / n_samples))
      local g_gt_sum, g_neg_sum, h_gt_sum, h_neg_sum = 0, 0, 0, 0
      local n_gt_pairs, n_neg_pairs = 0, 0
      local neg_step = math.max(1, math.floor(n_labels / 10))
      for si = 0, n_samples - 1 do
        local d = si * step
        if d >= train.n then break end
        local gt_start = train.label_csr.offsets:get(d)
        local gt_end = train.label_csr.offsets:get(d + 1)
        local gt_set = {}
        for j = gt_start, gt_end - 1 do
          local li = train.label_csr.neighbors:get(j)
          gt_set[li] = true
          local nid = li + train.n
          g_gt_sum = g_gt_sum + train.graph_index:similarity(d, nid, cfg.nystrom.decay, cfg.nystrom.bandwidth)
          h_gt_sum = h_gt_sum + spectral_index:similarity(d, nid)
          n_gt_pairs = n_gt_pairs + 1
        end
        local nc = 0
        for l = 0, n_labels - 1, neg_step do
          if not gt_set[l] and nc < 10 then
            local nid = l + train.n
            g_neg_sum = g_neg_sum + train.graph_index:similarity(d, nid, cfg.nystrom.decay, cfg.nystrom.bandwidth)
            h_neg_sum = h_neg_sum + spectral_index:similarity(d, nid)
            n_neg_pairs = n_neg_pairs + 1
            nc = nc + 1
          end
        end
      end
      local g_gt = n_gt_pairs > 0 and g_gt_sum / n_gt_pairs or 0
      local g_neg = n_neg_pairs > 0 and g_neg_sum / n_neg_pairs or 0
      local h_gt = n_gt_pairs > 0 and h_gt_sum / n_gt_pairs or 0
      local h_neg = n_neg_pairs > 0 and h_neg_sum / n_neg_pairs or 0
      str.printf("               graph    hamming\n")
      str.printf("  mean GT:     %.4f   %.4f\n", g_gt, h_gt)
      str.printf("  mean NEG:    %.4f   %.4f\n", g_neg, h_neg)
      str.printf("  gap:         %.4f   %.4f\n", g_gt - g_neg, h_gt - h_neg)
      str.printf("  (%d docs, %d GT pairs, %d NEG pairs)\n", n_samples, n_gt_pairs, n_neg_pairs)
    end

    str.printf("\nLabel code quality (%d labels, %d binary bits)\n", n_embedded_labels, bin_n_bits)
    do
      local sample_step = math.max(1, math.floor(n_embedded_labels / 30))
      local sim_sum, n_pairs, n_collisions = 0, 0, 0
      local collision_threshold = 1 - 5 / bin_n_bits
      for i = 0, n_embedded_labels - 1, sample_step do
        for j = i + 1, n_embedded_labels - 1, sample_step do
          local sim = spectral_index:similarity(
            train.embedded_label_ids:get(i),
            train.embedded_label_ids:get(j))
          sim_sum = sim_sum + sim
          n_pairs = n_pairs + 1
          if sim >= collision_threshold then
            n_collisions = n_collisions + 1
          end
        end
      end
      str.printf("  Mean pairwise label sim: %.4f (random expect: 0.5000)\n",
        n_pairs > 0 and sim_sum / n_pairs or 0)
      str.printf("  Near-collisions (<=5 bits): %d/%d pairs (%.1f%%)\n",
        n_collisions, n_pairs, n_pairs > 0 and 100 * n_collisions / n_pairs or 0)
    end

    str.printf("\nSpot-checks (GT labels + top non-GT retrieval neighbors)\n")
    for _, doc_id in ipairs({ 0, 100, 1000 }) do
      local gt_start = train.label_csr.offsets:get(doc_id)
      local gt_end = train.label_csr.offsets:get(doc_id + 1)
      local n_gt = gt_end - gt_start
      local gt_set = {}
      for j = gt_start, gt_end - 1 do
        gt_set[train.label_csr.neighbors:get(j)] = true
      end
      str.printf("\n  Doc %d (%d GT labels):\n", doc_id, n_gt)
      str.printf("    %-8s %-4s %-10s %-10s\n", "label", "GT?", "graph", "hamming")
      str.printf("    %-8s %-4s %-10s %-10s\n", "-----", "---", "-----", "-------")
      for j = gt_start, gt_end - 1 do
        local li = train.label_csr.neighbors:get(j)
        local nid = li + train.n
        str.printf("    %-8d  *   %.4f    %.4f\n", li,
          train.graph_index:similarity(doc_id, nid, cfg.nystrom.decay, cfg.nystrom.bandwidth),
          spectral_index:similarity(doc_id, nid))
      end
      local sp_hood = sp_hoods:get(doc_id)
      local shown = 0
      for j = 0, sp_hood:size() - 1 do
        if shown >= 5 then break end
        local idx, _ = sp_hood:get(j)
        local uid = sp_hood_ids:get(idx)
        local li = uid - train.n
        if not gt_set[li] then
          str.printf("    %-8d       %.4f    %.4f  <- non-GT\n", li,
            train.graph_index:similarity(doc_id, uid, cfg.nystrom.decay, cfg.nystrom.bandwidth),
            spectral_index:similarity(doc_id, uid))
          shown = shown + 1
        end
      end
      local hits = 0
      for j = 0, sp_hood:size() - 1 do
        local idx, _ = sp_hood:get(j)
        if gt_set[sp_hood_ids:get(idx) - train.n] then hits = hits + 1 end
      end
      str.printf("    Retrieval: %d/%d GT in top-%d\n", hits, n_gt, sp_hood:size())
    end

    print("\nTrimming spectral dims to used dims")
    local used_dims = encoder:used_dims()
    local n_used_dims = used_dims:size()
    str.printf("  %d / %d dims used\n", n_used_dims, spectral_dims)
    all_raw_codes:mtx_select(used_dims, nil, spectral_dims)
    train_raw_codes:mtx_select(used_dims, nil, spectral_dims)
    label_raw_codes:mtx_select(used_dims, nil, spectral_dims)
    encoder:restrict(used_dims)
    train.dims = n_used_dims

  end

  local predicted_buf = dvec.create()
  local n_selected = cfg.feature_selection.n_selected

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  dev.problems = nil
  test_set.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  print("\nBNS feature selection")
  train.solutions:add_scaled(n_labels)
  local bns_ids = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_labels,
    cfg.feature_selection.n_bns, nil, "max")
  train.solutions:add_scaled(-n_labels)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  dev.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  n_tokens = bns_ids:size()
  str.printf("  %d BNS features selected\n", n_tokens)

  print("\nFeature selection for regressor (F-score)")
  local union_ids, _, class_offsets, class_feat_ids, class_scores = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, train.dims, n_selected, nil, "sum")
  str.printf("  Features: %d grouped (%d per dim x %d dims), union: %d\n",
    class_feat_ids:size(), n_selected, train.dims, union_ids:size())
  str.printf("  F-scores: min=%.2f max=%.2f mean=%.2f\n",
    class_scores:min(), class_scores:max(), class_scores:sum() / class_scores:size())

  print("\nApplying feature selection")
  train.tokens:bits_select(union_ids, nil, n_tokens)
  dev.tokens:bits_select(union_ids, nil, n_tokens)
  test_set.tokens:bits_select(union_ids, nil, n_tokens)
  class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, union_ids)
  n_tokens = union_ids:size()
  str.printf("  Reduced vocabulary: %d tokens\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  local absorb_ranking_global = ivec.create(n_tokens):fill_indices()

  print("\nTraining TM regressor")
  train.tm = optimize.regressor({
    outputs = train.dims,
    samples = train.n,
    features = cfg.regressor.features,
    n_tokens = n_tokens,
    absorb_interval = cfg.regressor.absorb_interval,
    absorb_insert_offset = cfg.regressor.absorb_insert_offset,
    absorb_threshold = cfg.regressor.absorb_threshold,
    absorb_maximum = cfg.regressor.absorb_maximum,
    clauses = cfg.regressor.clauses,
    clause_maximum = cfg.regressor.clause_maximum,
    clause_tolerance_fraction = cfg.regressor.clause_tolerance_fraction,
    target_fraction = cfg.regressor.target_fraction,
    specificity = cfg.regressor.specificity,
    alpha_tolerance = cfg.regressor.alpha_tolerance,
    alpha_maximum = cfg.regressor.alpha_maximum,
    alpha_target = cfg.regressor.alpha_target,
    alpha_specificity = cfg.regressor.alpha_specificity,
    output_weights = model.eigenvalues,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    absorb_ranking_global = absorb_ranking_global,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    targets = train_raw_codes,
    search_subsample_samples = cfg.regressor.search_subsample_samples,
    search_subsample_targets = cfg.regressor.search_subsample_targets,
    search_metric = function (t, targs)
      local input = { tokens = targs.tokens, n_samples = targs.samples }
      local predicted = t:regress(input, targs.samples, true, predicted_buf)
      local stats = eval.regression_accuracy(predicted, targs.targets)
      return -stats.mean, stats
    end,
    each = util.make_regressor_log(stopwatch),
  })

  print("\nRegressor predicted raw ranking (train)")
  local train_predicted_raw = train.tm:regress({ tokens = train.tokens, n_samples = train.n }, train.n, true, predicted_buf)
  do
    local combined_raw = dvec.create()
    combined_raw:copy(train_predicted_raw)
    combined_raw:copy(label_raw_codes)
    local combined_ids = ivec.create(train.n):fill_indices()
    combined_ids:copy(train.embedded_label_ids)
    local pred_raw_stats = eval.ranking_accuracy({
      raw_codes = combined_raw, ids = combined_ids, n_dims = train.dims,
      eval_ids = train_eval_ids, eval_offsets = train_eval_offsets,
      eval_neighbors = train_eval_neighbors, eval_weights = train_eval_weights,
    })
    str.printf("  Predicted raw ranking (bipartite): %.4f (spectral ceiling: %.4f)\n",
      pred_raw_stats.score, spectral_metrics.raw_score)
  end

  print("\nDoc-doc ranking accuracy (apples-to-apples regression check)")
  do
    local doc_uids, doc_hoods = train.docs_index:neighborhoods(
      cfg.eval.knn, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    local doc_off, doc_nbr, doc_w = doc_hoods:to_csr(doc_uids)
    local rp_off, rp_nbr, rp_w = csr.random_pairs(doc_uids, cfg.eval.random_pairs)
    csr.weight_from_index(doc_uids, rp_off, rp_nbr, rp_w,
      train.docs_index, cfg.nystrom.decay, cfg.nystrom.bandwidth)
    csr.merge(doc_off, doc_nbr, doc_w, rp_off, rp_nbr, rp_w)
    csr.symmetrize(doc_off, doc_nbr, doc_w, doc_uids:size())
    str.printf("  Doc-doc eval: %d nodes, %d edges\n", doc_uids:size(), doc_nbr:size())
    local spectral_doc = eval.ranking_accuracy({
      raw_codes = train_raw_codes, ids = train.ids, n_dims = train.dims,
      eval_ids = doc_uids, eval_offsets = doc_off,
      eval_neighbors = doc_nbr, eval_weights = doc_w,
    })
    local predicted_doc = eval.ranking_accuracy({
      raw_codes = train_predicted_raw, ids = train.ids, n_dims = train.dims,
      eval_ids = doc_uids, eval_offsets = doc_off,
      eval_neighbors = doc_nbr, eval_weights = doc_w,
    })
    str.printf("  Spectral doc-doc:  %.4f\n", spectral_doc.score)
    str.printf("  Predicted doc-doc: %.4f (%.1f%% of ceiling)\n",
      predicted_doc.score, 100 * predicted_doc.score / spectral_doc.score)
  end

  print("\nPer-dimension regression analysis")
  do
    local D = train.dims
    local pd = eval.regression_per_dim(train_predicted_raw, train_raw_codes, train.n, D)
    str.printf("  %-6s %10s %10s %10s %12s\n", "Dim", "MAE", "Pearson r", "Var ratio", "Eigenvalue")
    local ev = model.eigenvalues
    local bands = { { 0, math.min(7, D - 1) } }
    if D > 8 then bands[#bands + 1] = { 8, math.min(31, D - 1) } end
    if D > 32 then bands[#bands + 1] = { 32, math.min(63, D - 1) } end
    if D > 64 then bands[#bands + 1] = { 64, math.min(127, D - 1) } end
    if D > 128 then bands[#bands + 1] = { 128, math.min(255, D - 1) } end
    if D > 256 then bands[#bands + 1] = { 256, D - 1 } end
    for _, band in ipairs(bands) do
      local lo, hi = band[1], band[2]
      local cnt = hi - lo + 1
      local s_mae, s_corr, s_vr, s_ev = 0, 0, 0, 0
      for dd = lo, hi do
        s_mae = s_mae + pd.mae:get(dd)
        s_corr = s_corr + pd.corr:get(dd)
        s_vr = s_vr + pd.var_ratio:get(dd)
        if dd < ev:size() then s_ev = s_ev + ev:get(dd) end
      end
      str.printf("  [%3d-%3d] %8.6f %10.4f %10.4f %12.6f\n",
        lo, hi, s_mae / cnt, s_corr / cnt, s_vr / cnt, s_ev / cnt)
    end
    local worst_mae_d = pd.mae:rmaxargs(D):get(0)
    local best_corr_d = pd.corr:rmaxargs(D):get(0)
    local worst_corr_d = pd.corr:rminargs(D):get(0)
    str.printf("  Best corr:  dim %d = %.4f\n", best_corr_d, pd.corr:get(best_corr_d))
    str.printf("  Worst corr: dim %d = %.4f\n", worst_corr_d, pd.corr:get(worst_corr_d))
    str.printf("  Worst MAE:  dim %d = %.6f\n", worst_mae_d, pd.mae:get(worst_mae_d))
  end

  local post_encoder, post_n_bits, post_label_codes

  do
    print("\nTraining post-quantizer on merged predicted+label codes")
    local post_raw = dvec.create()
    post_raw:copy(train_predicted_raw)
    post_raw:copy(label_raw_codes)
    local post_ids = ivec.create(train.n):fill_indices()
    post_ids:copy(train.embedded_label_ids)
    local post_n = train.n + n_embedded_labels

    local post_sfbs_ids, post_sfbs_off, post_sfbs_nbr, post_sfbs_w = csr.bipartite_neg(
      train.label_csr.offsets, train.label_csr.neighbors,
      ivec.create(train.n):fill_indices(), label_node_ids,
      cfg.sfbs.n_neg)

    post_encoder = quantizer.create({
      raw_codes = post_raw,
      ids = post_ids,
      n_samples = post_n,
      n_dims = train.dims,
      tolerance = cfg.sfbs.post.tolerance,
      expected_ids = post_sfbs_ids,
      expected_offsets = post_sfbs_off,
      expected_neighbors = post_sfbs_nbr,
      expected_weights = post_sfbs_w,
      each = function(bit_idx, dim, threshold, gain, score, action)
        str.printf("  %s bit %d: dim=%d thresh=%.6f gain=%.6f score=%.4f\n",
          action, bit_idx, dim, threshold, gain, score)
      end,
    })
    post_n_bits = post_encoder:n_bits()
    post_label_codes = post_encoder:encode(label_raw_codes)
    str.printf("  Post-quantizer: %d bits (pre: %s bits)\n",
      post_n_bits, bin_n_bits and tostring(bin_n_bits) or "n/a")
  end

  if post_encoder then

    print("\nBuilding labels-only ANN for retrieval evaluation")
    local label_ann = ann.create({ features = post_n_bits })
    label_ann:add(post_label_codes, train.embedded_label_ids)
    str.printf("  Label ANN: %d labels indexed, %d bits\n", train.embedded_label_ids:size(), post_n_bits)

    local train_eval_adj = {
      ids = train_eval_ids, offsets = train_eval_offsets,
      neighbors = train_eval_neighbors, weights = train_eval_weights,
    }

    local function evaluate_split(split, name, eval_adj)
      print("\nEvaluating " .. name)
      local predicted_raw = train.tm:regress({ tokens = split.tokens, n_samples = split.n }, split.n, true)
      local predicted_bin = post_encoder:encode(predicted_raw)
      local combined_raw = dvec.create()
      combined_raw:copy(predicted_raw)
      combined_raw:copy(label_raw_codes)
      local combined_bin = cvec.create()
      combined_bin:copy(predicted_bin)
      combined_bin:copy(post_label_codes)
      local combined_ids = ivec.create():copy(split.ids)
      combined_ids:copy(train.embedded_label_ids)
      local raw_stats = eval.ranking_accuracy({
        raw_codes = combined_raw, ids = combined_ids, n_dims = train.dims,
        eval_ids = eval_adj.ids, eval_offsets = eval_adj.offsets,
        eval_neighbors = eval_adj.neighbors, eval_weights = eval_adj.weights,
      })
      local bin_stats = eval.ranking_accuracy({
        codes = combined_bin, ids = combined_ids, n_dims = post_n_bits,
        eval_ids = eval_adj.ids, eval_offsets = eval_adj.offsets,
        eval_neighbors = eval_adj.neighbors, eval_weights = eval_adj.weights,
      })
      local bin_entropy = eval.entropy_stats(predicted_bin, split.n, post_n_bits)
      str.printf("  Raw ranking:  %.4f\n", raw_stats.score)
      str.printf("  Bin ranking:  %.4f\n", bin_stats.score)
      str.printf("  Bin entropy: mean=%.4f min=%.4f max=%.4f std=%.4f\n",
        bin_entropy.mean, bin_entropy.min, bin_entropy.max, bin_entropy.std)
      local expected_neighbors = ivec.create():copy(split.label_csr.neighbors):add(train.n)
      local hood_ids, hoods = label_ann:neighborhoods_by_vecs(predicted_bin, cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
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
      str.printf("  Retrieval optimal k: min=%d max=%d mean=%.1f\n",
        ks:min(), ks:max(), ks:sum() / ks:size())
      return raw_stats.score, bin_stats.score, ret
    end

    local train_raw, train_bin, train_ret = evaluate_split(train, "train", train_eval_adj)

    local dev_eval_ids, dev_eval_off, dev_eval_nbr, dev_eval_w = csr.bipartite_neg(
      dev.label_csr.offsets, dev.label_csr.neighbors,
      dev.ids, train.embedded_label_ids,
      cfg.eval.random_pairs)
    local dev_eval_adj = {
      ids = dev_eval_ids, offsets = dev_eval_off,
      neighbors = dev_eval_nbr, weights = dev_eval_w,
    }
    local dev_raw, dev_bin, dev_ret = evaluate_split(dev, "dev", dev_eval_adj)

    local test_eval_ids, test_eval_off, test_eval_nbr, test_eval_w = csr.bipartite_neg(
      test_set.label_csr.offsets, test_set.label_csr.neighbors,
      test_set.ids, train.embedded_label_ids,
      cfg.eval.random_pairs)
    local test_eval_adj = {
      ids = test_eval_ids, offsets = test_eval_off,
      neighbors = test_eval_nbr, weights = test_eval_w,
    }
    local test_raw, test_bin, test_ret = evaluate_split(test_set, "test", test_eval_adj)

    print("\n" .. string.rep("=", 60))
    print("SUMMARY")
    print(string.rep("=", 60))
    str.printf("  Spectral dims: %d  Pre bits: %s  Post bits: %d\n",
      train.dims, bin_n_bits and tostring(bin_n_bits) or "n/a", post_n_bits)
    str.printf("\n  Ranking Accuracy:\n")
    str.printf("                          Train      Dev      Test\n")
    str.printf("    Spectral kernel:     %.4f\n", spectral_metrics.kernel_score)
    str.printf("    Spectral raw:        %.4f\n", spectral_metrics.raw_score)
    if sp_raw_stats then
    str.printf("    Spectral train raw:  %.4f\n", sp_raw_stats.score)
    str.printf("    Spectral train bin:  %.4f\n", sp_bin_stats.score)
    end
    str.printf("    Predicted raw:       %.4f   %.4f   %.4f\n", train_raw, dev_raw, test_raw)
    str.printf("    Predicted bin:       %.4f   %.4f   %.4f\n", train_bin, dev_bin, test_bin)
    if sp_entropy then
    str.printf("\n  Entropy:\n")
    str.printf("    Spectral train bin:  mean=%.4f min=%.4f max=%.4f std=%.4f\n",
      sp_entropy.mean, sp_entropy.min, sp_entropy.max, sp_entropy.std)
    end
    str.printf("\n  Retrieval (labels-only ANN, k=%d, radius=%d):\n", cfg.eval.retrieval_k, cfg.eval.retrieval_radius)
    str.printf("                          micro P    micro R  micro F1   macro P    macro R  macro F1\n")
    if sp_ret then
    str.printf("    Spectral train:      %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
      sp_ret.micro_precision, sp_ret.micro_recall, sp_ret.micro_f1,
      sp_ret.macro_precision, sp_ret.macro_recall, sp_ret.macro_f1)
    end
    str.printf("    Predicted train:     %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
      train_ret.micro_precision, train_ret.micro_recall, train_ret.micro_f1,
      train_ret.macro_precision, train_ret.macro_recall, train_ret.macro_f1)
    str.printf("    Predicted dev:       %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
      dev_ret.micro_precision, dev_ret.micro_recall, dev_ret.micro_f1,
      dev_ret.macro_precision, dev_ret.macro_recall, dev_ret.macro_f1)
    str.printf("    Predicted test:      %.4f   %.4f   %.4f   %.4f   %.4f   %.4f\n",
      test_ret.micro_precision, test_ret.micro_recall, test_ret.micro_f1,
      test_ret.macro_precision, test_ret.macro_recall, test_ret.macro_f1)

  end

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
