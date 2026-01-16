local dvec = require("santoku.dvec")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local utc = require("santoku.utc")
local ivec = require("santoku.ivec")
local str = require("santoku.string")
local eval = require("santoku.tsetlin.evaluator")
local inv = require("santoku.tsetlin.inv")
local ann = require("santoku.tsetlin.ann")
local graph = require("santoku.tsetlin.graph")
local hlth = require("santoku.tsetlin.hlth")
local optimize = require("santoku.tsetlin.optimize")
local itq = require("santoku.tsetlin.itq")
local tokenizer = require("santoku.tokenizer")

local cfg; cfg = {
  early_exit_after = "retrieval",  -- nil, "spectral", "encoder", "retrieval"
  lookup_mode = "token",        -- "token" (k-NN in token space) | "unsupervised" (k-NN in unsup spectral space)
  aggregate_mode = "supervised", -- "supervised" (aggregate sup codes) | "unsupervised" (aggregate unsup codes)
  data = {
    max_per_class = nil,
    n_classes = 20,
    tvr = 0.1,
  },
  landmarks = {
    n_landmarks = { def = 16, min = 4, max = 32, int = true },
    n_thresholds = { def = 8, min = 2, max = 16, int = true },
    landmark_mode = "frequency",
    quantile = false,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 4,
    skips = 1,
    negations = 0,
  },
  feature_selection = {
    min_df = -2,
    max_df = 0.98,
    max_vocab = nil,
  },
  landmark_selection = {
    method = nil,  -- "chi2" (spectral codes), "coherence", or nil (use IDF)
    max_vocab = 8192,
    lambda = 0.5,     -- for coherence regularization
  },
  ann = {
    bucket_size = nil,
  },
  nystrom = {
    k_neighbors = 0,
    cmp = "jaccard",
    cmp_alpha = 0.5,
    cmp_beta = 0.5,
    probe_radius = 2,
    rank_filter = -1,
  },
  cluster = {
    enabled = true,
    verbose = true,
    knn = 64,
  },
  landmark_filter = {
    enabled = false,
    k_neighbors = 8,
    k_per_cluster = 400,
    invert = false,
    use_class_labels = true,
    random_select = true,
  },
  encoder = {
    enabled = true,
    verbose = true,
    mode = "raw",  -- "landmarks", "raw", or "unsup_to_sup"
    raw_max_vocab = 32768,
    raw_selection = "chi2",  -- "chi2" or "mi"
    unsup_n_dims = nil,  -- nil = use search.spectral.n_dims, or set explicit value
  },
  tm = {
    clauses = { def = 32, min = 8, max = 64, round = 8 },
    clause_tolerance = { def = 76, min = 16, max = 128, int = true },
    clause_maximum = { def = 21, min = 16, max = 128, int = true },
    target = { def = 33, min = 16, max = 128, int = true },
    specificity = { def = 520, min = 400, max = 4000 },
    include_bits = { def = 2, min = 1, max = 4, int = true },
  },
  tm_search = {
    patience = 2,
    rounds = 0,
    trials = 10,
    iterations = 20,
  },
  training = {
    patience = 20,
    iterations = 400,
  },
  classifier = {
    enabled = true,
    clauses = { def = 16, min = 8, max = 32, round = 8 },
    clause_tolerance = { def = 64, min = 16, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 32, min = 16, max = 128, int = true },
    specificity = { def = 1000, min = 400, max = 4000 },
    include_bits = { def = 1, min = 1, max = 4, int = true },
    negative = 0.5,
    search_patience = 2,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 20,
    final_iterations = 100,
  },
  search = {
    rounds = 0,  -- 0 = use baselines (def values), >0 = explore
    patience = 3,
    adjacency_samples = 8,
    spectral_samples = 4,
    select_samples = 8,
    eval_samples = 4,
    adjacency = {
      knn = { def = 24, min = 20, max = 35, int = true },
      knn_alpha = { def = 15, min = 10, max = 16, int = true },
      weight_decay = { def = 3.47, min = 2, max = 6 },
      knn_mutual = false,
      knn_mode = "cknn",
      knn_cache = 128,
      bridge = "mst",
    },
    spectral = {
      laplacian = "unnormalized",
      n_dims = 32,
      eps = 1e-8,
      threshold = {
        method = "itq",
        itq_iterations = 500,
        itq_tolerance = 1e-8,
      },
    },
    eval = {
      knn = 32,
      anchors = 16,
      pairs = 64,
      ranking = "ndcg",
      metric = "avg",
      max_consecutive_zeros = 3,
    },
    cluster_eval = {
      ranking = "ndcg",
      elbow = "plateau",
      elbow_alpha = 0.01,
      elbow_target = "quality",
    },
    verbose = true,
  },
}

test("newsgroups", function()

  local stopwatch = utc.stopwatch()

  print("Reading data")
  local train, test, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max_per_class,
    nil,
    cfg.data.tvr)

  str.printf("  Train:    %6d (%d categories)\n", train.n, train.n_labels)
  str.printf("  Validate: %6d (%d categories)\n", validate.n, validate.n_labels)
  str.printf("  Test:     %6d (%d categories)\n", test.n, test.n_labels)

  print("\nTraining tokenizer")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  str.printf("  Vocabulary: %d tokens\n", n_tokens)

  print("\nTokenizing train")
  train.tokens = tok:tokenize(train.problems)

  print("\nFeature selection (IDF filtering)")
  local idf_sorted, idf_weights
  idf_sorted, idf_weights = train.tokens:bits_top_df(
    train.n, n_tokens, cfg.feature_selection.max_vocab,
    cfg.feature_selection.min_df, cfg.feature_selection.max_df)
  local n_top_v = idf_sorted:size()
  str.printf("  DF filtered: %d features\n", n_top_v)
  str.printf("  IDF range: %.3f - %.3f\n", idf_weights:min(), idf_weights:max())
  tok:restrict(idf_sorted)
  print("\nRe-tokenizing with IDF-filtered vocabulary")
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test.tokens = tok:tokenize(test.problems)

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  validate.ids = ivec.create(validate.n)
  validate.ids:fill_indices()
  validate.ids:add(train.n)
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.ids:add(train.n + validate.n)

  local need_unsup_spectral = (cfg.lookup_mode == "unsupervised")
                           or (cfg.aggregate_mode == "unsupervised")
                           or (cfg.encoder.mode == "unsup_to_sup")

  print("\nBuilding supervised graph (categories + tokens)")
  do
    local graph_problems = ivec.create()
    train.tokens:bits_select(nil, train.ids, n_top_v, graph_problems)
    local cat_ext = ivec.create()
    cat_ext:copy(train.solutions)
    cat_ext:add_scaled(cfg.data.n_classes)
    graph_problems:bits_extend(cat_ext, n_top_v, cfg.data.n_classes)
    local graph_weights = dvec.create(n_top_v + cfg.data.n_classes)
    for i = 0, n_top_v - 1 do
      graph_weights:set(i, idf_weights:get(i))
    end
    graph_weights:fill(1.0, n_top_v, n_top_v + cfg.data.n_classes)
    local graph_ranks = ivec.create(n_top_v + cfg.data.n_classes)
    graph_ranks:fill(1, 0, n_top_v)
    graph_ranks:fill(0, n_top_v, n_top_v + cfg.data.n_classes)
    train.index_graph_sup = inv.create({
      features = graph_weights,
      ranks = graph_ranks,
      n_ranks = 2,
    })
    train.index_graph_sup:add(graph_problems, train.ids)
    str.printf("  Sup graph: %d features (%d tokens + %d categories)\n",
      n_top_v + cfg.data.n_classes, n_top_v, cfg.data.n_classes)
  end

  if need_unsup_spectral then
    print("\nBuilding unsupervised graph (tokens only)")
    train.index_graph_unsup = inv.create({
      features = idf_weights,
    })
    train.index_graph_unsup:add(train.tokens, train.ids)
    str.printf("  Unsup graph: %d tokens\n", n_top_v)
  end

  print("\nBuilding knn index (IDF-weighted tokens only)")
  train.node_features_graph = inv.create({
    features = idf_weights,
  })
  train.node_features_graph:add(train.tokens, train.ids)
  -- train.node_features_graph: knn index for spectral (tokens only, IDF-weighted)
  str.printf("  KNN index: %d tokens with IDF weights\n", n_top_v)
  train.train_raw = train.tokens:bits_to_cvec(train.n, n_top_v)

  local function build_category_ground_truth (ids, solutions)
    local cat_index = inv.create({ features = cfg.data.n_classes })
    local data = ivec.create()
    data:copy(solutions)
    data:add_scaled(cfg.data.n_classes)
    cat_index:add(data, ids)
    data:destroy()

    local adj_expected_ids, adj_expected_offsets, adj_expected_neighbors, adj_expected_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
      })

    local adj_cluster_ids, adj_cluster_offsets, adj_cluster_neighbors, adj_cluster_weights =
      graph.adjacency({
        category_index = cat_index,
        category_anchors = cfg.search.eval.anchors,
      })

    return cat_index, {
      retrieval = {
        ids = adj_expected_ids,
        offsets = adj_expected_offsets,
        neighbors = adj_expected_neighbors,
        weights = adj_expected_weights,
      },
      clustering = {
        ids = adj_cluster_ids,
        offsets = adj_cluster_offsets,
        neighbors = adj_cluster_neighbors,
        weights = adj_cluster_weights,
      },
    }
  end

  local function build_token_ground_truth (knn_index, knn)
    local adj_expected_ids, adj_expected_offsets, adj_expected_neighbors, adj_expected_weights =
      graph.adjacency({
        knn_index = knn_index,
        knn = knn,
        bridge = "none",
      })
    return {
      retrieval = {
        ids = adj_expected_ids,
        offsets = adj_expected_offsets,
        neighbors = adj_expected_neighbors,
        weights = adj_expected_weights,
      },
      clustering = {
        ids = adj_expected_ids,
        offsets = adj_expected_offsets,
        neighbors = adj_expected_neighbors,
        weights = adj_expected_weights,
      },
    }
  end

  print("\nBuilding category ground truth (for supervised spectral)")
  train.cat_index, train.ground_truth_sup = build_category_ground_truth(train.ids, train.solutions)

  if need_unsup_spectral then
    print("\nBuilding token k-NN ground truth (for unsupervised spectral)")
    train.ground_truth_unsup = build_token_ground_truth(train.node_features_graph, cfg.search.eval.knn)
    str.printf("  Using token-space %d-NN as expected neighbors\n", cfg.search.eval.knn)
  end

  print("\nOptimizing supervised spectral pipeline")
  local model_sup, spectral_best_params_sup, spectral_best_metrics_sup = optimize.spectral({
    index = train.index_graph_sup,
    knn_index = train.node_features_graph,
    bucket_size = cfg.ann.bucket_size,
    rounds = cfg.search.rounds,
    patience = cfg.search.patience,
    adjacency_samples = cfg.search.adjacency_samples,
    spectral_samples = cfg.search.spectral_samples,
    select_samples = cfg.search.select_samples,
    eval_samples = cfg.search.eval_samples,
    adjacency = cfg.search.adjacency,
    spectral = cfg.search.spectral,
    eval = cfg.search.eval,
    expected_ids = train.ground_truth_sup.retrieval.ids,
    expected_offsets = train.ground_truth_sup.retrieval.offsets,
    expected_neighbors = train.ground_truth_sup.retrieval.neighbors,
    expected_weights = train.ground_truth_sup.retrieval.weights,
    adjacency_each = cfg.search.verbose and function (ns, cs, es, stg)
      str.printf("    graph[%s]: %d nodes, %d edges, %d components\n", stg, ns, es, cs)
    end or nil,
    spectral_each = cfg.search.verbose and function (t, s)
      if t == "done" then
        str.printf("    spectral: %d matvecs\n", s)
      end
    end or nil,
    each = cfg.search.verbose and function (info)
      if info.event == "round_start" then
        str.printf("\n=== Round %d/%d ===\n", info.round, info.rounds)
      elseif info.event == "round_end" then
        str.printf("\n--- Round %d complete: best=%.4f (global=%.4f) ---\n",
          info.round, info.round_best_score, info.global_best_score)
      elseif info.event == "adjacency_cached" then
        str.printf("\n  [%d] CACHED key=%s\n", info.adj_sample, info.adj_key)
      elseif info.event == "stage" and info.stage == "adjacency" then
        local p = info.params.adjacency
        if info.is_final then
          str.printf("\n  [Final] decay=%.2f knn=%d alpha=%d mutual=%s mode=%s\n",
            p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual), p.knn_mode)
        else
          str.printf("\n  [%d] decay=%.2f knn=%d alpha=%d mutual=%s mode=%s\n",
            info.sample, p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual), p.knn_mode)
        end
      elseif info.event == "stage" and info.stage == "spectral" then
        local p = info.params.spectral
        str.printf("    lap=%s dims=%d\n", p.laplacian, p.n_dims)
      elseif info.event == "stage" and info.stage == "select" then
        local p = info.params.select
        local thresh_str = "unknown"
        if p.threshold_params and p.threshold_params.method then
          local m = p.threshold_params.method
          if m == "itq" then
            local iters = p.threshold_params.iterations or 100
            local tol = p.threshold_params.tolerance or 1e-8
            thresh_str = str.format("itq(i=%d,t=%.0e)", iters, tol)
          elseif m == "otsu" then
            local metric = p.threshold_params.metric or "variance"
            local minimize = p.threshold_params.minimize and "min" or "max"
            thresh_str = str.format("otsu(%s,%s)", metric, minimize)
          else
            thresh_str = m
          end
        end
        local select_str = p.select_metric or "none"
        local select_elbow_str = p.select_elbow or "none"
        local select_alpha_str = p.select_elbow_alpha and str.format("%.1f", p.select_elbow_alpha) or "-"
        str.printf("    thresh=%s select=%s(%s,a=%s)\n",
          thresh_str, select_str, select_elbow_str, select_alpha_str)
      elseif info.event == "select_result" then
        str.printf("    selected_dims=%d\n", info.selected_dims)
      elseif info.event == "eval" then
        local e = info.params.eval
        local m = info.metrics
        str.printf("    knn=%d score=%.4f\n", e.knn, m.score)
      end
    end or nil,
  })

  train.codes_sup = model_sup.codes
  train.ids_sup = model_sup.ids
  train.dims_sup = model_sup.dims
  train.index_sup = model_sup.index
  train.spectral_params_sup = spectral_best_params_sup
  train.retrieval_stats = spectral_best_metrics_sup

  if need_unsup_spectral then
    print("\nOptimizing unsupervised spectral pipeline")
    local unsup_adjacency = {
      knn = 20,
      knn_alpha = 14,
      weight_decay = 0,
      knn_mutual = cfg.search.adjacency.knn_mutual,
      knn_mode = cfg.search.adjacency.knn_mode,
      knn_cache = cfg.search.adjacency.knn_cache,
      bridge = cfg.search.adjacency.bridge,
    }
    local unsup_spectral = {}
    for k, v in pairs(cfg.search.spectral) do unsup_spectral[k] = v end
    if cfg.encoder.unsup_n_dims then
      unsup_spectral.n_dims = cfg.encoder.unsup_n_dims
    end
    local unsup_eval = {}
    for k, v in pairs(cfg.search.eval) do unsup_eval[k] = v end
    local model_unsup = optimize.spectral({
      index = train.index_graph_unsup,
      knn_index = train.node_features_graph,
      bucket_size = cfg.ann.bucket_size,
      rounds = cfg.search.rounds,
      patience = cfg.search.patience,
      adjacency_samples = cfg.search.adjacency_samples,
      spectral_samples = cfg.search.spectral_samples,
      select_samples = cfg.search.select_samples,
      eval_samples = cfg.search.eval_samples,
      adjacency = unsup_adjacency,
      spectral = unsup_spectral,
      eval = unsup_eval,
      expected_ids = train.ground_truth_unsup.retrieval.ids,
      expected_offsets = train.ground_truth_unsup.retrieval.offsets,
      expected_neighbors = train.ground_truth_unsup.retrieval.neighbors,
      expected_weights = train.ground_truth_unsup.retrieval.weights,
      adjacency_each = cfg.search.verbose and function (ns, cs, es, stg)
        if stg == "kruskal" or stg == "done" then
          str.printf("    graph: %d nodes, %d edges, %d components\n", ns, es, cs)
        end
      end or nil,
      spectral_each = cfg.search.verbose and function (t, s)
        if t == "done" then
          str.printf("    spectral: %d matvecs\n", s)
        end
      end or nil,
    })
    train.codes_unsup = model_unsup.codes
    train.ids_unsup = model_unsup.ids
    train.dims_unsup = model_unsup.dims
    train.index_unsup = model_unsup.index
    train.raw_codes_unsup = model_unsup.raw_codes
    train.eigenvalues_unsup = model_unsup.eigs
    train.threshold_state_unsup = model_unsup.threshold_state
    train.threshold_fn_unsup = model_unsup.threshold_fn
    train.threshold_params_unsup = model_unsup.threshold_params
    str.printf("  Unsupervised model: %d dims, threshold_state=%s\n",
      train.dims_unsup,
      train.threshold_state_unsup and (train.threshold_state_unsup.rotation and "itq" or
        train.threshold_state_unsup.thresholds and "median/otsu" or "sign") or "nil")
  end

  local nystrom_encode
  if cfg.lookup_mode == "unsupervised" or cfg.encoder.mode == "unsup_to_sup" then
    print("\nCreating Nyström encoder for unsupervised codes")
    local threshold_fn = function(codes, n_dims)
      local state = train.threshold_state_unsup
      local method = train.threshold_params_unsup.method
      if method == "itq" and state and state.rotation and state.mean then
        return itq.itq({
          codes = codes,
          n_dims = n_dims,
          rotation = state.rotation,
          mean = state.mean,
        })
      elseif method == "median" and state and state.thresholds then
        return itq.median({
          codes = codes,
          n_dims = n_dims,
          thresholds = state.thresholds,
        })
      elseif method == "otsu" and state and state.thresholds and state.indices then
        return itq.otsu({
          codes = codes,
          n_dims = n_dims,
          thresholds = state.thresholds,
          indices = state.indices,
        })
      else
        return train.threshold_fn_unsup(codes, n_dims)
      end
    end
    nystrom_encode = hlth.nystrom_encoder({
      features_index = train.node_features_graph,
      eigenvectors = train.raw_codes_unsup,
      eigenvalues = train.eigenvalues_unsup,
      ids = train.ids_unsup,
      n_dims = train.dims_unsup,
      threshold = threshold_fn,
      k_neighbors = cfg.nystrom.k_neighbors,
      cmp = cfg.nystrom.cmp,
      cmp_alpha = cfg.nystrom.cmp_alpha,
      cmp_beta = cfg.nystrom.cmp_beta,
      probe_radius = cfg.nystrom.probe_radius,
      rank_filter = cfg.nystrom.rank_filter,
    })
    str.printf("  Nyström encoder: %d dims\n", train.dims_unsup)
  end

  if cfg.early_exit_after == "spectral" then
    print("\n=== EARLY EXIT after spectral ===")
    return
  end

  local landmark_weights, n_landmark_v
  if cfg.encoder.mode ~= "raw" then
    if cfg.landmark_selection.method == "coherence" then
      print("\nSelecting tokens by code coherence")
      local landmark_vocab, coherence_scores = train.tokens:bits_top_coherence(
        train.codes_sup,
        train.n,
        n_top_v,
        train.dims_sup,
        cfg.landmark_selection.max_vocab,
        cfg.landmark_selection.lambda)
      n_landmark_v = landmark_vocab:size()
      str.printf("  Selected %d tokens by coherence (score range: %.4f - %.4f)\n",
        n_landmark_v, coherence_scores:get(coherence_scores:size() - 1), coherence_scores:get(0))
      local words = tok:index()
      print("\n  Top 30 by coherence score:")
      for i = 0, math.min(29, n_landmark_v - 1) do
        local id = landmark_vocab:get(i)
        local score = coherence_scores:get(i)
        str.printf("    %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
      end
      print("\n  Bottom 30 by coherence score:")
      for i = 0, math.min(29, n_landmark_v - 1) do
        local idx = n_landmark_v - 1 - i
        local id = landmark_vocab:get(idx)
        local score = coherence_scores:get(idx)
        str.printf("    %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
      end
      tok:restrict(landmark_vocab)
      train.tokens_landmark = tok:tokenize(train.problems)
      validate.tokens_landmark = tok:tokenize(validate.problems)
      test.tokens_landmark = tok:tokenize(test.problems)
      landmark_weights = coherence_scores
    elseif cfg.landmark_selection.method == "chi2" then
      print("\nSelecting tokens by chi2 (spectral code association)")
      local landmark_vocab, chi2_scores = train.tokens:bits_top_chi2(
        train.codes_sup,
        train.n,
        n_top_v,
        train.dims_sup,
        cfg.landmark_selection.max_vocab)
      n_landmark_v = landmark_vocab:size()
      str.printf("  Selected %d tokens by chi2 (score range: %.4f - %.4f)\n",
        n_landmark_v, chi2_scores:get(chi2_scores:size() - 1), chi2_scores:get(0))
      local words = tok:index()
      print("\n  Top 30 by chi2 score:")
      for i = 0, math.min(29, n_landmark_v - 1) do
        local id = landmark_vocab:get(i)
        local score = chi2_scores:get(i)
        str.printf("    %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
      end
      print("\n  Bottom 30 by chi2 score:")
      for i = 0, math.min(29, n_landmark_v - 1) do
        local idx = n_landmark_v - 1 - i
        local id = landmark_vocab:get(idx)
        local score = chi2_scores:get(idx)
        str.printf("    %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
      end
      tok:restrict(landmark_vocab)
      train.tokens_landmark = tok:tokenize(train.problems)
      validate.tokens_landmark = tok:tokenize(validate.problems)
      test.tokens_landmark = tok:tokenize(test.problems)
      landmark_weights = chi2_scores
    else
      train.tokens_landmark = train.tokens
      validate.tokens_landmark = validate.tokens
      test.tokens_landmark = test.tokens
      landmark_weights = idf_weights
      n_landmark_v = n_top_v
    end

    print("\nBuilding landmark index")
    train.node_features = inv.create({
      features = landmark_weights,
    })
    train.node_features:add(train.tokens_landmark, train.ids)
    str.printf("  Landmark index: %d tokens\n", n_landmark_v)
  end


  local spectral_ground_truth = build_token_ground_truth(train.index_sup, cfg.search.eval.knn)
  train.adj_expected_ids = spectral_ground_truth.retrieval.ids
  train.adj_expected_offsets = spectral_ground_truth.retrieval.offsets
  train.adj_expected_neighbors = spectral_ground_truth.retrieval.neighbors
  train.adj_expected_weights = spectral_ground_truth.retrieval.weights

  print("\nSpot-check: k-NN Classification & Precision (spectral codes)")
  do
    local max_k = 10
    local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
      knn_index = train.index_sup,
      knn = max_k + 1,
      bridge = "none",
    })

    local id_to_idx = {}
    for i = 0, knn_ids:size() - 1 do
      id_to_idx[knn_ids:get(i)] = i
    end

    local k_values = { 1, 3, 5, 10 }
    for _, k in ipairs(k_values) do
      local correct = 0
      local total_precision = 0
      local counted = 0
      for i = 0, train.n - 1 do
        local query_id = train.ids:get(i)
        local idx = id_to_idx[query_id]
        if idx then
          local query_label = train.solutions:get(i)
          local offset_start = knn_offsets:get(idx)
          local offset_end = knn_offsets:get(idx + 1)
          local votes = {}
          local same_class = 0
          local neighbor_count = 0
          for j = offset_start, offset_end - 1 do
            local neighbor_idx = knn_neighbors:get(j)
            local neighbor_id = knn_ids:get(neighbor_idx)
            if neighbor_id ~= query_id and neighbor_id < train.n then
              neighbor_count = neighbor_count + 1
              if neighbor_count <= k then
                local neighbor_label = train.solutions:get(neighbor_id)
                votes[neighbor_label] = (votes[neighbor_label] or 0) + 1
                if neighbor_label == query_label then
                  same_class = same_class + 1
                end
              end
            end
          end
          if neighbor_count > 0 then
            local best_label, best_count = -1, 0
            for lbl, cnt in pairs(votes) do
              if cnt > best_count then
                best_label, best_count = lbl, cnt
              end
            end
            if best_label == query_label then
              correct = correct + 1
            end
            total_precision = total_precision + same_class / k
            counted = counted + 1
          end
        end
      end
      if counted > 0 then
        str.printf("  k=%2d: Accuracy = %.2f%% | Precision = %.2f%%\n",
          k, 100 * correct / counted, 100 * total_precision / counted)
      end
    end
    knn_ids:destroy()
    knn_offsets:destroy()
    knn_neighbors:destroy()
  end

  print("\nSpot-check: Example Retrievals (5 random samples)")
  do
    local k = 5
    local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
      knn_index = train.index_sup,
      knn = k + 1,
      bridge = "none",
    })
    local id_to_idx = {}
    for i = 0, knn_ids:size() - 1 do
      id_to_idx[knn_ids:get(i)] = i
    end
    local samples = { 0, 100, 500, 1000, 5000 }
    for _, i in ipairs(samples) do
      if i < train.n then
        local query_id = train.ids:get(i)
        local idx = id_to_idx[query_id]
        if idx then
          local query_label = train.solutions:get(i)
          local query_cat = train.categories[query_label + 1]
          local offset_start = knn_offsets:get(idx)
          local offset_end = knn_offsets:get(idx + 1)
          str.printf("  Sample %d [%s]:\n", i, query_cat)
          local shown = 0
          for j = offset_start, offset_end - 1 do
            local neighbor_idx = knn_neighbors:get(j)
            local neighbor_id = knn_ids:get(neighbor_idx)
            if neighbor_id ~= query_id and neighbor_id < train.n and shown < k then
              local neighbor_label = train.solutions:get(neighbor_id)
              local neighbor_cat = train.categories[neighbor_label + 1]
              local match = neighbor_cat == query_cat and "+" or "-"
              str.printf("    %d. [%s] %s\n", shown + 1, match, neighbor_cat)
              shown = shown + 1
            end
          end
        end
      end
    end
    knn_ids:destroy()
    knn_offsets:destroy()
    knn_neighbors:destroy()
  end

  print("\nPer-class train k=5 accuracy (spectral codes, train-to-train)")
  do
    local k = 5
    local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
      knn_index = train.index_sup,
      knn = k + 1,
      bridge = "none",
    })
    local id_to_idx = {}
    for i = 0, knn_ids:size() - 1 do
      id_to_idx[knn_ids:get(i)] = i
    end
    local class_correct = {}
    local class_total = {}
    for c = 0, cfg.data.n_classes - 1 do
      class_correct[c] = 0
      class_total[c] = 0
    end
    for i = 0, train.n - 1 do
      local query_id = train.ids:get(i)
      local idx = id_to_idx[query_id]
      if idx then
        local query_label = train.solutions:get(i)
        local offset_start = knn_offsets:get(idx)
        local offset_end = knn_offsets:get(idx + 1)
        local votes = {}
        local neighbor_count = 0
        for j = offset_start, offset_end - 1 do
          local neighbor_idx = knn_neighbors:get(j)
          local neighbor_id = knn_ids:get(neighbor_idx)
          if neighbor_id ~= query_id and neighbor_id < train.n then
            neighbor_count = neighbor_count + 1
            if neighbor_count <= k then
              local neighbor_label = train.solutions:get(neighbor_id)
              votes[neighbor_label] = (votes[neighbor_label] or 0) + 1
            end
          end
        end
        if neighbor_count > 0 then
          class_total[query_label] = class_total[query_label] + 1
          local best_label, best_count = -1, 0
          for lbl, cnt in pairs(votes) do
            if cnt > best_count then
              best_label, best_count = lbl, cnt
            end
          end
          if best_label == query_label then
            class_correct[query_label] = class_correct[query_label] + 1
          end
        end
      end
    end
    local sorted_classes = {}
    for c = 0, cfg.data.n_classes - 1 do
      local acc = class_total[c] > 0 and (class_correct[c] / class_total[c]) or 0
      table.insert(sorted_classes, { c = c, acc = acc, n = class_total[c] })
    end
    table.sort(sorted_classes, function(a, b) return a.acc < b.acc end)
    for _, item in ipairs(sorted_classes) do
      local cat = train.categories[item.c + 1]
      str.printf("  %5.1f%% (%4d) %s\n", 100 * item.acc, item.n, cat)
    end
    knn_ids:destroy()
    knn_offsets:destroy()
    knn_neighbors:destroy()
  end

  if cfg.cluster.enabled then
    print("\nSetting up clustering adjacency")
    local adj_cluster_ids, adj_cluster_offsets, adj_cluster_neighbors =
      graph.adjacency({
        knn_index = train.index_sup,
        knn = cfg.cluster.knn,
        each = function (ns, cs, es, stg)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Stage: %-10s  Nodes: %5d  Components: %5s  Edges: %5s\n",
            d, dd, stg, ns, cs, es)
        end
      })

    print("\nClustering")
    local cluster_codes = train.index_sup:get(adj_cluster_ids)
    train.codes_clusters = eval.cluster({
      codes = cluster_codes,
      n_dims = train.dims_sup,
      ids = adj_cluster_ids,
      offsets = adj_cluster_offsets,
      neighbors = adj_cluster_neighbors,
      quality = true,
    })

    if cfg.cluster.verbose then
      for step = 0, train.codes_clusters.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Quality: %.2f | Clusters: %d\n",
          d, dd, step, train.codes_clusters.quality_curve:get(step),
          train.codes_clusters.n_clusters_curve:get(step))
      end
    end

    str.printf("\nClustering Metrics\n")
    str.printf("  Best Step: %d (lmethod elbow on log-scaled intracluster similarity)\n", train.codes_clusters.best_step)
    str.printf("  Quality: %.4f (average similarity to cluster centroid)\n",
      train.codes_clusters.quality)
    str.printf("  Clusters: %d (number of clusters at best step, target is %d categories)\n",
      train.codes_clusters.n_clusters, cfg.data.n_classes)

    if cfg.landmark_filter.enabled then
      print("\nLandmark Filtering (cluster-based authority selection)")

      local cluster_assignments
      local n_clusters_used
      if cfg.landmark_filter.use_class_labels then
        print("  Using CLASS LABELS as perfect clusters")
        cluster_assignments = train.solutions
        n_clusters_used = cfg.data.n_classes
      else
        print("  Getting cluster assignments at best step...")
        cluster_assignments = eval.dendro_cut(
          train.codes_clusters.offsets,
          train.codes_clusters.merges,
          train.codes_clusters.best_step)
        n_clusters_used = train.codes_clusters.n_clusters
      end
      str.printf("  Cluster assignments: %d samples, %d clusters\n",
        cluster_assignments:size(), n_clusters_used)

      print("  Computing authority scores (mean hamming to token-space k-NN)...")
      local authority_scores = hlth.authority_scores({
        token_index = train.node_features,
        code_index = train.index_sup,
        ids = train.ids,
        tokens = train.tokens,
        n_samples = train.n,
        k_neighbors = cfg.landmark_filter.k_neighbors,
      })
      local min_score, max_score, mean_score = 1.0, 0.0, 0.0
      for i = 0, train.n - 1 do
        local s = authority_scores:get(i)
        if s < min_score then min_score = s end
        if s > max_score then max_score = s end
        mean_score = mean_score + s
      end
      mean_score = mean_score / train.n
      str.printf("  Authority scores: min=%.3f max=%.3f mean=%.3f\n", min_score, max_score, mean_score)
      if cfg.landmark_filter.invert then
        print("  INVERTING scores (selecting HIGHEST inconsistency)")
        for i = 0, train.n - 1 do
          authority_scores:set(i, -authority_scores:get(i))
        end
      end

      print("  Selecting top-K landmarks per cluster...")
      local remapped_assignments
      local next_seq
      if cfg.landmark_filter.use_class_labels then
        remapped_assignments = cluster_assignments
        next_seq = n_clusters_used
      else
        local cluster_id_to_seq = {}
        next_seq = 0
        remapped_assignments = ivec.create(cluster_assignments:size())
        for i = 0, cluster_assignments:size() - 1 do
          local c = cluster_assignments:get(i)
          if not cluster_id_to_seq[c] then
            cluster_id_to_seq[c] = next_seq
            next_seq = next_seq + 1
          end
          remapped_assignments:set(i, cluster_id_to_seq[c])
        end
      end
      str.printf("  Found %d unique clusters\n", next_seq)

      local selected_indices
      if cfg.landmark_filter.random_select then
        print("  Using RANDOM selection per cluster")
        for i = 0, train.n - 1 do
          authority_scores:set(i, math.random())
        end
      end
      selected_indices = hlth.select_landmarks({
        scores = authority_scores,
        assignments = remapped_assignments,
        k_per_cluster = cfg.landmark_filter.k_per_cluster,
      })
      str.printf("  Selected %d landmarks (from %d clusters, k=%d per cluster)\n",
        selected_indices:size(), next_seq, cfg.landmark_filter.k_per_cluster)
      if not cfg.landmark_filter.use_class_labels then
        remapped_assignments:destroy()
      end

      print("  Building filtered landmark index...")
      local selected_ids = ivec.create(selected_indices:size())
      local selected_tokens = ivec.create()
      for i = 0, selected_indices:size() - 1 do
        local idx = selected_indices:get(i)
        local uid = train.ids:get(idx)
        selected_ids:set(i, uid)
      end
      train.tokens:bits_select(nil, selected_ids, n_top_v, selected_tokens)

      train.filtered_node_features = inv.create({
        features = idf_weights,
      })
      train.filtered_node_features:add(selected_tokens, selected_ids)
      train.landmark_ids = selected_ids
      str.printf("  Filtered landmark index: %d landmarks\n", selected_ids:size())

      selected_indices:destroy()
      cluster_assignments:destroy()
      authority_scores:destroy()
      selected_tokens:destroy()
    end
  end

  collectgarbage("collect")

  if cfg.encoder.enabled then

    local train_solutions = train.index_sup:get(train.ids)

    local encoder_args = {
      hidden = train.dims_sup,
      codes = train_solutions,
      samples = train.n,
      clauses = cfg.tm.clauses,
      clause_tolerance = cfg.tm.clause_tolerance,
      clause_maximum = cfg.tm.clause_maximum,
      target = cfg.tm.target,
      specificity = cfg.tm.specificity,
      include_bits = cfg.tm.include_bits,
      search_patience = cfg.tm_search.patience,
      search_rounds = cfg.tm_search.rounds,
      search_trials = cfg.tm_search.trials,
      search_iterations = cfg.tm_search.iterations,
      search_tolerance = cfg.tm_search.tolerance,
      final_patience = cfg.training.patience,
      final_iterations = cfg.training.iterations,
      search_metric = function (t, enc_info)
        local predicted = t:predict(enc_info.sentences, enc_info.samples)
        local accuracy = eval.encoding_accuracy(predicted, train_solutions, enc_info.samples, train.dims_sup)
        return accuracy.mean_hamming, accuracy
      end,
    }

    if cfg.encoder.mode == "raw" then
      local selection_method = cfg.encoder.raw_selection or "chi2"
      str.printf("\nTraining encoder (mode=raw, selecting %d features by %s)\n", cfg.encoder.raw_max_vocab, selection_method)
      local raw_vocab, raw_scores
      if selection_method == "mi" then
        raw_vocab, raw_scores = train.tokens:bits_top_mi(
          train.codes_sup,
          train.n,
          n_top_v,
          train.dims_sup,
          cfg.encoder.raw_max_vocab)
      else
        raw_vocab, raw_scores = train.tokens:bits_top_chi2(
          train.codes_sup,
          train.n,
          n_top_v,
          train.dims_sup,
          cfg.encoder.raw_max_vocab)
      end
      local n_raw_v = raw_vocab:size()
      str.printf("  Selected %d tokens by %s\n", n_raw_v, selection_method)
      str.printf("  Score range: %.4f - %.4f\n", raw_scores:get(n_raw_v - 1), raw_scores:get(0))
      train.raw_encoder_vocab = raw_vocab
      train.raw_encoder_n_features = n_raw_v
      local train_raw_selected = ivec.create()
      train.tokens:bits_select(raw_vocab, nil, n_top_v, train_raw_selected)
      local validate_raw_selected = ivec.create()
      validate.tokens:bits_select(raw_vocab, nil, n_top_v, validate_raw_selected)
      local test_raw_selected = ivec.create()
      test.tokens:bits_select(raw_vocab, nil, n_top_v, test_raw_selected)
      local train_raw_sentences = train_raw_selected:bits_to_cvec(train.n, n_raw_v, true)
      validate.raw_encoder_sentences = validate_raw_selected:bits_to_cvec(validate.n, n_raw_v, true)
      test.raw_encoder_sentences = test_raw_selected:bits_to_cvec(test.n, n_raw_v, true)
      encoder_args.sentences = train_raw_sentences
      encoder_args.visible = n_raw_v
      print("Building expected adjacency for validate")
      validate.cat_index = inv.create({
        features = cfg.data.n_classes,
        expected_size = validate.n,
      })
      local val_cat_data = ivec.create()
      val_cat_data:copy(validate.solutions)
      val_cat_data:add_scaled(cfg.data.n_classes)
      validate.cat_index:add(val_cat_data, validate.ids)
      val_cat_data:destroy()
      local val_adj_expected_ids, val_adj_expected_offsets, val_adj_expected_neighbors, val_adj_expected_weights =
        graph.adjacency({
          category_index = validate.cat_index,
          category_anchors = cfg.search.eval.anchors,
          random_pairs = cfg.search.eval.pairs,
        })
      encoder_args.search_metric = function (t)
        local val_pred = t:predict(validate.raw_encoder_sentences, validate.n)
        local idx_val = ann.create({ features = train.dims_sup, expected_size = validate.n })
        idx_val:add(val_pred, validate.ids)
        local val_retrieved_ids, val_retrieved_offsets, val_retrieved_neighbors, val_retrieved_weights =
          graph.adjacency({
            weight_index = idx_val,
            seed_ids = val_adj_expected_ids,
            seed_offsets = val_adj_expected_offsets,
            seed_neighbors = val_adj_expected_neighbors,
          })
        local val_stats = eval.score_retrieval({
          retrieved_ids = val_retrieved_ids,
          retrieved_offsets = val_retrieved_offsets,
          retrieved_neighbors = val_retrieved_neighbors,
          retrieved_weights = val_retrieved_weights,
          expected_ids = val_adj_expected_ids,
          expected_offsets = val_adj_expected_offsets,
          expected_neighbors = val_adj_expected_neighbors,
          expected_weights = val_adj_expected_weights,
          ranking = cfg.search.eval.ranking,
          metric = cfg.search.eval.metric,
          n_dims = train.dims_sup,
        })
        idx_val:destroy()
        val_retrieved_ids:destroy()
        val_retrieved_offsets:destroy()
        val_retrieved_neighbors:destroy()
        val_retrieved_weights:destroy()
        return val_stats.score, val_stats
      end
      encoder_args.each = function (_, is_final, metrics, params, epoch, round, trial)
        local d, dd = stopwatch()
        local phase = is_final and "[F]" or str.format("[R%d T%d]", round, trial)
        str.printf("  [E%d]%s %.2f %.2f C=%d L=%d/%d T=%d S=%.0f IB=%d V=%d score=%.4f\n",
          epoch, phase, d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
          params.target, params.specificity, params.include_bits, n_raw_v, metrics.score)
      end
    elseif cfg.encoder.mode == "unsup_to_sup" then
      str.printf("\nTraining encoder (mode=unsup_to_sup, %d unsup dims -> %d sup dims)\n",
        train.dims_unsup, train.dims_sup)
      local train_unsup_cvec = train.index_unsup:get(train.ids)
      train_unsup_cvec:bits_flip_interleave(train.dims_unsup)
      print("  Encoding validation samples to unsup codes via Nyström")
      local validate_unsup_cvec = nystrom_encode(validate.tokens, validate.n)
      validate_unsup_cvec:bits_flip_interleave(train.dims_unsup)
      print("  Encoding test samples to unsup codes via Nyström")
      local test_unsup_cvec = nystrom_encode(test.tokens, test.n)
      test_unsup_cvec:bits_flip_interleave(train.dims_unsup)
      encoder_args.sentences = train_unsup_cvec
      encoder_args.visible = train.dims_unsup
      train.unsup_encoder_sentences = train_unsup_cvec
      validate.unsup_encoder_sentences = validate_unsup_cvec
      test.unsup_encoder_sentences = test_unsup_cvec
      print("Building expected adjacency for validate")
      validate.cat_index = inv.create({
        features = cfg.data.n_classes,
        expected_size = validate.n,
      })
      local val_cat_data = ivec.create()
      val_cat_data:copy(validate.solutions)
      val_cat_data:add_scaled(cfg.data.n_classes)
      validate.cat_index:add(val_cat_data, validate.ids)
      val_cat_data:destroy()
      local val_adj_expected_ids, val_adj_expected_offsets, val_adj_expected_neighbors, val_adj_expected_weights =
        graph.adjacency({
          category_index = validate.cat_index,
          category_anchors = cfg.search.eval.anchors,
          random_pairs = cfg.search.eval.pairs,
        })
      encoder_args.search_metric = function (t)
        local val_pred = t:predict(validate.unsup_encoder_sentences, validate.n)
        local idx_val = ann.create({ features = train.dims_sup, expected_size = validate.n })
        idx_val:add(val_pred, validate.ids)
        local val_retrieved_ids, val_retrieved_offsets, val_retrieved_neighbors, val_retrieved_weights =
          graph.adjacency({
            weight_index = idx_val,
            seed_ids = val_adj_expected_ids,
            seed_offsets = val_adj_expected_offsets,
            seed_neighbors = val_adj_expected_neighbors,
          })
        local val_stats = eval.score_retrieval({
          retrieved_ids = val_retrieved_ids,
          retrieved_offsets = val_retrieved_offsets,
          retrieved_neighbors = val_retrieved_neighbors,
          retrieved_weights = val_retrieved_weights,
          expected_ids = val_adj_expected_ids,
          expected_offsets = val_adj_expected_offsets,
          expected_neighbors = val_adj_expected_neighbors,
          expected_weights = val_adj_expected_weights,
          ranking = cfg.search.eval.ranking,
          metric = cfg.search.eval.metric,
          n_dims = train.dims_sup,
        })
        idx_val:destroy()
        val_retrieved_ids:destroy()
        val_retrieved_offsets:destroy()
        val_retrieved_neighbors:destroy()
        val_retrieved_weights:destroy()
        return val_stats.score, val_stats
      end
      encoder_args.each = function (_, is_final, metrics, params, epoch, round, trial)
        local d, dd = stopwatch()
        local phase = is_final and "[F]" or str.format("[R%d T%d]", round, trial)
        str.printf("  [E%d]%s %.2f %.2f C=%d L=%d/%d T=%d S=%.0f IB=%d unsup_to_sup score=%.4f\n",
          epoch, phase, d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
          params.target, params.specificity, params.include_bits, metrics.score)
      end
    else
      local base_landmarks_index = train.filtered_node_features or train.node_features
      train.encoder_lookup_index = (cfg.lookup_mode == "token")
        and base_landmarks_index
        or train.index_unsup
      train.encoder_codes_index = (cfg.aggregate_mode == "supervised")
        and train.index_sup
        or train.index_unsup
      local train_lookup_features
      if cfg.lookup_mode == "unsupervised" then
        print("\nUsing precomputed unsup codes for train lookup")
        train_lookup_features = train.codes_unsup
        str.printf("  Train unsup codes: %d samples × %d dims\n", train.n, train.dims_unsup)
      else
        train_lookup_features = train.tokens_landmark
      end
      str.printf("\nTraining encoder (lookup=%s, aggregate=%s)\n", cfg.lookup_mode, cfg.aggregate_mode)
      encoder_args.features = train_lookup_features
      encoder_args.landmarks_index = train.encoder_lookup_index
      encoder_args.codes_index = train.encoder_codes_index
      encoder_args.n_landmarks = cfg.landmarks.n_landmarks
      encoder_args.n_thresholds = cfg.landmarks.n_thresholds
      encoder_args.landmark_mode = cfg.landmarks.landmark_mode
      encoder_args.quantile = cfg.landmarks.quantile
      encoder_args.each = function (_, is_final, train_accuracy, params, epoch, round, trial)
        local d, dd = stopwatch()
        local lm = params.n_landmarks or "-"
        local mode = params.landmark_mode or "?"
        local uses_thresh = mode == "frequency" or mode == "weighted"
        local th = uses_thresh and (params.n_thresholds or "-") or "-"
        local mode_short = mode == "frequency" and "freq" or (mode == "weighted" and "wt" or "cat")
        local phase = is_final and "[F]" or str.format("[R%d T%d]", round, trial)
        str.printf("  [E%d]%s %.2f %.2f C=%d L=%d/%d T=%d S=%.0f IB=%d %s LM=%s TH=%s ham=%.2f\n",
          epoch, phase, d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
          params.target, params.specificity, params.include_bits, mode_short, lm, th, train_accuracy.mean_hamming)
      end
    end

    train.encoder, train.encoder_accuracy, train.encoder_params = optimize.encoder(encoder_args)
    -- train.encoder: TM that predicts spectral codes from landmark features

    print("\nFinal encoder performance")
    if cfg.encoder.mode == "raw" or cfg.encoder.mode == "unsup_to_sup" then
      str.printf("  Val | Score: %.4f\n", train.encoder_accuracy.score)
    else
      str.printi("  Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.encoder_accuracy)
    end
    if cfg.encoder.mode == "raw" then
      print("\n  Mode: raw")
      str.printf("    n_features=%d\n", train.raw_encoder_n_features)
    elseif cfg.encoder.mode == "unsup_to_sup" then
      print("\n  Mode: unsup_to_sup")
      str.printf("    input_dims=%d (unsup), output_dims=%d (sup)\n", train.dims_unsup, train.dims_sup)
    else
      print("\n  Best landmark params:")
      str.printf("    mode=%s n_landmarks=%d n_thresholds=%s\n",
        train.encoder_params.landmark_mode,
        train.encoder_params.n_landmarks,
        train.encoder_params.landmark_mode == "frequency" and tostring(train.encoder_params.n_thresholds) or "n/a")
    end
    print("  Best TM params:")
    str.printf("    clauses=%d clause_tolerance=%d clause_maximum=%d target=%d specificity=%.2f\n",
      train.encoder_params.clauses,
      train.encoder_params.clause_tolerance,
      train.encoder_params.clause_maximum,
      train.encoder_params.target,
      train.encoder_params.specificity)

    if cfg.early_exit_after == "encoder" then
      print("\n=== EARLY EXIT after encoder ===")
      return
    end

    local train_encoder_input, test_encoder_input, validate_encoder_input
    if cfg.encoder.mode == "raw" then
      print("\nUsing raw token features for prediction")
      train_encoder_input = encoder_args.sentences
      test_encoder_input = test.raw_encoder_sentences
      validate_encoder_input = validate.raw_encoder_sentences
      str.printf("  train_encoder_input: size=%d bytes\n", train_encoder_input:size())
      str.printf("  test_encoder_input: size=%d bytes\n", test_encoder_input:size())
    elseif cfg.encoder.mode == "unsup_to_sup" then
      print("\nUsing unsupervised codes for prediction")
      train_encoder_input = train.unsup_encoder_sentences
      test_encoder_input = test.unsup_encoder_sentences
      validate_encoder_input = validate.unsup_encoder_sentences
      str.printf("  train_encoder_input: size=%d bytes\n", train_encoder_input:size())
      str.printf("  test_encoder_input: size=%d bytes\n", test_encoder_input:size())
      str.printf("  validate_encoder_input: size=%d bytes\n", validate_encoder_input:size())
    else
      print("\nCreating final landmark encoder for prediction")
      local encode_landmarks, n_latent = hlth.landmark_encoder({
        landmarks_index = train.encoder_lookup_index,
        codes_index = train.encoder_codes_index,
        n_landmarks = train.encoder_params.n_landmarks,
        mode = train.encoder_params.landmark_mode,
        n_thresholds = train.encoder_params.landmark_mode == "frequency" and train.encoder_params.n_thresholds or nil,
        quantile = cfg.landmarks.quantile,
      })
      print("Transforming data for prediction")
      str.printf("  n_latent from encoder: %d\n", n_latent)
      local train_pred_features, test_pred_features, validate_pred_features
      if cfg.lookup_mode == "unsupervised" then
        train_pred_features = train.codes_unsup
        print("  Encoding test/validate samples to unsup codes via Nyström")
        test_pred_features = nystrom_encode(test.tokens, test.n)
        validate_pred_features = nystrom_encode(validate.tokens, validate.n)
        str.printf("  Test unsup codes: %d samples × %d dims\n", test.n, train.dims_unsup)
        str.printf("  Validate unsup codes: %d samples × %d dims\n", validate.n, train.dims_unsup)
      else
        train_pred_features = train.tokens_landmark
        test_pred_features = test.tokens_landmark
        validate_pred_features = validate.tokens_landmark
      end
      train_encoder_input = encode_landmarks(train_pred_features, train.n)
      str.printf("  train_encoder_input before flip: size=%d bytes\n", train_encoder_input:size())
      train_encoder_input:bits_flip_interleave(n_latent)
      str.printf("  train_encoder_input after flip: size=%d bytes\n", train_encoder_input:size())
      test_encoder_input = encode_landmarks(test_pred_features, test.n)
      str.printf("  test_encoder_input before flip: size=%d bytes\n", test_encoder_input:size())
      test_encoder_input:bits_flip_interleave(n_latent)
      str.printf("  test_encoder_input after flip: size=%d bytes\n", test_encoder_input:size())
      validate_encoder_input = encode_landmarks(validate_pred_features, validate.n)
      str.printf("  validate_encoder_input before flip: size=%d bytes\n", validate_encoder_input:size())
      validate_encoder_input:bits_flip_interleave(n_latent)
      str.printf("  validate_encoder_input after flip: size=%d bytes\n", validate_encoder_input:size())
    end

    print("\nPredicting codes with encoder")
    local train_predicted = train.encoder:predict(train_encoder_input, train.n)
    local test_predicted = train.encoder:predict(test_encoder_input, test.n)
    local validate_predicted = train.encoder:predict(validate_encoder_input, validate.n)
    str.printf("  train_predicted: size=%d bytes, expected=%d bytes\n",
      train_predicted:size(), train.n * math.ceil(train.dims_sup / 8))
    str.printf("  test_predicted: size=%d bytes, expected=%d bytes\n",
      test_predicted:size(), test.n * math.ceil(train.dims_sup / 8))
    str.printf("  validate_predicted: size=%d bytes, expected=%d bytes\n",
      validate_predicted:size(), validate.n * math.ceil(train.dims_sup / 8))

    local train_acc = eval.encoding_accuracy(train_predicted, train_solutions, train.n, train.dims_sup)
    str.printf("  train encoding accuracy: %.4f hamming\n", train_acc.mean_hamming)

    print("Indexing predicted codes")
    local idx_train_pred = ann.create({ features = train.dims_sup, expected_size = train.n })
    idx_train_pred:add(train_predicted, train.ids)

    local idx_test_pred = ann.create({ features = train.dims_sup, expected_size = test.n })
    idx_test_pred:add(test_predicted, test.ids)

    print("\nBuilding retrieved adjacency for predicted codes (train)")
    local train_pred_retrieved_ids, train_pred_retrieved_offsets, train_pred_retrieved_neighbors, train_pred_retrieved_weights =
      graph.adjacency({
        weight_index = idx_train_pred,
        seed_ids = train.adj_expected_ids,
        seed_offsets = train.adj_expected_offsets,
        seed_neighbors = train.adj_expected_neighbors,
      })

    print("Building expected adjacency for test")
    test.cat_index = inv.create({
      features = cfg.data.n_classes,
      expected_size = test.n,
    })
    local test_data = ivec.create()
    test_data:copy(test.solutions)
    test_data:add_scaled(cfg.data.n_classes)
    test.cat_index:add(test_data, test.ids)
    test_data:destroy()

    local test_adj_expected_ids, test_adj_expected_offsets, test_adj_expected_neighbors, test_adj_expected_weights =
      graph.adjacency({
        category_index = test.cat_index,
        category_anchors = cfg.search.eval.anchors,
        random_pairs = cfg.search.eval.pairs,
      })

    print("Building retrieved adjacency for predicted codes (test)")
    local test_pred_retrieved_ids, test_pred_retrieved_offsets, test_pred_retrieved_neighbors, test_pred_retrieved_weights =
      graph.adjacency({
        weight_index = idx_test_pred,
        seed_ids = test_adj_expected_ids,
        seed_offsets = test_adj_expected_offsets,
        seed_neighbors = test_adj_expected_neighbors,
      })

    print("\nEvaluating train predicted codes")
    local train_pred_stats = eval.score_retrieval({
      retrieved_ids = train_pred_retrieved_ids,
      retrieved_offsets = train_pred_retrieved_offsets,
      retrieved_neighbors = train_pred_retrieved_neighbors,
      retrieved_weights = train_pred_retrieved_weights,
      expected_ids = train.adj_expected_ids,
      expected_offsets = train.adj_expected_offsets,
      expected_neighbors = train.adj_expected_neighbors,
      expected_weights = train.adj_expected_weights,
      ranking = cfg.search.eval.ranking,
      metric = cfg.search.eval.metric,
      n_dims = train.dims_sup,
    })

    print("Evaluating test predicted codes")
    local test_pred_stats = eval.score_retrieval({
      retrieved_ids = test_pred_retrieved_ids,
      retrieved_offsets = test_pred_retrieved_offsets,
      retrieved_neighbors = test_pred_retrieved_neighbors,
      retrieved_weights = test_pred_retrieved_weights,
      expected_ids = test_adj_expected_ids,
      expected_offsets = test_adj_expected_offsets,
      expected_neighbors = test_adj_expected_neighbors,
      expected_weights = test_adj_expected_weights,
      ranking = cfg.search.eval.ranking,
      metric = cfg.search.eval.metric,
      n_dims = train.dims_sup,
    })

    str.printf("\nEncoder Retrieval Results\n")
    str.printf("  Original | Score: %.4f\n", train.retrieval_stats.score)
    str.printf("  Train    | Score: %.4f\n", train_pred_stats.score)
    str.printf("  Test     | Score: %.4f\n", test_pred_stats.score)

    print("\nSpot-check: Test k-NN Classification (test-to-test)")
    do
      local max_k = 10
      local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
        knn_index = idx_test_pred,
        knn = max_k + 1,
        bridge = "none",
      })

      local id_to_idx = {}
      for i = 0, knn_ids:size() - 1 do
        id_to_idx[knn_ids:get(i)] = i
      end

      local k_values = { 1, 3, 5, 10 }
      for _, k in ipairs(k_values) do
        local correct = 0
        local total_precision = 0
        local counted = 0
        for i = 0, test.n - 1 do
          local query_id = test.ids:get(i)
          local idx = id_to_idx[query_id]
          if idx then
            local query_label = test.solutions:get(i)
            local offset_start = knn_offsets:get(idx)
            local offset_end = knn_offsets:get(idx + 1)
            local votes = {}
            local same_class = 0
            local neighbor_count = 0
            for j = offset_start, offset_end - 1 do
              local neighbor_idx = knn_neighbors:get(j)
              local neighbor_id = knn_ids:get(neighbor_idx)
              local test_id_start = train.n + validate.n
              if neighbor_id ~= query_id and neighbor_id >= test_id_start then
                local neighbor_test_idx = neighbor_id - test_id_start
                if neighbor_test_idx < test.n then
                  neighbor_count = neighbor_count + 1
                  if neighbor_count <= k then
                    local neighbor_label = test.solutions:get(neighbor_test_idx)
                    votes[neighbor_label] = (votes[neighbor_label] or 0) + 1
                    if neighbor_label == query_label then
                      same_class = same_class + 1
                    end
                  end
                end
              end
            end
            if neighbor_count > 0 then
              local best_label, best_count = -1, 0
              for lbl, cnt in pairs(votes) do
                if cnt > best_count then
                  best_label, best_count = lbl, cnt
                end
              end
              if best_label == query_label then
                correct = correct + 1
              end
              total_precision = total_precision + same_class / k
              counted = counted + 1
            end
          end
        end
        if counted > 0 then
          str.printf("  k=%2d: Accuracy = %.2f%% | Precision = %.2f%%\n",
            k, 100 * correct / counted, 100 * total_precision / counted)
        end
      end
      knn_ids:destroy()
      knn_offsets:destroy()
      knn_neighbors:destroy()
    end

    print("\nPer-class test k=5 accuracy (test-to-test)")
    do
      local k = 5
      local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
        knn_index = idx_test_pred,
        knn = k + 1,
        bridge = "none",
      })
      local id_to_idx = {}
      for i = 0, knn_ids:size() - 1 do
        id_to_idx[knn_ids:get(i)] = i
      end
      local class_correct = {}
      local class_total = {}
      for c = 0, cfg.data.n_classes - 1 do
        class_correct[c] = 0
        class_total[c] = 0
      end
      for i = 0, test.n - 1 do
        local query_id = test.ids:get(i)
        local idx = id_to_idx[query_id]
        if idx then
          local query_label = test.solutions:get(i)
          local offset_start = knn_offsets:get(idx)
          local offset_end = knn_offsets:get(idx + 1)
          local votes = {}
          local neighbor_count = 0
          local test_id_start = train.n + validate.n
          for j = offset_start, offset_end - 1 do
            local neighbor_idx = knn_neighbors:get(j)
            local neighbor_id = knn_ids:get(neighbor_idx)
            if neighbor_id ~= query_id and neighbor_id >= test_id_start then
              local neighbor_test_idx = neighbor_id - test_id_start
              if neighbor_test_idx < test.n then
                neighbor_count = neighbor_count + 1
                if neighbor_count <= k then
                  local neighbor_label = test.solutions:get(neighbor_test_idx)
                  votes[neighbor_label] = (votes[neighbor_label] or 0) + 1
                end
              end
            end
          end
          if neighbor_count > 0 then
            class_total[query_label] = class_total[query_label] + 1
            local best_label, best_count = -1, 0
            for lbl, cnt in pairs(votes) do
              if cnt > best_count then
                best_label, best_count = lbl, cnt
              end
            end
            if best_label == query_label then
              class_correct[query_label] = class_correct[query_label] + 1
            end
          end
        end
      end
      local sorted_classes = {}
      for c = 0, cfg.data.n_classes - 1 do
        local acc = class_total[c] > 0 and (class_correct[c] / class_total[c]) or 0
        table.insert(sorted_classes, { c = c, acc = acc, n = class_total[c] })
      end
      table.sort(sorted_classes, function(a, b) return a.acc < b.acc end)
      for _, item in ipairs(sorted_classes) do
        local cat = test.categories[item.c + 1]
        str.printf("  %5.1f%% (%4d) %s\n", 100 * item.acc, item.n, cat)
      end
      knn_ids:destroy()
      knn_offsets:destroy()
      knn_neighbors:destroy()
    end

    print("\nSpot-check: Test Example Retrievals (5 random samples, test-to-test)")
    do
      local k = 5
      local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
        knn_index = idx_test_pred,
        knn = k + 1,
        bridge = "none",
      })
      local id_to_idx = {}
      for i = 0, knn_ids:size() - 1 do
        id_to_idx[knn_ids:get(i)] = i
      end
      local samples = { 0, 100, 500, 1000, 3000 }
      for _, test_idx in ipairs(samples) do
        if test_idx < test.n then
          local query_id = test.ids:get(test_idx)
          local idx = id_to_idx[query_id]
          if idx then
            local query_label = test.solutions:get(test_idx)
            local query_cat = test.categories[query_label + 1]
            local offset_start = knn_offsets:get(idx)
            local offset_end = knn_offsets:get(idx + 1)
            str.printf("  Test Sample %d [%s]:\n", test_idx, query_cat)
            local shown = 0
            local test_id_start = train.n + validate.n
            for j = offset_start, offset_end - 1 do
              local neighbor_idx = knn_neighbors:get(j)
              local neighbor_id = knn_ids:get(neighbor_idx)
              if neighbor_id ~= query_id and neighbor_id >= test_id_start then
                local neighbor_test_idx = neighbor_id - test_id_start
                if neighbor_test_idx < test.n and shown < k then
                  local neighbor_label = test.solutions:get(neighbor_test_idx)
                  local neighbor_cat = test.categories[neighbor_label + 1]
                  local match = neighbor_cat == query_cat and "+" or "-"
                  str.printf("    %d. [%s] %s\n", shown + 1, match, neighbor_cat)
                  shown = shown + 1
                end
              end
            end
          end
        end
      end
      knn_ids:destroy()
      knn_offsets:destroy()
      knn_neighbors:destroy()
    end

    idx_train_pred:destroy()
    idx_test_pred:destroy()

    if cfg.early_exit_after == "retrieval" then
      print("\n=== EARLY EXIT after retrieval ===")
      return
    end

    if cfg.classifier.enabled then
      print("\n=== Classification on Predicted Codes ===")
      str.printf("  Train: %d samples, Test: %d samples\n", train.n, test.n)
      str.printf("  Features: %d bits, Classes: %d\n", train.dims_sup, cfg.data.n_classes)

      print("Flip-interleaving predicted codes for classifier")
      train_predicted:bits_flip_interleave(train.dims_sup)
      validate_predicted:bits_flip_interleave(train.dims_sup)
      test_predicted:bits_flip_interleave(train.dims_sup)

      print("\nOptimizing classifier on train predicted codes")
      local classifier = optimize.classifier({
        features = train.dims_sup,
        classes = cfg.data.n_classes,
        clauses = cfg.classifier.clauses,
        clause_tolerance = cfg.classifier.clause_tolerance,
        clause_maximum = cfg.classifier.clause_maximum,
        target = cfg.classifier.target,
        negative = cfg.classifier.negative,
        specificity = cfg.classifier.specificity,
        include_bits = cfg.classifier.include_bits,
        samples = train.n,
        problems = train_predicted,
        solutions = train.solutions,
        search_patience = cfg.classifier.search_patience,
        search_rounds = cfg.classifier.search_rounds,
        search_trials = cfg.classifier.search_trials,
        search_iterations = cfg.classifier.search_iterations,
        final_iterations = cfg.classifier.final_iterations,
        search_metric = function (t)
          local predicted = t:predict(validate_predicted, validate.n)
          local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.data.n_classes)
          return accuracy.f1, accuracy
        end,
        each = function (t, is_final, val_accuracy, params, epoch, round, trial)
          local test_pred = t:predict(test_predicted, test.n)
          local test_accuracy = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.data.n_classes)
          local d, dd = stopwatch()
          local phase = is_final and "[F]" or str.format("[R%d T%d]", round, trial)
          str.printf("  [E%d]%s %.2f %.2f C=%d L=%d/%d T=%d S=%.0f IB=%d f1=(%.2f,%.2f)\n",
            epoch, phase, d, dd, params.clauses, params.clause_tolerance, params.clause_maximum,
            params.target, params.specificity, params.include_bits, val_accuracy.f1, test_accuracy.f1)
        end,
      })

      print("\nFinal Classification Results")
      local train_class_pred = classifier:predict(train_predicted, train.n)
      local val_class_pred = classifier:predict(validate_predicted, validate.n)
      local test_class_pred = classifier:predict(test_predicted, test.n)
      local train_class_stats = eval.class_accuracy(train_class_pred, train.solutions, train.n, cfg.data.n_classes)
      local val_class_stats = eval.class_accuracy(val_class_pred, validate.solutions, validate.n, cfg.data.n_classes)
      local test_class_stats = eval.class_accuracy(test_class_pred, test.solutions, test.n, cfg.data.n_classes)

      str.printf("  Train F1: %.2f (P=%.2f R=%.2f)\n",
        train_class_stats.f1, train_class_stats.precision, train_class_stats.recall)
      str.printf("  Val   F1: %.2f (P=%.2f R=%.2f)\n",
        val_class_stats.f1, val_class_stats.precision, val_class_stats.recall)
      str.printf("  Test  F1: %.2f (P=%.2f R=%.2f)\n",
        test_class_stats.f1, test_class_stats.precision, test_class_stats.recall)

      print("\nPer-class Test Accuracy (sorted by difficulty):")
      local class_order = {}
      for c = 0, cfg.data.n_classes - 1 do
        table.insert(class_order, c)
      end
      table.sort(class_order, function (a, b)
        return test_class_stats.classes[a + 1].f1 < test_class_stats.classes[b + 1].f1
      end)
      for _, c in ipairs(class_order) do
        local ts = test_class_stats.classes[c + 1]
        local cat = test.categories[c + 1] or ("class_" .. c)
        str.printf("  %-28s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
      end

      print("=== END Classification ===\n")
    end
  end

  collectgarbage("collect")

end)
