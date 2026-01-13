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
local tokenizer = require("santoku.tokenizer")

local cfg; cfg = {
  data = {
    max_per_class = nil,
    n_classes = 20,
  },
  landmarks = {
    n_landmarks = 4, -- { def = 4, min = 4, max = 32, log = true, int = true },
    n_thresholds = 7, -- { def = 8, min = 4, max = 12, log = true, int = true },
    landmark_mode = "frequency",
  },
  tokenizer = {
    max_len = 20,
    min_len = 2,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 0,
    cgrams_max = 0,
    skips = 1,
    negations = 0,
  },
  feature_selection = {
    min_df = -2,
    max_df = 1.0,
    max_vocab = 8192,
  },
  ann = {
    bucket_size = nil,
  },
  cluster = {
    enabled = true,
    verbose = true,
    knn = 64,
  },
  encoder = {
    enabled = true,
    verbose = true,
  },
  tm = {
    clauses = 8, -- { def = 8, min = 4, max = 16, log = true, int = true },
    clause_tolerance = 7, -- { def = 8, min = 4, max = 16, log = true, int = true },
    clause_maximum = 22, -- { def = 16, min = 8, max = 32, log = true, int = true },
    target = 8, -- { def = 4, min = 2, max = 8, log = true, int = true },
    specificity = 12, -- { def = 10, min = 2, max = 1000, log = true, int = true },
  },
  tm_search = {
    patience = 3,
    rounds = 1, -- 6,
    trials = 1, -- 6,
    tolerance = 1e-4,
    iterations = 8,
  },
  training = {
    patience = 20,
    iterations = 100,
  },
  search = {
    rounds = 1, -- 4,
    patience = 3,
    adjacency_samples = 1, -- 8,
    spectral_samples = 1, -- 4,
    eval_samples = 1, -- 4,
    adjacency = {
      knn = 29, -- { def = 25, min = 20, max = 30, int = true },
      knn_alpha = 14, -- { def = 12, min = 8, max = 14, int = true },
      weight_decay = 4.04, -- { def = 2, min = -2, max = 6 },
      knn_mutual = false, -- { true, false },
      knn_mode = "cknn",
      knn_cache = 128,
      bridge = "mst",
    },
    spectral = {
      laplacian = "unnormalized",
      n_dims = 32,
      eps = 1e-8,
      threshold = {
        method = "itq", -- { "itq", "median", "otsu" },
        itq_iterations = 500,
        itq_tolerance = 1e-8,
        otsu_metric = "variance",
        otsu_minimize = false,
        otsu_n_bins = 32,
      },
    },
    eval = {
      knn = 32,
      anchors = 16,
      pairs = 64,
      ranking = "ndcg",
      metric = "min",
      elbow = "first_gap", -- { "first_gap", "plateau", "lmethod" },
      elbow_alpha = 4.4, -- { def = 5, min = 2, max = 15 },
      target = "combined",
      max_consecutive_zeros = 3,
    },
    cluster_eval = {
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
  local train, test = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max_per_class)

  str.printf("  Train: %6d (%d categories)\n", train.n, train.n_labels)
  str.printf("  Test:  %6d (%d categories)\n", test.n, test.n_labels)

  print("\nTraining tokenizer")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  str.printf("  Vocabulary: %d tokens\n", n_tokens)

  print("\nTokenizing train")
  train.tokens = tok:tokenize(train.problems)

  print("\nFeature selection (chi2 on class labels)")
  train.solutions:add_scaled(cfg.data.n_classes)
  local top_v = train.tokens:bits_top_chi2(
    train.solutions, train.n, n_tokens, cfg.data.n_classes, cfg.feature_selection.max_vocab)
  train.solutions:add_scaled(-cfg.data.n_classes)
  local n_top_v = top_v:size()
  str.printf("  Chi2 selected: %d features\n", n_top_v)

  print("\nRe-tokenizing with selected vocabulary")
  tok:restrict(top_v)
  train.tokens = tok:tokenize(train.problems)
  test.tokens = tok:tokenize(test.problems)
  -- train.tokens, test.tokens: tokenized with chi2-selected vocab

  print("\nComputing IDF scores and reordering by IDF")
  local idf_sorted, idf_weights = train.tokens:bits_top_df(train.n, n_top_v)
  local n_idf = idf_sorted:size()
  str.printf("  IDF reordered: %d features\n", n_idf)
  str.printf("  IDF range: %.3f - %.3f\n", idf_weights:get(n_idf - 1), idf_weights:get(0))

  tok:restrict(idf_sorted)
  n_top_v = n_idf
  train.tokens = tok:tokenize(train.problems)
  test.tokens = tok:tokenize(test.problems)

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.ids:add(train.n)
  -- train.ids: 0 to train.n-1, test.ids: train.n to train.n+test.n-1

  print("\nBuilding feature similarity graph (weight_index with categories + tokens)")
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
    train.index_graph = inv.create({
      features = graph_weights,
      ranks = graph_ranks,
      n_ranks = 2,
    })
    train.index_graph:add(graph_problems, train.ids)
    -- train.index_graph: weight index for spectral (categories rank 0, tokens rank 1)
    str.printf("  Weight index: %d features (%d tokens + %d categories), 2 ranks\n",
      n_top_v + cfg.data.n_classes, n_top_v, cfg.data.n_classes)
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

  print("\nBuilding category ground truth")
  train.cat_index, train.ground_truth = build_category_ground_truth(train.ids, train.solutions)

  print("\nOptimizing spectral pipeline")
  local model, _, spectral_best_metrics = optimize.spectral({
    index = train.index_graph,
    knn_index = train.node_features_graph,
    bucket_size = cfg.ann.bucket_size,
    rounds = cfg.search.rounds,
    patience = cfg.search.patience,
    adjacency_samples = cfg.search.adjacency_samples,
    spectral_samples = cfg.search.spectral_samples,
    eval_samples = cfg.search.eval_samples,
    adjacency = cfg.search.adjacency,
    spectral = cfg.search.spectral,
    eval = cfg.search.eval,
    expected_ids = train.ground_truth.retrieval.ids,
    expected_offsets = train.ground_truth.retrieval.offsets,
    expected_neighbors = train.ground_truth.retrieval.neighbors,
    expected_weights = train.ground_truth.retrieval.weights,
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
    each = cfg.search.verbose and function (info)
      if info.event == "round_start" then
        str.printf("\n=== Round %d/%d ===\n", info.round, info.rounds)
      elseif info.event == "round_end" then
        str.printf("\n--- Round %d complete: best=%.4f (global=%.4f) ---\n",
          info.round, info.round_best_score, info.global_best_score)
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
        elseif type(p.threshold) == "string" then
          thresh_str = p.threshold
        end
        str.printf("    lap=%s dims=%d thresh=%s\n", p.laplacian, p.n_dims, thresh_str)
      elseif info.event == "eval" then
        local e = info.params.eval
        local m = info.metrics
        local alpha_str = e.elbow_alpha and str.format("%.1f", e.elbow_alpha) or "-"
        str.printf("    elbow=%s(%s) knn=%d score=%.4f quality=%.4f combined=%.4f\n",
          e.elbow, alpha_str, e.knn, m.score, m.quality, m.combined)
      end
    end or nil,
  })

  train.codes_spectral = model.codes
  train.ids_spectral = model.ids
  train.dims_spectral = model.dims
  train.index_spectral = model.index
  -- train.codes_spectral: binary spectral codes from optimize.spectral
  -- train.index_spectral: ANN on spectral codes for retrieval
  train.retrieval_stats = spectral_best_metrics

  print("\nBuilding landmark index (IDF-weighted chi2-selected tokens)")
  train.node_features = inv.create({
    features = idf_weights,
  })
  train.node_features:add(train.tokens, train.ids)
  -- train.node_features: landmark index (chi2-selected tokens, IDF-weighted)
  str.printf("  Landmark index: %d tokens\n", n_top_v)

  train.tokens_landmark = train.tokens
  -- train.tokens_landmark: same as train.tokens

  train.adj_expected_ids = train.ground_truth.retrieval.ids
  train.adj_expected_offsets = train.ground_truth.retrieval.offsets
  train.adj_expected_neighbors = train.ground_truth.retrieval.neighbors
  train.adj_expected_weights = train.ground_truth.retrieval.weights

  print("\nSpot-check: k-NN Classification & Precision (spectral codes)")
  do
    local max_k = 10
    local knn_ids, knn_offsets, knn_neighbors = graph.adjacency({
      knn_index = train.index_spectral,
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
      knn_index = train.index_spectral,
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

  if cfg.cluster.enabled then
    print("\nSetting up clustering adjacency")
    local adj_cluster_ids, adj_cluster_offsets, adj_cluster_neighbors =
      graph.adjacency({
        knn_index = train.index_spectral,
        knn = cfg.cluster.knn,
        each = function (ns, cs, es, stg)
          local d, dd = stopwatch()
          str.printf("  Time: %6.2f %6.2f | Stage: %-10s  Nodes: %5d  Components: %5s  Edges: %5s\n",
            d, dd, stg, ns, cs, es)
        end
      })

    print("\nClustering")
    train.codes_clusters = eval.cluster({
      codes = train.index_spectral:get(adj_cluster_ids),
      n_dims = train.dims_spectral,
      ids = adj_cluster_ids,
      offsets = adj_cluster_offsets,
      neighbors = adj_cluster_neighbors,
    })

    train.cluster_stats = eval.score_clustering({
      ids = train.codes_clusters.ids,
      offsets = train.codes_clusters.offsets,
      merges = train.codes_clusters.merges,
      expected_ids = train.ground_truth.retrieval.ids,
      expected_offsets = train.ground_truth.retrieval.offsets,
      expected_neighbors = train.ground_truth.retrieval.neighbors,
      expected_weights = train.ground_truth.retrieval.weights,
      metric = "min",
      elbow = cfg.search.cluster_eval.elbow,
      elbow_target = cfg.search.cluster_eval.elbow_target,
      elbow_alpha = cfg.search.cluster_eval.elbow_alpha,
    })

    if cfg.cluster.verbose then
      for step = 0, train.cluster_stats.n_steps do
        local d, dd = stopwatch()
        str.printf("  Time: %6.2f %6.2f | Step: %2d | Quality: %.2f | Clusters: %d\n",
          d, dd, step, train.cluster_stats.quality_curve:get(step),
          train.cluster_stats.n_clusters_curve:get(step))
      end
    end

    str.printf("\nClustering Metrics\n")
    str.printf("  Best Step: %d (dendrogram cut point)\n", train.cluster_stats.best_step)
    str.printf("  Quality: %.4f (rank-biserial correlation: cluster membership vs expected similarity)\n",
      train.cluster_stats.quality)
    str.printf("  Clusters: %d (number of clusters at best step, target is %d categories)\n",
      train.cluster_stats.n_clusters, cfg.data.n_classes)
    str.printf("  Interpretation: Quality near 1.0 means cluster membership strongly correlates with\n")
    str.printf("                  category-based expected neighbors; cluster count near %d is ideal.\n", cfg.data.n_classes)
  end

  collectgarbage("collect")

  if cfg.encoder.enabled then

    print("\n=== DIAGNOSTIC: Data Alignment Check ===")
    str.printf("  train.ids: size=%d, first=%d, last=%d\n",
      train.ids:size(), train.ids:get(0), train.ids:get(train.ids:size()-1))
    str.printf("  train.ids_spectral: size=%d, first=%d, last=%d\n",
      train.ids_spectral:size(), train.ids_spectral:get(0), train.ids_spectral:get(train.ids_spectral:size()-1))
    str.printf("  test.ids: size=%d, first=%d, last=%d\n",
      test.ids:size(), test.ids:get(0), test.ids:get(test.ids:size()-1))
    str.printf("  train.train_raw: size=%d bytes, n_top_v=%d, expected=%d bytes\n",
      train.train_raw:size(), n_top_v, train.n * math.ceil(n_top_v / 8))

    local ids_match = true
    local first_mismatch = -1
    for i = 0, math.min(train.ids:size(), train.ids_spectral:size()) - 1 do
      if train.ids:get(i) ~= train.ids_spectral:get(i) then
        ids_match = false
        first_mismatch = i
        break
      end
    end
    if ids_match then
      print("  train.ids == train.ids_spectral: YES (element-wise equal)")
    else
      str.printf("  train.ids == train.ids_spectral: NO (first mismatch at index %d)\n", first_mismatch)
      str.printf("    train.ids[%d] = %d, train.ids_spectral[%d] = %d\n",
        first_mismatch, train.ids:get(first_mismatch),
        first_mismatch, train.ids_spectral:get(first_mismatch))
    end

    local train_solutions = train.index_spectral:get(train.ids)
    local test_raw = test.tokens:bits_to_cvec(test.n, n_top_v)

    str.printf("  test_raw: size=%d bytes, expected=%d bytes\n",
      test_raw:size(), test.n * math.ceil(n_top_v / 8))
    str.printf("  train_solutions: size=%d bytes\n", train_solutions:size())
    print("=== END DIAGNOSTIC ===\n")

    print("\nTraining encoder (with landmark search)")
    train.encoder, train.encoder_accuracy, train.encoder_params = optimize.encoder({
      hidden = train.dims_spectral,
      features = train.tokens_landmark,
      landmarks_index = train.node_features,
      codes_index = train.index_spectral,
      codes = train_solutions,
      samples = train.n,
      n_landmarks = cfg.landmarks.n_landmarks,
      n_thresholds = cfg.landmarks.n_thresholds,
      landmark_mode = cfg.landmarks.landmark_mode,
      clauses = cfg.tm.clauses,
      clause_tolerance = cfg.tm.clause_tolerance,
      clause_maximum = cfg.tm.clause_maximum,
      target = cfg.tm.target,
      specificity = cfg.tm.specificity,
      search_patience = cfg.tm_search.patience,
      search_rounds = cfg.tm_search.rounds,
      search_trials = cfg.tm_search.trials,
      search_iterations = cfg.tm_search.iterations,
      search_tolerance = cfg.tm_search.tolerance,
      final_patience = cfg.training.patience,
      final_iterations = cfg.training.iterations,
      search_metric = function (t, enc_info)
        local predicted = t:predict(enc_info.sentences, enc_info.samples)
        local accuracy = eval.encoding_accuracy(predicted, train_solutions, enc_info.samples, train.dims_spectral)
        return accuracy.mean_hamming, accuracy
      end,
      each = function (_, is_final, train_accuracy, params, epoch, round, trial)
        local d, dd = stopwatch()
        local lm = params.n_landmarks or "-"
        local mode = params.landmark_mode or "?"
        local th = mode == "frequency" and (params.n_thresholds or "-") or "-"
        local mode_short = mode == "frequency" and "freq" or "cat"
        if is_final then
          str.printf("  Time %3.2f %3.2f  Finalizing  C=%d LF=%d L=%d T=%d S=%.2f  %s LM=%s TH=%s  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, mode_short, lm, th, epoch)
        else
          str.printf("  Time %3.2f %3.2f  Exploring  C=%d LF=%d L=%d T=%d S=%.2f  %s LM=%s TH=%s  R=%d T=%d  Epoch  %d\n",
            d, dd, params.clauses, params.clause_tolerance, params.clause_maximum, params.target, params.specificity, mode_short, lm, th, round, trial, epoch)
        end
        str.printi("    Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train_accuracy)
      end,
    })
    -- train.encoder: TM that predicts spectral codes from landmark features

    print("\nFinal encoder performance")
    str.printi("  Train | Ham: %.2f#(mean_hamming) | BER: %.2f#(ber_min) %.2f#(ber_max) %.2f#(ber_std)", train.encoder_accuracy)
    print("\n  Best landmark params:")
    str.printf("    mode=%s n_landmarks=%d n_thresholds=%s\n",
      train.encoder_params.landmark_mode,
      train.encoder_params.n_landmarks,
      train.encoder_params.landmark_mode == "frequency" and tostring(train.encoder_params.n_thresholds) or "n/a")
    print("  Best TM params:")
    str.printf("    clauses=%d clause_tolerance=%d clause_maximum=%d target=%d specificity=%.2f\n",
      train.encoder_params.clauses,
      train.encoder_params.clause_tolerance,
      train.encoder_params.clause_maximum,
      train.encoder_params.target,
      train.encoder_params.specificity)

    print("\nCreating final landmark encoder for prediction")
    local encode_landmarks, n_latent = hlth.landmark_encoder({
      landmarks_index = train.node_features,
      codes_index = train.index_spectral,
      n_landmarks = train.encoder_params.n_landmarks,
      mode = train.encoder_params.landmark_mode,
      n_thresholds = train.encoder_params.landmark_mode == "frequency" and train.encoder_params.n_thresholds or nil,
    })

    print("Transforming data for prediction")
    str.printf("  n_latent from encoder: %d\n", n_latent)
    local train_landmark_feats = encode_landmarks(train.tokens_landmark, train.n)
    str.printf("  train_landmark_feats before flip: size=%d bytes\n", train_landmark_feats:size())
    train_landmark_feats:bits_flip_interleave(n_latent)
    str.printf("  train_landmark_feats after flip: size=%d bytes\n", train_landmark_feats:size())
    local test_landmark_feats = encode_landmarks(test.tokens, test.n)
    str.printf("  test_landmark_feats before flip: size=%d bytes\n", test_landmark_feats:size())
    test_landmark_feats:bits_flip_interleave(n_latent)
    str.printf("  test_landmark_feats after flip: size=%d bytes\n", test_landmark_feats:size())

    print("\nPredicting codes with encoder")
    local train_predicted = train.encoder:predict(train_landmark_feats, train.n)
    local test_predicted = train.encoder:predict(test_landmark_feats, test.n)
    str.printf("  train_predicted: size=%d bytes, expected=%d bytes\n",
      train_predicted:size(), train.n * math.ceil(train.dims_spectral / 8))
    str.printf("  test_predicted: size=%d bytes, expected=%d bytes\n",
      test_predicted:size(), test.n * math.ceil(train.dims_spectral / 8))

    local train_acc = eval.encoding_accuracy(train_predicted, train_solutions, train.n, train.dims_spectral)
    str.printf("  train encoding accuracy: %.4f hamming\n", train_acc.mean_hamming)

    print("Indexing predicted codes")
    local idx_train_pred = ann.create({ features = train.dims_spectral, expected_size = train.n })
    idx_train_pred:add(train_predicted, train.ids)

    local idx_test_pred = ann.create({ features = train.dims_spectral, expected_size = test.n })
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
      ranking = "ndcg",
      metric = "min",
      elbow = "lmethod",
      n_dims = train.dims_spectral,
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
      ranking = "ndcg",
      metric = "min",
      elbow = "lmethod",
      n_dims = train.dims_spectral,
    })

    str.printf("\nEncoder Retrieval Results\n")
    str.printf("  Original | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      train.retrieval_stats.score, train.retrieval_stats.quality,
      train.retrieval_stats.combined)
    str.printf("  Train    | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      train_pred_stats.score, train_pred_stats.quality,
      train_pred_stats.combined)
    str.printf("  Test     | Score: %.4f | Quality: %.4f | Combined: %.4f\n",
      test_pred_stats.score, test_pred_stats.quality,
      test_pred_stats.combined)

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
              if neighbor_id ~= query_id and neighbor_id >= train.n then
                local neighbor_test_idx = neighbor_id - train.n
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
            for j = offset_start, offset_end - 1 do
              local neighbor_idx = knn_neighbors:get(j)
              local neighbor_id = knn_ids:get(neighbor_idx)
              if neighbor_id ~= query_id and neighbor_id >= train.n then
                local neighbor_test_idx = neighbor_id - train.n
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
  end

  collectgarbage("collect")

end)
