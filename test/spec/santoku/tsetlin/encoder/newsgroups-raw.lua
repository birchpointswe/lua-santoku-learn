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
local optimize = require("santoku.tsetlin.optimize")
local tokenizer = require("santoku.tokenizer")

local cfg; cfg = {
  data = {
    max_per_class = nil,
    n_classes = 20,
    tvr = 0.1,
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
  encoder = {
    individualized = true,
    max_vocab = 12288,
    selection = "chi2",
  },
  tm = {
    clauses = { def = 24, min = 8, max = 256, round = 8 },
    clause_tolerance = { def = 17, min = 16, max = 128, int = true },
    clause_maximum = { def = 85, min = 16, max = 128, int = true },
    target = { def = 17, min = 16, max = 128, int = true },
    specificity = { def = 735, min = 400, max = 4000 },
    include_bits = { def = 3, min = 1, max = 4, int = true },
  },
  tm_search = {
    patience = 2,
    rounds = 6,
    trials = 10,
    iterations = 20,
  },
  training = {
    patience = 20,
    iterations = 400,
  },
  classifier = {
    enabled = true,
    clauses = { def = 8, min = 8, max = 32, round = 8 },
    clause_tolerance = { def = 35, min = 16, max = 128, int = true },
    clause_maximum = { def = 44, min = 16, max = 128, int = true },
    target = { def = 58, min = 16, max = 128, int = true },
    specificity = { def = 858, min = 400, max = 4000 },
    include_bits = { def = 2, min = 1, max = 4, int = true },
    negative = 0.5,
    search_patience = 2,
    search_rounds = 4,
    search_trials = 10,
    search_iterations = 20,
    final_iterations = 100,
  },
  search = {
    rounds = 4,
    patience = 3,
    adjacency_samples = 8,
    spectral_samples = 4,
    select_samples = 8,
    eval_samples = 4,
    adjacency = {
      knn = { def = 23, min = 20, max = 35, int = true },
      knn_alpha = { def = 14, min = 10, max = 16, int = true },
      weight_decay = { def = 2.71, min = 2, max = 6 },
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
    },
    verbose = true,
  },
}

test("newsgroups-raw", function()

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

  print("\nBuilding knn index (IDF-weighted tokens only)")
  train.node_features_graph = inv.create({
    features = idf_weights,
  })
  train.node_features_graph:add(train.tokens, train.ids)
  str.printf("  KNN index: %d tokens with IDF weights\n", n_top_v)

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

    return cat_index, {
      retrieval = {
        ids = adj_expected_ids,
        offsets = adj_expected_offsets,
        neighbors = adj_expected_neighbors,
        weights = adj_expected_weights,
      },
    }
  end

  print("\nBuilding category ground truth (for supervised spectral)")
  train.cat_index, train.ground_truth_sup = build_category_ground_truth(train.ids, train.solutions)

  print("\nOptimizing supervised spectral pipeline")
  local model_sup, spectral_best_params_sup, spectral_best_metrics_sup = optimize.spectral({
    index = train.index_graph_sup,
    knn_index = train.node_features_graph,
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
    }
  end

  local spectral_ground_truth = build_token_ground_truth(train.index_sup, cfg.search.eval.knn)
  train.adj_expected_ids = spectral_ground_truth.retrieval.ids
  train.adj_expected_offsets = spectral_ground_truth.retrieval.offsets
  train.adj_expected_neighbors = spectral_ground_truth.retrieval.neighbors
  train.adj_expected_weights = spectral_ground_truth.retrieval.weights

  collectgarbage("collect")

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

  local selection_method = cfg.encoder.selection
  local use_ind = cfg.encoder.individualized
  str.printf("\nTraining encoder (selecting %d features by %s, individualized=%s)\n",
    cfg.encoder.max_vocab, selection_method, tostring(use_ind))

  if use_ind then
    local ids_union, feat_offsets, feat_ids = train.tokens:bits_top_chi2_ind(
      train.codes_sup, train.n, n_top_v, train.dims_sup,
      cfg.encoder.max_vocab)
    local union_size = ids_union:size()
    local total_features = feat_offsets:get(train.dims_sup)
    str.printf("  Per-dimension chi2: union=%d total=%d (%.1fx expansion)\n",
      union_size, total_features, total_features / union_size)
    tok:restrict(ids_union)
    local function to_ind_bitmap (split)
      local toks = tok:tokenize(split.problems)
      local ind, ind_off = toks:bits_individualize(feat_offsets, feat_ids, union_size)
      local bitmap, dim_off = ind:bits_to_cvec_ind(ind_off, feat_offsets, split.n, true)
      toks:destroy()
      ind:destroy()
      ind_off:destroy()
      return bitmap, dim_off
    end
    local train_bitmap, train_dim_off = to_ind_bitmap(train)
    local val_bitmap, val_dim_off = to_ind_bitmap(validate)
    local test_bitmap, test_dim_off = to_ind_bitmap(test)
    encoder_args.sentences = train_bitmap
    encoder_args.visible = union_size
    encoder_args.individualized = true
    encoder_args.feat_offsets = feat_offsets
    encoder_args.dim_offsets = train_dim_off
    validate.raw_encoder_sentences = val_bitmap
    validate.raw_encoder_dim_offsets = val_dim_off
    test.raw_encoder_sentences = test_bitmap
    test.raw_encoder_dim_offsets = test_dim_off
    train.raw_encoder_feat_offsets = feat_offsets
    train.raw_encoder_n_features = union_size
  else
    local raw_vocab, raw_scores
    if selection_method == "mi" then
      raw_vocab, raw_scores = train.tokens:bits_top_mi(
        train.codes_sup,
        train.n,
        n_top_v,
        train.dims_sup,
        cfg.encoder.max_vocab)
    else
      raw_vocab, raw_scores = train.tokens:bits_top_chi2(
        train.codes_sup,
        train.n,
        n_top_v,
        train.dims_sup,
        cfg.encoder.max_vocab)
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
  end
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
  encoder_args.search_metric = function (t, _)
    local val_pred
    if validate.raw_encoder_dim_offsets then
      val_pred = t:predict(validate.raw_encoder_sentences, validate.raw_encoder_dim_offsets, validate.n)
    else
      val_pred = t:predict(validate.raw_encoder_sentences, validate.n)
    end
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
      params.target, params.specificity, params.include_bits, train.raw_encoder_n_features, metrics.score)
  end

  train.encoder, train.encoder_accuracy, train.encoder_params = optimize.encoder(encoder_args)

  print("\nFinal encoder performance")
  str.printf("  Val | Score: %.4f\n", train.encoder_accuracy.score)
  print("\n  Best TM params:")
  str.printf("    clauses=%d clause_tolerance=%d clause_maximum=%d target=%d specificity=%.2f\n",
    train.encoder_params.clauses,
    train.encoder_params.clause_tolerance,
    train.encoder_params.clause_maximum,
    train.encoder_params.target,
    train.encoder_params.specificity)

  local train_encoder_input = encoder_args.sentences
  local test_encoder_input = test.raw_encoder_sentences
  local validate_encoder_input = validate.raw_encoder_sentences

  print("\nPredicting codes with encoder")
  local train_predicted, test_predicted, validate_predicted
  if cfg.encoder.individualized then
    train_predicted = train.encoder:predict(train_encoder_input, encoder_args.dim_offsets, train.n)
    test_predicted = train.encoder:predict(test_encoder_input, test.raw_encoder_dim_offsets, test.n)
    validate_predicted = train.encoder:predict(validate_encoder_input, validate.raw_encoder_dim_offsets, validate.n)
  else
    train_predicted = train.encoder:predict(train_encoder_input, train.n)
    test_predicted = train.encoder:predict(test_encoder_input, test.n)
    validate_predicted = train.encoder:predict(validate_encoder_input, validate.n)
  end

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

  str.printf("\nEncoder Retrieval: original=%.4f train=%.4f test=%.4f\n",
    train.retrieval_stats.score, train_pred_stats.score, test_pred_stats.score)

  idx_train_pred:destroy()
  idx_test_pred:destroy()

  if cfg.classifier.enabled then
    print("\nClassifier")
    train_predicted:bits_flip_interleave(train.dims_sup)
    validate_predicted:bits_flip_interleave(train.dims_sup)
    test_predicted:bits_flip_interleave(train.dims_sup)

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

    local train_class_pred = classifier:predict(train_predicted, train.n)
    local val_class_pred = classifier:predict(validate_predicted, validate.n)
    local test_class_pred = classifier:predict(test_predicted, test.n)
    local train_class_stats = eval.class_accuracy(train_class_pred, train.solutions, train.n, cfg.data.n_classes)
    local val_class_stats = eval.class_accuracy(val_class_pred, validate.solutions, validate.n, cfg.data.n_classes)
    local test_class_stats = eval.class_accuracy(test_class_pred, test.solutions, test.n, cfg.data.n_classes)

    str.printf("\nClassifier F1: train=%.2f val=%.2f test=%.2f\n",
      train_class_stats.f1, val_class_stats.f1, test_class_stats.f1)
  end

  collectgarbage("collect")

end)
