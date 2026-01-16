local cvec = require("santoku.cvec")
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

local cfg; cfg = {
  data = {
    ttr = 0.9,
    tvr = 0.1,
    max = nil,
    max_class = nil,
    visible = 784,
  },
  encoder = {
    individualized = true,
    max_vocab = 784,
    selection = "chi2",
  },
  tm = {
    clauses = { def = 24, min = 8, max = 256, round = 8 },
    clause_tolerance = { def = 64, min = 16, max = 128, int = true },
    clause_maximum = { def = 64, min = 16, max = 128, int = true },
    target = { def = 32, min = 16, max = 128, int = true },
    specificity = { def = 1000, min = 400, max = 4000 },
    include_bits = { def = 1, min = 1, max = 4, int = true },
  },
  tm_search = {
    patience = 2,
    rounds = 4,
    trials = 10,
    iterations = 20,
  },
  training = {
    patience = 2,
    iterations = 200,
  },
  classifier = {
    enabled = true,
    clauses = { def = 8, min = 8, max = 32, round = 8 },
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
    rounds = 4,
    adjacency_samples = 4,
    spectral_samples = 2,
    eval_samples = 2,
    adjacency = {
      knn = { def = 24, min = 16, max = 32, int = true },
      knn_alpha = { def = 20, min = 12, max = 28, int = true },
      weight_decay = { def = 8, min = 2, max = 16 },
      knn_mutual = true,
      knn_mode = "cknn",
      knn_cache = 32,
      bridge = "mst",
    },
    spectral = {
      laplacian = "unnormalized",
      n_dims = 24,
      eps = 1e-12,
      threshold = {
        method = "otsu",
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
      metric = "avg",
      select_elbow = "lmethod",
      select_metric = "entropy",
    },
    verbose = true,
  },
}

test("mnist-raw", function()

  local stopwatch = utc.stopwatch()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.visible, cfg.data.max, cfg.data.max_class)
  dataset.n_visible = cfg.data.visible
  dataset.n_hidden = cfg.search.spectral.n_dims

  print("\nSplitting")
  local train, test, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test.n)

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  validate.ids = ivec.create(validate.n)
  validate.ids:fill_indices()
  validate.ids:add(train.n)
  test.ids = ivec.create(test.n)
  test.ids:fill_indices()
  test.ids:add(train.n + validate.n)

  print("\nExtracting train pixels")
  train.pixels = ivec.create()
  dataset.problems:bits_select(nil, train.ids, cfg.data.visible, train.pixels)

  print("\nBuilding pixel similarity graph")
  do
    local graph_ids = ivec.create()
    graph_ids:copy(train.ids)
    local graph_problems = ivec.create()
    graph_problems:copy(train.pixels)
    local problems_ext = ivec.create()
    problems_ext:copy(train.solutions)
    problems_ext:add_scaled(10)
    graph_problems:bits_extend(problems_ext, cfg.data.visible, 10)
    local graph_ranks = ivec.create(cfg.data.visible + 10)
    graph_ranks:fill(1, 0, cfg.data.visible)
    graph_ranks:fill(0, cfg.data.visible, cfg.data.visible + 10)
    train.index_graph = inv.create({
      features = cfg.data.visible + 10,
      ranks = graph_ranks,
      n_ranks = 2,
    })
    train.index_graph:add(graph_problems, graph_ids)
  end

  print("\nBuilding node features index (raw pixels)")
  train.node_features = ann.create({ features = dataset.n_visible, expected_size = train.n })
  train.train_raw = train.pixels:bits_to_cvec(train.n, dataset.n_visible)
  train.node_features:add(train.train_raw, train.ids)

  local function build_category_ground_truth (ids)
    local cat_index = inv.create({ features = 10 })
    local data = ivec.create()
    data:copy(train.solutions)
    data:add_scaled(10)
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

  print("\nBuilding category ground truth")
  train.cat_index, train.ground_truth = build_category_ground_truth(train.ids)

  print("\nOptimizing spectral pipeline")
  local model = optimize.spectral({
    index = train.index_graph,
    knn_index = train.node_features,
    rounds = cfg.search.rounds,
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
        str.printf("\n--- Round %d complete: best=%.4f (global=%.4f) success=%.0f%% adapt=%.2f ---\n",
          info.round, info.round_best_score, info.global_best_score,
          (info.success_rate or 0) * 100, info.adapt_factor or 1)
      elseif info.event == "stage" and info.stage == "adjacency" then
        local p = info.params.adjacency
        if info.is_final then
          str.printf("\n  [Final] decay=%.2f knn=%d knn_alpha=%d knn_mutual=%s\n",
            p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual))
        else
          str.printf("\n  [%d] decay=%.2f knn=%d knn_alpha=%d knn_mutual=%s\n",
            info.sample, p.weight_decay, p.knn, p.knn_alpha, tostring(p.knn_mutual))
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
        local select_str = ""
        if e.select_metric and e.select_metric ~= "none" then
          local sel_alpha = e.select_elbow_alpha and str.format("%.1f", e.select_elbow_alpha) or "-"
          local dims_str
          if m.selected_elbow and m.selected_elbow <= 0 then
            dims_str = str.format("FAIL->%d", m.n_dims)
          elseif m.selected_elbow and m.selected_elbow ~= m.n_dims then
            dims_str = str.format("%d->%d", m.selected_elbow, m.n_dims)
          else
            dims_str = str.format("%d", m.n_dims)
          end
          select_str = str.format(" [%s/%s(%s) %s]", e.select_metric, e.select_elbow, sel_alpha, dims_str)
        end
        str.printf("    knn=%d score=%.4f%s\n", e.knn, m.score, select_str)
      end
    end or nil,
  })
  train.ids_sup = model.ids
  train.codes_sup = model.codes
  train.dims_sup = model.dims
  train.index_sup = model.index
  train.retrieval_stats = model.metrics

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

  print("\nCodebook stats")
  train.entropy = eval.entropy_stats(train.codes_sup, train.ids_sup:size(), train.dims_sup)
  str.printi("  Entropy: %.4f#(mean) | Min: %.4f#(min) | Max: %.4f#(max) | Std: %.4f#(std)", train.entropy)

  print("\nBuilding retrieved adjacency (spectral KNN)")
  train.adj_retrieved_ids,
  train.adj_retrieved_offsets,
  train.adj_retrieved_neighbors,
  train.adj_retrieved_weights = graph.adjacency({
    weight_index = train.index_sup,
    seed_ids = train.adj_expected_ids,
    seed_offsets = train.adj_expected_offsets,
    seed_neighbors = train.adj_expected_neighbors,
  })

  print("Building expected codes")
  train.codes_expected = cvec.create()
  train.codes_expected:bits_extend(train.codes_sup, train.adj_expected_ids, train.ids_sup, 0, train.dims_sup, true)

  local base_retrieval = eval.score_retrieval({
    retrieved_ids = train.adj_retrieved_ids,
    retrieved_offsets = train.adj_retrieved_offsets,
    retrieved_neighbors = train.adj_retrieved_neighbors,
    retrieved_weights = train.adj_retrieved_weights,
    expected_ids = train.adj_expected_ids,
    expected_offsets = train.adj_expected_offsets,
    expected_neighbors = train.adj_expected_neighbors,
    expected_weights = train.adj_expected_weights,
    ranking = cfg.search.eval.ranking,
    metric = cfg.search.eval.metric,
    n_dims = train.dims_sup,
  })

  str.printf("\nRetrieval: score=%.4f\n", base_retrieval.score)

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

  local validate_pixels = ivec.create()
  dataset.problems:bits_select(nil, validate.ids, cfg.data.visible, validate_pixels)
  local test_pixels = ivec.create()
  dataset.problems:bits_select(nil, test.ids, cfg.data.visible, test_pixels)

  if use_ind then
    local ids_union, feat_offsets, feat_ids = train.pixels:bits_top_chi2_ind(
      train.codes_sup, train.n, cfg.data.visible, train.dims_sup,
      cfg.encoder.max_vocab)
    local union_size = ids_union:size()
    local total_features = feat_offsets:get(train.dims_sup)
    str.printf("  Per-dimension chi2: union=%d total=%d (%.1fx expansion)\n",
      union_size, total_features, total_features / union_size)
    local function to_ind_bitmap (pixels, n)
      local selected = ivec.create()
      pixels:bits_select(ids_union, nil, cfg.data.visible, selected)
      local ind, ind_off = selected:bits_individualize(feat_offsets, feat_ids, union_size)
      local bitmap, dim_off = ind:bits_to_cvec_ind(ind_off, feat_offsets, n, true)
      selected:destroy()
      ind:destroy()
      ind_off:destroy()
      return bitmap, dim_off
    end
    local train_bitmap, train_dim_off = to_ind_bitmap(train.pixels, train.n)
    local val_bitmap, val_dim_off = to_ind_bitmap(validate_pixels, validate.n)
    local test_bitmap, test_dim_off = to_ind_bitmap(test_pixels, test.n)
    encoder_args.sentences = train_bitmap
    encoder_args.visible = union_size
    encoder_args.individualized = true
    encoder_args.feat_offsets = feat_offsets
    encoder_args.dim_offsets = train_dim_off
    validate.raw_encoder_sentences = val_bitmap
    validate.raw_encoder_dim_offsets = val_dim_off
    test.raw_encoder_sentences = test_bitmap
    test.raw_encoder_dim_offsets = test_dim_off
    train.raw_encoder_n_features = union_size
  else
    local raw_vocab, raw_scores
    if selection_method == "mi" then
      raw_vocab, raw_scores = train.pixels:bits_top_mi(
        train.codes_sup,
        train.n,
        cfg.data.visible,
        train.dims_sup,
        cfg.encoder.max_vocab)
    else
      raw_vocab, raw_scores = train.pixels:bits_top_chi2(
        train.codes_sup,
        train.n,
        cfg.data.visible,
        train.dims_sup,
        cfg.encoder.max_vocab)
    end
    local n_raw_v = raw_vocab:size()
    str.printf("  Selected %d pixels by %s\n", n_raw_v, selection_method)
    str.printf("  Score range: %.4f - %.4f\n", raw_scores:get(n_raw_v - 1), raw_scores:get(0))
    train.raw_encoder_vocab = raw_vocab
    train.raw_encoder_n_features = n_raw_v
    local train_raw_selected = ivec.create()
    train.pixels:bits_select(raw_vocab, nil, cfg.data.visible, train_raw_selected)
    local validate_raw_selected = ivec.create()
    validate_pixels:bits_select(raw_vocab, nil, cfg.data.visible, validate_raw_selected)
    local test_raw_selected = ivec.create()
    test_pixels:bits_select(raw_vocab, nil, cfg.data.visible, test_raw_selected)
    local train_raw_sentences = train_raw_selected:bits_to_cvec(train.n, n_raw_v, true)
    validate.raw_encoder_sentences = validate_raw_selected:bits_to_cvec(validate.n, n_raw_v, true)
    test.raw_encoder_sentences = test_raw_selected:bits_to_cvec(test.n, n_raw_v, true)
    encoder_args.sentences = train_raw_sentences
    encoder_args.visible = n_raw_v
  end

  print("Building expected adjacency for validate")
  validate.cat_index = inv.create({
    features = 10,
    expected_size = validate.n,
  })
  local val_data = ivec.create()
  val_data:copy(validate.solutions)
  val_data:add_scaled(10)
  validate.cat_index:add(val_data, validate.ids)
  val_data:destroy()
  local val_adj_expected_ids, val_adj_expected_offsets, val_adj_expected_neighbors, val_adj_expected_weights =
    graph.adjacency({
      category_index = validate.cat_index,
      category_anchors = cfg.search.eval.anchors,
      random_pairs = cfg.search.eval.pairs,
    })

  print("Building expected adjacency for test")
  test.cat_index = inv.create({
    features = 10,
    expected_size = test.n,
  })
  local test_data = ivec.create()
  test_data:copy(test.solutions)
  test_data:add_scaled(10)
  test.cat_index:add(test_data, test.ids)
  test_data:destroy()
  local test_adj_expected_ids, test_adj_expected_offsets, test_adj_expected_neighbors, test_adj_expected_weights =
    graph.adjacency({
      category_index = test.cat_index,
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
  local validate_encoder_input = validate.raw_encoder_sentences
  local test_encoder_input = test.raw_encoder_sentences

  print("\nPredicting codes with encoder")
  local train_predicted, validate_predicted, test_predicted
  if cfg.encoder.individualized then
    train_predicted = train.encoder:predict(train_encoder_input, encoder_args.dim_offsets, train.n)
    validate_predicted = train.encoder:predict(validate_encoder_input, validate.raw_encoder_dim_offsets, validate.n)
    test_predicted = train.encoder:predict(test_encoder_input, test.raw_encoder_dim_offsets, test.n)
  else
    train_predicted = train.encoder:predict(train_encoder_input, train.n)
    validate_predicted = train.encoder:predict(validate_encoder_input, validate.n)
    test_predicted = train.encoder:predict(test_encoder_input, test.n)
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
    base_retrieval.score, train_pred_stats.score, test_pred_stats.score)

  idx_train_pred:destroy()
  idx_test_pred:destroy()

  if cfg.classifier.enabled then
    print("\nClassifier")
    train_predicted:bits_flip_interleave(train.dims_sup)
    validate_predicted:bits_flip_interleave(train.dims_sup)
    test_predicted:bits_flip_interleave(train.dims_sup)

    local classifier = optimize.classifier({
      features = train.dims_sup,
      classes = 10,
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
        local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, 10)
        return accuracy.f1, accuracy
      end,
      each = function (t, is_final, val_accuracy, params, epoch, round, trial)
        local test_pred = t:predict(test_predicted, test.n)
        local test_accuracy = eval.class_accuracy(test_pred, test.solutions, test.n, 10)
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
    local train_class_stats = eval.class_accuracy(train_class_pred, train.solutions, train.n, 10)
    local val_class_stats = eval.class_accuracy(val_class_pred, validate.solutions, validate.n, 10)
    local test_class_stats = eval.class_accuracy(test_class_pred, test.solutions, test.n, 10)

    str.printf("\nClassifier F1: train=%.2f val=%.2f test=%.2f\n",
      train_class_stats.f1, val_class_stats.f1, test_class_stats.f1)
  end

  collectgarbage("collect")

end)
