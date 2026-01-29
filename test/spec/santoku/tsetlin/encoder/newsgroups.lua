local ann = require("santoku.tsetlin.ann")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.tsetlin.evaluator")
local graph = require("santoku.tsetlin.graph")
local inv = require("santoku.tsetlin.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.tsetlin.util")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max = nil,
    tvr = 0.1,
  },
  tokenizer = {
    max_len = 20,
    min_len = 1,
    max_run = 2,
    ngrams = 2,
    cgrams_min = 3,
    cgrams_max = 5,
    cgrams_cross = true,
    skips = 1,
  },
  feature_selection = {
    max_vocab_spectral = 2^16,
    max_vocab_encoder = 2^14,
  },
  encoder = {
    clauses = 64,
    clause_tolerance = { def = 54, min = 8, max = 128, int = true },
    clause_maximum = { def = 58, min = 16, max = 128, int = true },
    target = { def = 30, min = 4, max = 64, int = true },
    specificity = { def = 308, min = 50, max = 2000 },
    include_bits = { def = 1, min = 1, max = 6, int = true },
    search_patience = 4,
    search_rounds = 6,
    search_trials = 20,
    search_iterations = 10,
    final_patience = 40,
    final_iterations = 400,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 64,
    cmp = "cosine",
    combine = "exponential",
    decay = { def = 2.06, min = 0.0, max = 3.0 },
    binarize = "itq",
    trace_tol = 1e-6,
    rounds = 6,
    samples = 20,
  },
  eval = {
    anchors = 16,
    pairs = 64,
    ranking = "ndcg",
    cmp = "cosine",
  },
  classifier = {
    clauses = 64,
    clause_tolerance = { def = 75, min = 16, max = 128, int = true },
    clause_maximum = { def = 112, min = 16, max = 128, int = true },
    target = { def = 123, min = 16, max = 128, int = true },
    specificity = { def = 3675, min = 400, max = 4000 },
    include_bits = { def = 2, min = 1, max = 4, int = true },
    search_patience = 4,
    search_rounds = 6,
    search_trials = 20,
    search_iterations = 10,
    final_patience = 40,
    final_iterations = 400,
  },
  verbose = true,
}

local n_classes = 20

test("newsgroups", function()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, test_set, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max,
    nil,
    cfg.data.tvr)
  str.printf("  Train: %d  Validate: %d  Test: %d  Classes: %d\n",
    train.n, validate.n, test_set.n, n_classes)

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil -- luacheck: ignore
  validate.problems = nil -- luacheck: ignore
  test_set.problems = nil -- luacheck: ignore

  print("\nCreating IDs")
  train.ids = ivec.create(train.n)
  train.ids:fill_indices()
  validate.ids = ivec.create(validate.n)
  validate.ids:fill_indices()
  validate.ids:add(train.n)
  test_set.ids = ivec.create(test_set.n)
  test_set.ids:fill_indices()
  test_set.ids:add(train.n + validate.n)

  print("\nBuilding label CSR for lookups")
  local train_solutions_bitmap = ivec.create()
  train_solutions_bitmap:copy(train.solutions)
  train_solutions_bitmap:add_scaled(n_classes)
  local train_label_offsets, train_label_neighbors = train_solutions_bitmap:bits_to_csr(train.n, n_classes)
  train.label_csr = { offsets = train_label_offsets, neighbors = train_label_neighbors }
  local validate_solutions_bitmap = ivec.create()
  validate_solutions_bitmap:copy(validate.solutions)
  validate_solutions_bitmap:add_scaled(n_classes)
  local validate_label_offsets, validate_label_neighbors = validate_solutions_bitmap:bits_to_csr(validate.n, n_classes)
  validate.label_csr = { offsets = validate_label_offsets, neighbors = validate_label_neighbors }
  local test_solutions_bitmap = ivec.create()
  test_solutions_bitmap:copy(test_set.solutions)
  test_solutions_bitmap:add_scaled(n_classes)
  local test_label_offsets, test_label_neighbors = test_solutions_bitmap:bits_to_csr(test_set.n, n_classes)
  test_set.label_csr = { offsets = test_label_offsets, neighbors = test_label_neighbors }

  print("\nFeature selection")
  local df_ids, df_scores = train.tokens:bits_top_df(train.n, n_tokens)
  str.printf("  IDF: %d -> %d\n", n_tokens, df_ids:size())
  train.tokens:bits_select(df_ids, nil, n_tokens)
  validate.tokens:bits_select(df_ids, nil, n_tokens)
  test_set.tokens:bits_select(df_ids, nil, n_tokens)
  n_tokens = df_ids:size()
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train_solutions_bitmap, train.n, n_tokens, n_classes,
    cfg.feature_selection.max_vocab_spectral, "max")
  str.printf("  BNS: %d -> %d\n", n_tokens, bns_ids:size())
  local idf_gathered = dvec.create()
  idf_gathered:copy(df_scores, bns_ids)
  bns_scores:scalev(idf_gathered)
  df_ids = nil -- luacheck: ignore
  df_scores = nil -- luacheck: ignore
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  validate.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  local n_top_v = bns_ids:size()
  bns_ids = nil -- luacheck: ignore
  local n_graph_features = n_classes + n_top_v
  local graph_weights = dvec.create(n_graph_features)
  graph_weights:fill(1.0, 0, n_classes)
  graph_weights:copy(bns_scores, n_classes)
  bns_scores = nil -- luacheck: ignore

  print("\nBuilding graph_index (docs only, two-rank: labels + tokens)")
  local graph_features = ivec.create()
  graph_features:copy(train_solutions_bitmap)
  graph_features:bits_extend(train.tokens, n_classes, n_top_v)
  local graph_ranks = ivec.create(n_graph_features)
  graph_ranks:fill(0, 0, n_classes)
  graph_ranks:fill(1, n_classes, n_graph_features)
  train.graph_index = inv.create({
    features = graph_weights,
    ranks = graph_ranks,
    n_ranks = 2,
  })
  train.graph_index:add(graph_features, train.ids)
  graph_features = nil -- luacheck: ignore
  graph_ranks = nil -- luacheck: ignore
  str.printf("  Docs: %d  Features: %d (labels=%d, tokens=%d)\n",
    train.n, n_graph_features, n_classes, n_top_v)

  print("\nBuilding eval_index (labels only) and evaluation adjacency")
  train.eval_index = inv.create({ features = n_classes })
  train.eval_index:add(train_solutions_bitmap, train.ids)
  local train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights =
    graph.adjacency({
      category_index = train.eval_index,
      category_cmp = cfg.eval.cmp,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
  math.randomseed(12345)
  local spot_check_ids = {}
  for i = 1, 5 do
    spot_check_ids[i] = train_eval_ids:get(math.random(train_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights, "train eval")
    util.spot_check_neighbors_with_labels(train_eval_ids, train_eval_offsets, train_eval_neighbors, train_eval_weights,
      train.label_csr, 0, "eval adj (anchors+random)", spot_check_ids, 10)
  end

  print("\nRunning spectral embedding (Nyström)")
  local model = optimize.spectral({
    index = train.graph_index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    cmp = cfg.nystrom.cmp,
    combine = cfg.nystrom.combine,
    decay = cfg.nystrom.decay,
    binarize = cfg.nystrom.binarize,
    trace_tol = cfg.nystrom.trace_tol,
    rounds = cfg.nystrom.rounds,
    samples = cfg.nystrom.samples,
    expected = {
      ids = train_eval_ids,
      offsets = train_eval_offsets,
      neighbors = train_eval_neighbors,
      weights = train_eval_weights,
    },
    eval = { ranking = cfg.eval.ranking },
    each = cfg.verbose and util.spectral_log or nil,
  })
  train.index = model.index
  train.dims = model.dims

  print("\nExtracting spectral codes for training data")
  local train_target_codes = train.index:get(train.ids)
  util.spot_check_codes(train_target_codes, train.n, train.dims, "spectral codes")

  print("\nBit entropy stats for spectral codes:")
  local entropy_ids, entropy_scores = train_target_codes:bits_top_entropy(train.n, train.dims, train.dims)
  local zero_entropy_count = 0
  for i = 0, entropy_ids:size() - 1 do
    local dim = entropy_ids:get(i)
    local ent = entropy_scores:get(i)
    if ent < 0.01 then
      zero_entropy_count = zero_entropy_count + 1
      if zero_entropy_count <= 10 then
        str.printf("  dim %d: entropy=%.6f (near-constant)\n", dim, ent)
      end
    end
  end
  str.printf("  Total near-constant dims (entropy < 0.01): %d / %d\n", zero_entropy_count, train.dims)
  str.printf("  Entropy range: min=%.6f max=%.6f\n", entropy_scores:min(), entropy_scores:max())
  entropy_ids = nil -- luacheck: ignore
  entropy_scores = nil -- luacheck: ignore

  if cfg.verbose then
    print("\nSpot-checking spectral code KNN (compare to eval adj above)")
    local spectral_knn_ids, spectral_knn_offsets, spectral_knn_neighbors, spectral_knn_weights =
      graph.adjacency({
        index = train.index,
        knn_index = train.index,
        knn = 32,
        knn_cache = 32,
        bridge = "none",
      })
    util.spot_check_neighbors_with_labels(spectral_knn_ids, spectral_knn_offsets, spectral_knn_neighbors, spectral_knn_weights,
      train.label_csr, 0, "spectral code KNN", spot_check_ids, 10)
  end

  print("\nEvaluating spectral codes against eval adjacency")
  local spectral_raw_stats = eval.ranking_accuracy({
    raw_codes = model.raw_codes,
    ids = model.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  local spectral_eval_stats = eval.ranking_accuracy({
    index = train.index,
    ids = model.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Spectral codes ranking: raw=%.4f binary=%.4f\n", spectral_raw_stats.score, spectral_eval_stats.score)
  model.raw_codes = nil -- luacheck: ignore

  local encoder_feat_ids = train.tokens:bits_top_chi2(
    train_target_codes, train.n, n_top_v, train.dims,
    cfg.feature_selection.max_vocab_encoder, "max")
  local train_encoder_visible = encoder_feat_ids:size()
  str.printf("  Chi2: %d features\n", train_encoder_visible)
  local train_toks = ivec.create()
  train_toks:copy(train.tokens)
  train_toks:bits_select(encoder_feat_ids, nil, n_top_v)
  local train_encoder_sentences = train_toks:bits_to_cvec(train.n, train_encoder_visible, true)
  train_toks = nil -- luacheck: ignore

  print("\nTraining encoder")
  local encoder_args = {
    hidden = train.dims,
    codes = train_target_codes,
    samples = train.n,
    sentences = train_encoder_sentences,
    visible = train_encoder_visible,
    clauses = cfg.encoder.clauses,
    clause_tolerance = cfg.encoder.clause_tolerance,
    clause_maximum = cfg.encoder.clause_maximum,
    target = cfg.encoder.target,
    specificity = cfg.encoder.specificity,
    include_bits = cfg.encoder.include_bits,
    search_patience = cfg.encoder.search_patience,
    search_rounds = cfg.encoder.search_rounds,
    search_trials = cfg.encoder.search_trials,
    search_iterations = cfg.encoder.search_iterations,
    final_patience = cfg.encoder.final_patience,
    final_iterations = cfg.encoder.final_iterations,
    search_metric = function (t, enc_info)
      local predicted = t:predict(enc_info.sentences, enc_info.samples)
      local accuracy = eval.encoding_accuracy(predicted, train_target_codes, enc_info.samples, train.dims)
      return accuracy.mean_hamming, accuracy
    end,
    each = cfg.verbose and util.encoder_log or nil,
  }
  train.encoder = optimize.encoder(encoder_args)

  print("\nPredicting train codes")
  local train_predicted = train.encoder:predict(train_encoder_sentences, train.n)
  util.spot_check_codes(train_predicted, train.n, train.dims, "train predicted")

  local train_ham = eval.encoding_accuracy(train_predicted, train_target_codes, train.n, train.dims).mean_hamming
  str.printf("  Train hamming: %.4f\n", train_ham)
  train_target_codes = nil -- luacheck: ignore
  train_encoder_sentences = nil -- luacheck: ignore

  print("\nEvaluating predicted codes against eval adjacency")
  local train_pred_ann = ann.create({ features = train.dims })
  train_pred_ann:add(train_predicted, train.ids)

  local pred_eval_stats = eval.ranking_accuracy({
    index = train_pred_ann,
    ids = train.ids,
    eval_ids = train_eval_ids,
    eval_offsets = train_eval_offsets,
    eval_neighbors = train_eval_neighbors,
    eval_weights = train_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Predicted codes ranking score: %.4f (spectral: %.4f)\n",
    pred_eval_stats.score, spectral_eval_stats.score)

  train_eval_ids = nil -- luacheck: ignore
  train_eval_offsets = nil -- luacheck: ignore
  train_eval_neighbors = nil -- luacheck: ignore
  train_eval_weights = nil -- luacheck: ignore
  train_pred_ann = nil -- luacheck: ignore

  print("\nPredicting validate codes")
  local validate_toks = ivec.create()
  validate_toks:copy(validate.tokens)
  validate_toks:bits_select(encoder_feat_ids, nil, n_top_v)
  local validate_encoder_sentences = validate_toks:bits_to_cvec(validate.n, train_encoder_visible, true)
  validate_toks = nil -- luacheck: ignore
  local validate_predicted = train.encoder:predict(validate_encoder_sentences, validate.n)
  util.spot_check_codes(validate_predicted, validate.n, train.dims, "validate predicted")

  print("\nBuilding validate eval_index (labels only) and evaluation adjacency")
  validate.eval_index = inv.create({ features = n_classes })
  validate.eval_index:add(validate_solutions_bitmap, validate.ids)
  local validate_eval_ids, validate_eval_offsets, validate_eval_neighbors, validate_eval_weights =
    graph.adjacency({
      category_index = validate.eval_index,
      category_cmp = cfg.eval.cmp,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
  math.randomseed(23456)
  local validate_spot_check_ids = {}
  for i = 1, 5 do
    validate_spot_check_ids[i] = validate_eval_ids:get(math.random(validate_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(validate_eval_ids, validate_eval_offsets, validate_eval_neighbors, validate_eval_weights, "validate eval")
    util.spot_check_neighbors_with_labels(validate_eval_ids, validate_eval_offsets, validate_eval_neighbors, validate_eval_weights,
      validate.label_csr, train.n, "validate eval adj (anchors+random)", validate_spot_check_ids, 10)
  end

  print("\nEvaluating validate predicted codes")
  local validate_pred_ann = ann.create({ features = train.dims })
  validate_pred_ann:add(validate_predicted, validate.ids)
  local validate_pred_stats = eval.ranking_accuracy({
    index = validate_pred_ann,
    ids = validate.ids,
    eval_ids = validate_eval_ids,
    eval_offsets = validate_eval_offsets,
    eval_neighbors = validate_eval_neighbors,
    eval_weights = validate_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Validate ranking score: %.4f\n", validate_pred_stats.score)
  validate_encoder_sentences = nil -- luacheck: ignore
  validate_pred_ann = nil -- luacheck: ignore
  validate_eval_ids = nil -- luacheck: ignore
  validate_eval_offsets = nil -- luacheck: ignore
  validate_eval_neighbors = nil -- luacheck: ignore
  validate_eval_weights = nil -- luacheck: ignore

  print("\nPredicting test codes")
  local test_toks = ivec.create()
  test_toks:copy(test_set.tokens)
  test_toks:bits_select(encoder_feat_ids, nil, n_top_v)
  local test_encoder_sentences = test_toks:bits_to_cvec(test_set.n, train_encoder_visible, true)
  test_toks = nil -- luacheck: ignore
  local test_predicted = train.encoder:predict(test_encoder_sentences, test_set.n)
  util.spot_check_codes(test_predicted, test_set.n, train.dims, "test predicted")

  print("\nBuilding test eval_index (labels only) and evaluation adjacency")
  test_set.eval_index = inv.create({ features = n_classes })
  test_set.eval_index:add(test_solutions_bitmap, test_set.ids)
  local test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights =
    graph.adjacency({
      category_index = test_set.eval_index,
      category_cmp = cfg.eval.cmp,
      category_anchors = cfg.eval.anchors,
      random_pairs = cfg.eval.pairs,
    })
  math.randomseed(34567)
  local test_spot_check_ids = {}
  for i = 1, 5 do
    test_spot_check_ids[i] = test_eval_ids:get(math.random(test_eval_ids:size()) - 1)
  end
  if cfg.verbose then
    util.spot_check_adjacency(test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights, "test eval")
    util.spot_check_neighbors_with_labels(test_eval_ids, test_eval_offsets, test_eval_neighbors, test_eval_weights,
      test_set.label_csr, train.n + validate.n, "test eval adj (anchors+random)", test_spot_check_ids, 10)
  end

  print("\nEvaluating test predicted codes")
  local test_pred_ann = ann.create({ features = train.dims })
  test_pred_ann:add(test_predicted, test_set.ids)
  local test_pred_stats = eval.ranking_accuracy({
    index = test_pred_ann,
    ids = test_set.ids,
    eval_ids = test_eval_ids,
    eval_offsets = test_eval_offsets,
    eval_neighbors = test_eval_neighbors,
    eval_weights = test_eval_weights,
    ranking = cfg.eval.ranking,
    n_dims = train.dims,
  })
  str.printf("  Test ranking score: %.4f\n", test_pred_stats.score)
  test_encoder_sentences = nil -- luacheck: ignore
  test_pred_ann = nil -- luacheck: ignore

  print("\nClassifier evaluation")
  train_predicted:bits_flip_interleave(train.dims)
  validate_predicted:bits_flip_interleave(train.dims)
  test_predicted:bits_flip_interleave(train.dims)

  local classifier = optimize.classifier({
    features = train.dims,
    classes = n_classes,
    clauses = cfg.classifier.clauses,
    clause_tolerance = cfg.classifier.clause_tolerance,
    clause_maximum = cfg.classifier.clause_maximum,
    target = cfg.classifier.target,
    specificity = cfg.classifier.specificity,
    include_bits = cfg.classifier.include_bits,
    samples = train.n,
    problems = train_predicted,
    solutions = train.solutions,
    search_patience = cfg.classifier.search_patience,
    search_rounds = cfg.classifier.search_rounds,
    search_trials = cfg.classifier.search_trials,
    search_iterations = cfg.classifier.search_iterations,
    final_patience = cfg.classifier.final_patience,
    final_iterations = cfg.classifier.final_iterations,
    search_metric = function (t)
      local predicted = t:predict(validate_predicted, validate.n)
      local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, n_classes)
      return accuracy.f1, accuracy
    end,
    each = cfg.verbose and util.make_classifier_log(stopwatch) or nil,
  })

  local train_class_pred = classifier:predict(train_predicted, train.n)
  local validate_class_pred = classifier:predict(validate_predicted, validate.n)
  local test_class_pred = classifier:predict(test_predicted, test_set.n)
  local train_class_stats = eval.class_accuracy(train_class_pred, train.solutions, train.n, n_classes)
  local validate_class_stats = eval.class_accuracy(validate_class_pred, validate.solutions, validate.n, n_classes)
  local test_class_stats = eval.class_accuracy(test_class_pred, test_set.solutions, test_set.n, n_classes)

  print("\n" .. string.rep("=", 60))
  print("SUMMARY")
  print(string.rep("=", 60))
  str.printf("  Spectral dims: %d\n", train.dims)
  str.printf("  Train spectral score: %.4f\n", spectral_eval_stats.score)
  str.printf("  Train predicted score: %.4f  hamming: %.4f\n", pred_eval_stats.score, train_ham)
  str.printf("  Validate predicted score: %.4f\n", validate_pred_stats.score)
  str.printf("  Test predicted score: %.4f\n", test_pred_stats.score)
  str.printf("  Classifier F1: train=%.2f validate=%.2f test=%.2f\n",
    train_class_stats.f1, validate_class_stats.f1, test_class_stats.f1)
  str.printf("  Time: %.1fs\n", stopwatch())

end)
