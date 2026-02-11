local csr = require("santoku.tsetlin.csr")
local ds = require("santoku.tsetlin.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.tsetlin.evaluator")
local inv = require("santoku.tsetlin.inv")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local util = require("santoku.tsetlin.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = {
    max = nil,
    tvr = 0.1,
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
    n_bns = nil,
    n_selected = 8192,
  },
  nystrom = {
    n_landmarks = 4096,
    n_dims = 256,
    decay = 0,
    bandwidth = -1,
    rounds = 0,
  },
  eval = {
    knn = 16,
    random_pairs = 16,
    ranking = "ndcg",
  },
  regressor = {
    features = 2048,
    absorb_interval = 1, --{ def = 10, min = 1, max = 40 },
    absorb_threshold = { def = 0, min = 0, max = 127, int = true },
    absorb_maximum = { def = 0, min = 0, max = 128, int = true },
    absorb_insert = { def = 1, min = 1, max = 126, int = true },
    clauses = 32,
    clause_tolerance = { def = 16, min = 8, max = 1024, int = true },
    clause_maximum = { def = 16, min = 8, max = 1024, int = true },
    target = { def = 16, min = 8, max = 1024, int = true },
    specificity = { def = 800, min = 2, max = 2000 },
    search_rounds = 6,
    search_trials = 10,
    search_iterations = 10,
    search_subsample = 0.2,
    final_patience = 20,
    final_batch = 40,
    final_iterations = 400,
  },
}

test("newsgroups-embedding", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, test_set, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max,
    nil,
    cfg.data.tvr)
  str.printf("  Train: %d  Validate: %d  Test: %d\n", train.n, validate.n, test_set.n)

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil
  train.problems = nil
  validate.problems = nil
  test_set.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  local n_classes = train.n_labels
  print("\nBNS feature selection")
  train.solutions:add_scaled(n_classes)
  local bns_ids, bns_scores = train.tokens:bits_top_bns(
    train.solutions, train.n, n_tokens, n_classes,
    cfg.feature_selection.n_bns, nil, "max")
  train.solutions:add_scaled(-n_classes)
  train.tokens:bits_select(bns_ids, nil, n_tokens)
  validate.tokens:bits_select(bns_ids, nil, n_tokens)
  test_set.tokens:bits_select(bns_ids, nil, n_tokens)
  local n_features = bns_ids:size()
  n_tokens = n_features
  str.printf("  %d features selected\n", n_features)

  print("\nBuilding doc-doc index")
  train.ids = ivec.create(train.n):fill_indices()
  validate.ids = ivec.create(validate.n):fill_indices():add(train.n)
  test_set.ids = ivec.create(test_set.n):fill_indices():add(train.n + validate.n)
  local index = inv.create({ features = bns_scores })
  index:add(train.tokens, train.ids)
  index:add(validate.tokens, validate.ids)
  index:add(test_set.tokens, test_set.ids)
  str.printf("  %d docs indexed\n", train.n + validate.n + test_set.n)

  print("\nBuilding eval adjacency")
  local eval_uids, inv_hoods = index:neighborhoods(cfg.eval.knn, 0, -1)
  local eval_off, eval_nbr, eval_w = inv_hoods:to_csr(eval_uids)
  local rp_off, rp_nbr, rp_w = csr.random_pairs(eval_uids, cfg.eval.random_pairs)
  csr.weight_from_index(eval_uids, rp_off, rp_nbr, rp_w, index, 0, -1)
  eval_off, eval_nbr, eval_w = csr.merge(eval_off, eval_nbr, eval_w, rp_off, rp_nbr, rp_w)
  eval_off, eval_nbr, eval_w = csr.symmetrize(eval_off, eval_nbr, eval_w, eval_uids:size())
  str.printf("  %d nodes, %d edges\n", eval_uids:size(), eval_nbr:size())

  print("\nSpectral embedding (Nystrom)")
  local spectral_metrics
  local model = optimize.spectral({
    index = index,
    n_landmarks = cfg.nystrom.n_landmarks,
    n_dims = cfg.nystrom.n_dims,
    decay = cfg.nystrom.decay,
    bandwidth = cfg.nystrom.bandwidth,
    rounds = cfg.nystrom.rounds,
    expected_ids = eval_uids,
    expected_offsets = eval_off,
    expected_neighbors = eval_nbr,
    expected_weights = eval_w,
    ranking = cfg.eval.ranking,
    each = function (ev)
      util.spectral_log(ev)
      if ev.event == "eval" or ev.event == "done" then
        spectral_metrics = ev.metrics or ev.best_metrics
      end
    end,
  })
  local spectral_dims = model.dims
  local all_raw_codes = model.raw_codes
  local embedded_ids = model.ids
  str.printf("  dims=%d, embedded=%d\n", spectral_dims, embedded_ids:size())

  local train_raw_codes = dvec.create():mtx_extend(
    all_raw_codes, train.ids, embedded_ids, 0, spectral_dims, true)

  local train_bns_tokens = ivec.create():copy(train.tokens)
  local val_bns_tokens = ivec.create():copy(validate.tokens)
  local test_bns_tokens = ivec.create():copy(test_set.tokens)

  print("\nFeature selection for regressor (F-score)")
  local n_selected = cfg.feature_selection.n_selected
  local reg_ids = train.tokens:bits_top_reg_f(
    train_raw_codes, train.n, n_tokens, spectral_dims, n_selected)
  train.tokens:bits_select(reg_ids, nil, n_tokens)
  validate.tokens:bits_select(reg_ids, nil, n_tokens)
  test_set.tokens:bits_select(reg_ids, nil, n_tokens)
  n_tokens = reg_ids:size()
  str.printf("  %d features selected\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  print("\nTraining regressor")
  local predicted_buf = dvec.create()
  local tm = optimize.regressor({
    outputs = spectral_dims,
    samples = train.n,
    features = cfg.regressor.features,
    n_tokens = n_tokens,
    absorb_interval = cfg.regressor.absorb_interval,
    absorb_threshold = cfg.regressor.absorb_threshold,
    absorb_maximum = cfg.regressor.absorb_maximum,
    absorb_insert = cfg.regressor.absorb_insert,
    clauses = cfg.regressor.clauses,
    clause_tolerance = cfg.regressor.clause_tolerance,
    clause_maximum = cfg.regressor.clause_maximum,
    target = cfg.regressor.target,
    specificity = cfg.regressor.specificity,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    targets = train_raw_codes,
    search_rounds = cfg.regressor.search_rounds,
    search_trials = cfg.regressor.search_trials,
    search_iterations = cfg.regressor.search_iterations,
    search_subsample = cfg.regressor.search_subsample,
    final_batch = cfg.regressor.final_batch,
    final_patience = cfg.regressor.final_patience,
    final_iterations = cfg.regressor.final_iterations,
    search_metric = function (t, targs)
      local input = { tokens = targs.tokens, n_samples = targs.samples }
      local predicted = t:regress(input, targs.samples, true, predicted_buf)
      local stats = eval.regression_accuracy(predicted, targs.targets)
      return -stats.mean, stats
    end,
    each = util.make_regressor_log(stopwatch),
  })

  print("\n" .. string.rep("=", 60))
  print("FINAL EVALUATION")
  print(string.rep("=", 60))

  local function build_split_eval(tokens, n)
    local ids = ivec.create(n):fill_indices()
    local idx = inv.create({ features = bns_scores })
    idx:add(tokens, ids)
    local uids, hoods = idx:neighborhoods(cfg.eval.knn, 0, -1)
    local off, nbr, w = hoods:to_csr(uids)
    local rp_off, rp_nbr, rp_w = csr.random_pairs(uids, cfg.eval.random_pairs)
    csr.weight_from_index(uids, rp_off, rp_nbr, rp_w, idx, 0, -1)
    off, nbr, w = csr.merge(off, nbr, w, rp_off, rp_nbr, rp_w)
    off, nbr, w = csr.symmetrize(off, nbr, w, uids:size())
    return uids, off, nbr, w
  end

  print("\nBuilding per-split evaluation indexes")
  local tr_uids, tr_off, tr_nbr, tr_w = build_split_eval(train_bns_tokens, train.n)
  local va_uids, va_off, va_nbr, va_w = build_split_eval(val_bns_tokens, validate.n)
  local te_uids, te_off, te_nbr, te_w = build_split_eval(test_bns_tokens, test_set.n)
  str.printf("  Train: %d nodes, %d edges\n", tr_uids:size(), tr_nbr:size())
  str.printf("  Validate: %d nodes, %d edges\n", va_uids:size(), va_nbr:size())
  str.printf("  Test: %d nodes, %d edges\n", te_uids:size(), te_nbr:size())

  print("\nPredicting embeddings")
  local train_predicted = tm:regress(
    { tokens = train.tokens, n_samples = train.n }, train.n, true)
  local val_predicted = tm:regress(
    { tokens = validate.tokens, n_samples = validate.n }, validate.n, true)
  local test_predicted = tm:regress(
    { tokens = test_set.tokens, n_samples = test_set.n }, test_set.n, true)

  local train_ids = ivec.create(train.n):fill_indices()
  local val_ids = ivec.create(validate.n):fill_indices()
  local test_ids = ivec.create(test_set.n):fill_indices()

  local spectral_raw = eval.ranking_accuracy({
    raw_codes = train_raw_codes, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = tr_uids, eval_offsets = tr_off,
    eval_neighbors = tr_nbr, eval_weights = tr_w,
    ranking = cfg.eval.ranking,
  })

  local train_ranking = eval.ranking_accuracy({
    raw_codes = train_predicted, ids = train_ids,
    n_dims = spectral_dims,
    eval_ids = tr_uids, eval_offsets = tr_off,
    eval_neighbors = tr_nbr, eval_weights = tr_w,
    ranking = cfg.eval.ranking,
  })

  local val_ranking = eval.ranking_accuracy({
    raw_codes = val_predicted, ids = val_ids,
    n_dims = spectral_dims,
    eval_ids = va_uids, eval_offsets = va_off,
    eval_neighbors = va_nbr, eval_weights = va_w,
    ranking = cfg.eval.ranking,
  })

  local test_ranking = eval.ranking_accuracy({
    raw_codes = test_predicted, ids = test_ids,
    n_dims = spectral_dims,
    eval_ids = te_uids, eval_offsets = te_off,
    eval_neighbors = te_nbr, eval_weights = te_w,
    ranking = cfg.eval.ranking,
  })

  local train_reg = eval.regression_accuracy(train_predicted, train_raw_codes)

  str.printf("\n  Spectral dims: %d\n", spectral_dims)
  str.printf("\n  Ranking (%s):\n", cfg.eval.ranking)
  str.printf("  %-28s %8.4f\n", "Spectral raw (train ceil):", spectral_raw.score)
  str.printf("  %-28s %8.4f\n", "Predicted train:", train_ranking.score)
  str.printf("  %-28s %8.4f\n", "Predicted validate:", val_ranking.score)
  str.printf("  %-28s %8.4f\n", "Predicted test:", test_ranking.score)
  str.printf("\n  Regression MAE (train): %.4f\n", train_reg.mean)

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
