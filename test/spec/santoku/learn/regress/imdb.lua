local arr = require("santoku.array")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local ivec = require("santoku.ivec")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")
local util = require("santoku.learn.util")

local cfg = {
  data = {
    ttr = 0.5,
    tvr = 0.1,
    max = nil,
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
    n_selected = 8192,
  },
  tm = {
    state = 8,
    classes = 2,
    features = 1024,
    clauses = 32,
    absorb_interval = { def = 1, min = 1, max = 40 },
    absorb_threshold = { def = 0, min = 0, max = 256, int = true },
    absorb_maximum = { def = 0, min = 0, max = 256, int = true },
    absorb_insert = { def = 1, min = 1, max = 256, int = true },
    clause_tolerance = { def = 16, min = 8, max = 1024, int = true },
    clause_maximum = { def = 16, min = 8, max = 1024, int = true },
    target = { def = 16, min = 8, max = 1024, int = true },
    specificity = { def = 800, min = 2, max = 2000, int = true },
  },
  search = {
    rounds = 6,
    trials = 20,
    iterations = 80,
    subsample_samples = 0.2,
  },
  training = {
    patience = 400,
    batch = 40,
    iterations = 2000,
  },
}

test("imdb regressor", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  str.printf("  Vocabulary: %d\n", n_tokens)

  local class_names = { "negative", "positive" }
  local stopwatch = utc.stopwatch()
  local predicted_buf = ivec.create()

  print("\nTokenizing")
  train.tokens = tok:tokenize(train.problems)
  validate.tokens = tok:tokenize(validate.problems)
  test_set.tokens = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore

  local class_offsets, class_feat_ids
  if cfg.feature_selection.n_selected then
    print("\nFeature ranking (Chi2)")
    train.solutions:add_scaled(cfg.tm.classes)
    local chi2_ranking
    chi2_ranking, _, class_offsets, class_feat_ids = train.tokens:bits_top_chi2( -- luacheck: ignore
      train.solutions, train.n, n_tokens, cfg.tm.classes,
      cfg.feature_selection.n_selected, nil, "sum")
    train.tokens:bits_select(chi2_ranking, nil, n_tokens)
    validate.tokens:bits_select(chi2_ranking, nil, n_tokens)
    test_set.tokens:bits_select(chi2_ranking, nil, n_tokens)
    class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, chi2_ranking)
    n_tokens = chi2_ranking:size()
    train.solutions:add_scaled(-cfg.tm.classes)
    str.printf("  Selected %d features\n", n_tokens)
  end

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  local absorb_ranking_global = ivec.create(n_tokens):fill_indices()

  print("\nOptimizing Regressor")
  local t = optimize.regressor({
    state = cfg.tm.state,
    outputs = cfg.tm.classes,
    features = cfg.tm.features,
    n_tokens = n_tokens,
    absorb_interval = cfg.tm.absorb_interval,
    absorb_threshold = cfg.tm.absorb_threshold,
    absorb_maximum = cfg.tm.absorb_maximum,
    absorb_insert = cfg.tm.absorb_insert,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,
    samples = train.n,
    solutions = train.solutions,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    absorb_ranking_global = absorb_ranking_global,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample_samples = cfg.search.subsample_samples,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,
    search_metric = function (regressor)
      local scores = regressor:classify(
        { tokens = validate.tokens, n_samples = validate.n },
        validate.n, false, predicted_buf)
      local stats = eval.class_accuracy(scores, validate.solutions, validate.n, cfg.tm.classes)
      return stats.f1, stats
    end,
    each = util.make_classifier_log(stopwatch),
  })

  print()
  print("Final Evaluation")

  print("\nClassification metrics:")
  local train_scores = t:classify({ tokens = train.tokens, n_samples = train.n }, train.n, false)
  local val_scores = t:classify({ tokens = validate.tokens, n_samples = validate.n }, validate.n, false)
  local test_scores = t:classify({ tokens = test_set.tokens, n_samples = test_set.n }, test_set.n, false)

  local train_stats = eval.class_accuracy(train_scores, train.solutions, train.n, cfg.tm.classes)
  local val_stats = eval.class_accuracy(val_scores, validate.solutions, validate.n, cfg.tm.classes)
  local test_stats = eval.class_accuracy(test_scores, test_set.solutions, test_set.n, cfg.tm.classes)
  str.printf("  F1:   Train=%.2f  Val=%.2f  Test=%.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", class_names[c], ts.f1, ts.precision, ts.recall)
  end

end)
