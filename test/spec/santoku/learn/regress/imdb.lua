local arr = require("santoku.array")
local csr = require("santoku.learn.csr")
local csr_m = require("santoku.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
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
    cgrams_cross = false,
    skips = 1,
  },
  feature_selection = {
    n_selected = 65536,
  },
  tm = {
    classes = 2,
    clauses = { def = 1, min = 1, max = 8 },
    features = 8192, --{ def = 8192, min = 256, max = 8192, pow2 = true },
    absorb_threshold = { def = 0 },
    absorb_maximum_fraction = { def = 0.031 },
    absorb_insert_offset = { def = 1 },
    absorb_ranking_fraction = { def = 0.125 },
    absorb_ranking_limit = { def = 0.125 },
    clause_maximum_fraction = { def = 0.001 },
    clause_tolerance_fraction = { def = 0.5 },
    target_fraction = { def = 0.25 },
    specificity_fraction = { def = 0.00125 },
    alpha_tolerance = { def = 0, min = -3, max = 3 },
    alpha_maximum = { def = 0, min = -3, max = 3 },
    alpha_target = { def = 0, min = -3, max = 3 },
    alpha_specificity = { def = 0, min = -3, max = 3 },
  },
  search = {
    trials = 200,
    iterations = 40,
    subsample = 0.2,
  },
  training = {
    patience = 4,
    batch = 40,
    iterations = 800,
  },
}

local class_names = { "negative", "positive" }

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

  print("\nTokenizing")
  local train_tok_off, train_tok_nbr = tok:tokenize(train.problems)
  local val_tok_off, val_tok_nbr = tok:tokenize(validate.problems)
  local test_tok_off, test_tok_nbr = tok:tokenize(test_set.problems)
  tok = nil -- luacheck: ignore

  print("\nFeature ranking (Chi2)")
  local chi2_ranking, _, class_offsets, class_feat_ids = csr_m.top_chi2(
    train_tok_off, train_tok_nbr,
    train.sol_offsets, train.sol_neighbors,
    n_tokens, cfg.tm.classes,
    cfg.feature_selection.n_selected, nil, "sum")
  train_tok_off, train_tok_nbr = csr_m.select(train_tok_off, train_tok_nbr, chi2_ranking)
  val_tok_off, val_tok_nbr = csr_m.select(val_tok_off, val_tok_nbr, chi2_ranking)
  test_tok_off, test_tok_nbr = csr_m.select(test_tok_off, test_tok_nbr, chi2_ranking)
  class_offsets, class_feat_ids = csr_m.select(class_offsets, class_feat_ids, chi2_ranking)
  n_tokens = chi2_ranking:size()
  str.printf("  Selected %d features\n", n_tokens)

  print("\nConverting to TM representation")
  validate.tokens = csr.to_bits(val_tok_off, val_tok_nbr, validate.n, n_tokens)
  test_set.tokens = csr.to_bits(test_tok_off, test_tok_nbr, test_set.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  local absorb_ranking_global = ivec.create(n_tokens):fill_indices()

  local df_ids, df_scores = csr_m.top_df(train.sol_offsets, train.sol_neighbors, cfg.tm.classes)
  local output_weights = dvec.create():copy(df_scores, df_ids, true)

  local sol_offsets, sol_neighbors = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors

  print("\nOptimizing Regressor")
  local stopwatch = utc.stopwatch()
  local t = optimize.regressor({
    outputs = cfg.tm.classes,
    features = cfg.tm.features,
    n_tokens = n_tokens,
    absorb_threshold = cfg.tm.absorb_threshold,
    absorb_maximum_fraction = cfg.tm.absorb_maximum_fraction,
    absorb_insert_offset = cfg.tm.absorb_insert_offset,
    absorb_ranking_fraction = cfg.tm.absorb_ranking_fraction,
    absorb_ranking_limit = cfg.tm.absorb_ranking_limit,
    clauses = cfg.tm.clauses,
    clause_maximum_fraction = cfg.tm.clause_maximum_fraction,
    clause_tolerance_fraction = cfg.tm.clause_tolerance_fraction,
    target_fraction = cfg.tm.target_fraction,
    specificity_fraction = cfg.tm.specificity_fraction,
    alpha_tolerance = cfg.tm.alpha_tolerance,
    alpha_maximum = cfg.tm.alpha_maximum,
    alpha_target = cfg.tm.alpha_target,
    alpha_specificity = cfg.tm.alpha_specificity,
    output_weights = output_weights,
    samples = train.n,
    sol_offsets = sol_offsets,
    sol_neighbors = sol_neighbors,
    token_offsets = train_tok_off,
    token_neighbors = train_tok_nbr,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    absorb_ranking_global = absorb_ranking_global,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample = cfg.search.subsample,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,
    search_metric = function (regressor)
      local input = { tokens = validate.tokens, n_samples = validate.n }
      local micro_f1, sample_f1 = regressor:label_f1(input, validate.n, val_label_off, val_label_nbr)
      return sample_f1, { micro_f1 = micro_f1, sample_f1 = sample_f1 }
    end,
    each = util.make_labeler_log(stopwatch),
  })

  print()
  print("Final Evaluation")

  print("\nClassification metrics:")
  train.tokens = csr.to_bits(train_tok_off, train_tok_nbr, train.n, n_tokens)
  local _, train_labels = t:label({ tokens = train.tokens, n_samples = train.n }, train.n, 1)
  local _, val_labels = t:label({ tokens = validate.tokens, n_samples = validate.n }, validate.n, 1)
  local _, test_labels = t:label({ tokens = test_set.tokens, n_samples = test_set.n }, test_set.n, 1)

  local train_stats = eval.class_accuracy(train_labels, train.sol_offsets, train.sol_neighbors, train.n, cfg.tm.classes)
  local val_stats = eval.class_accuracy(val_labels, validate.sol_offsets, validate.sol_neighbors, validate.n, cfg.tm.classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.sol_offsets, test_set.sol_neighbors, test_set.n, cfg.tm.classes)
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

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
