local arr = require("santoku.array")
local csr = require("santoku.learn.csr")
local csr_m = require("santoku.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local fs = require("santoku.fs")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tm = require("santoku.learn")
local utc = require("santoku.utc")
local util = require("santoku.learn.util")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.1,
    max = nil,
    features = 784,
  },
  tm = {
    flat = true,
    classes = 10,
    clauses = { def = 8, min = 1, max = 16, int = true },
    clause_maximum_fraction = { def = 0.14 },
    clause_tolerance_fraction = { def = 0.082 },
    target_fraction = { def = 0.12 },
    specificity_fraction = { def = 0.0108 },
    alpha_tolerance = { def = -2.5, min = -3, max = 3 },
    alpha_maximum = { def = -0.1, min = -3, max = 3 },
    alpha_target = { def = -0.1, min = -3, max = 3 },
    alpha_specificity = { def = 0.7, min = -3, max = 3 },
  },
  search = {
    trials = 200,
    iterations = 40,
    subsample = 0.2,
  },
  training = {
    patience = 8,
    batch = 40,
    iterations = 800,
  },
}

test("mnist classifier", function ()

  print("Reading data")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max)
  print("Splitting")
  local train, test_set, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test_set.n)

  print("\nConverting to TM representation")
  local train_tok_off, train_tok_nbr = csr_m.subsample(dataset.problem_offsets, dataset.problem_neighbors, train.ids)

  local val_tok_off, val_tok_nbr = csr_m.subsample(dataset.problem_offsets, dataset.problem_neighbors, validate.ids)
  validate.tokens = csr.to_bits(val_tok_off, val_tok_nbr, validate.n, cfg.data.features)

  local test_tok_off, test_tok_nbr = csr_m.subsample(dataset.problem_offsets, dataset.problem_neighbors, test_set.ids)
  test_set.tokens = csr.to_bits(test_tok_off, test_tok_nbr, test_set.n, cfg.data.features)

  local df_ids, df_scores = csr_m.top_df(train.sol_offsets, train.sol_neighbors, cfg.tm.classes)
  local output_weights = dvec.create():copy(df_scores, df_ids, true)

  local sol_offsets, sol_neighbors = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors

  print("\nTraining")
  local stopwatch = utc.stopwatch()
  local t = optimize.regressor({

    features = cfg.data.features,
    outputs = cfg.tm.classes,
    n_tokens = cfg.data.features,

    samples = train.n,
    token_offsets = train_tok_off,
    token_neighbors = train_tok_nbr,
    sol_offsets = sol_offsets,
    sol_neighbors = sol_neighbors,

    output_weights = output_weights,

    flat = cfg.tm.flat,
    clauses = cfg.tm.clauses,
    clause_maximum_fraction = cfg.tm.clause_maximum_fraction,
    clause_tolerance_fraction = cfg.tm.clause_tolerance_fraction,
    target_fraction = cfg.tm.target_fraction,
    specificity_fraction = cfg.tm.specificity_fraction,
    alpha_tolerance = cfg.tm.alpha_tolerance,
    alpha_maximum = cfg.tm.alpha_maximum,
    alpha_target = cfg.tm.alpha_target,
    alpha_specificity = cfg.tm.alpha_specificity,

    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample = cfg.search.subsample,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (regressor)
      local micro_f1, sample_f1 = regressor:label_f1(
        { tokens = validate.tokens, n_samples = validate.n },
        validate.n, val_label_off, val_label_nbr)
      return sample_f1, { micro_f1 = micro_f1, sample_f1 = sample_f1 }
    end,

    each = util.make_labeler_log(stopwatch)

  })

  print()
  print("Persisting")
  fs.rm("regressor.bin", true)
  t:persist("regressor.bin")

  print("Testing restore")
  t = tm.load("regressor.bin")

  print("\nClassification metrics:")
  train.tokens = csr.to_bits(train_tok_off, train_tok_nbr, train.n, cfg.data.features)
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
    str.printf("  digit_%-2d  F1=%.2f  P=%.2f  R=%.2f\n", c - 1, ts.f1, ts.precision, ts.recall)
  end

end)
