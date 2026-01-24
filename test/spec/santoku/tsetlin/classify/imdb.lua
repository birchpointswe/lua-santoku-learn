local arr = require("santoku.array")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")

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
    max_vocab = 16384,
  },
  tm = {
    classes = 2,
    clauses = { def = 48, min = 8, max = 256, round = 8 },
    clause_tolerance = { def = 35, min = 16, max = 128, int = true },
    clause_maximum = { def = 93, min = 16, max = 128, int = true },
    target = { def = 50, min = 16, max = 128, int = true },
    specificity = { def = 941, min = 400, max = 4000 },
    include_bits = { def = 4, min = 1, max = 4, int = true },
  },
  search = {
    patience = 4,
    rounds = 4,
    trials = 10,
    iterations = 10,
  },
  training = {
    patience = 40,
    iterations = 400,
  },
}

test("imdb", function ()

  print("Reading data")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)

  str.printf("  Train:    %6d\n", train.n)
  str.printf("  Validate: %6d\n", validate.n)
  str.printf("  Test:     %6d\n", test.n)

  print("\nTraining tokenizer\n")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_features = tok:features()
  str.printf("Feat\t\t%d\t\t\n", n_features)

  print("Tokenizing train")
  train.problems0 = tok:tokenize(train.problems)
  train.solutions:add_scaled(cfg.tm.classes)

  print("\nPer-class Chi2 feature selection")
  local ids_union, feat_offsets, feat_ids = train.problems0:bits_top_chi2_ind(
    train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.max_vocab)
  local n_top_v = ids_union:size()
  local total_features = feat_offsets:get(cfg.tm.classes)
  str.printf("  Per-class Chi2: union=%d total=%d (%.1fx expansion)\n",
    n_top_v, total_features, total_features / n_top_v)
  train.solutions:add_scaled(-cfg.tm.classes)
  train.problems0 = nil
  tok:restrict(ids_union)

  local function to_ind_bitmap (split)
    local toks = tok:tokenize(split.problems)
    local ind, ind_off = toks:bits_individualize(feat_offsets, feat_ids, n_top_v)
    local bitmap, dim_off = ind:bits_to_cvec_ind(ind_off, feat_offsets, split.n, true)
    toks:destroy()
    ind:destroy()
    ind_off:destroy()
    return bitmap, dim_off
  end

  local train_dim_offsets, validate_dim_offsets, test_dim_offsets
  train.problems, train_dim_offsets = to_ind_bitmap(train)
  validate.problems, validate_dim_offsets = to_ind_bitmap(validate)
  test.problems, test_dim_offsets = to_ind_bitmap(test)
  tok:destroy()

  print("Optimizing Classifier")
  local stopwatch = utc.stopwatch()
  local t = optimize.classifier({

    features = n_top_v,
    individualized = true,
    feat_offsets = feat_offsets,
    dim_offsets = train_dim_offsets,

    classes = cfg.tm.classes,
    clauses = cfg.tm.clauses,
    clause_tolerance = cfg.tm.clause_tolerance,
    clause_maximum = cfg.tm.clause_maximum,
    target = cfg.tm.target,
    specificity = cfg.tm.specificity,
    include_bits = cfg.tm.include_bits,

    samples = train.n,
    problems = train.problems,
    solutions = train.solutions,

    search_patience = cfg.search.patience,
    search_rounds = cfg.search.rounds,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,

    search_metric = function (t0, _)
      local predicted = t0:predict(validate.problems, validate_dim_offsets, validate.n)
      local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.tm.classes)
      return accuracy.f1, accuracy
    end,

    each = function (_, is_final, val_accuracy, params, epoch, round, trial)
      local d, dd = stopwatch()
      local phase = is_final and "F" or str.format("R%d T%d", round, trial)
      str.printf("[CLASSIFY %s E%d] C=%d L=%d/%d T=%d S=%.0f IB=%d F1=%.2f (%.2fs +%.2fs)\n",
        phase, epoch, params.clauses, params.clause_tolerance, params.clause_maximum,
        params.target, params.specificity, params.include_bits, val_accuracy.f1, d, dd)
    end

  })

  print()
  print("Final Evaluation")
  local train_pred = t:predict(train.problems, train_dim_offsets, train.n)
  local val_pred = t:predict(validate.problems, validate_dim_offsets, validate.n)
  local test_pred = t:predict(test.problems, test_dim_offsets, test.n)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes)
  local val_stats = eval.class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes)
  str.printf("Evaluate\tTrain\t%4.2f\tVal\t%4.2f\tTest\t%4.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_names = { "negative", "positive" }
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  %-12s  F1=%.2f  P=%.2f  R=%.2f\n", class_names[c], ts.f1, ts.precision, ts.recall)
  end

end)
