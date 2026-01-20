local str = require("santoku.string")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local optimize = require("santoku.tsetlin.optimize")
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
    top_k = 16384,
    individualized = true,
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
    patience = 6,
    rounds = 0,
    trials = 10,
    iterations = 40,
  },
  training = {
    patience = 40,
    iterations = 400,
  },
  threads = nil,
}

test("tsetlin", function ()

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

  local n_top_v, feat_offsets, train_dim_offsets, validate_dim_offsets, test_dim_offsets

  str.printf("\nPer-class chi2 feature selection (individualized=%s)\n", tostring(cfg.feature_selection.individualized))
  local ids_union, feat_offs, feat_ids = train.problems0:bits_top_chi2_ind(
    train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
  feat_offsets = feat_offs
  local union_size = ids_union:size()
  local total_features = feat_offsets:get(cfg.tm.classes)
  str.printf("  Per-class chi2: union=%d total=%d (%.1fx expansion)\n",
    union_size, total_features, total_features / union_size)
  train.solutions:add_scaled(-cfg.tm.classes)
  train.problems0 = nil
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

  train.problems, train_dim_offsets = to_ind_bitmap(train)
  validate.problems, validate_dim_offsets = to_ind_bitmap(validate)
  test.problems, test_dim_offsets = to_ind_bitmap(test)
  n_top_v = union_size
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
      local predicted = t0:predict(validate.problems, validate_dim_offsets, validate.n, cfg.threads)
      local accuracy = eval.class_accuracy(predicted, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
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
  local train_pred = t:predict(train.problems, train_dim_offsets, train.n, cfg.threads)
  local val_pred = t:predict(validate.problems, validate_dim_offsets, validate.n, cfg.threads)
  local test_pred = t:predict(test.problems, test_dim_offsets, test.n, cfg.threads)
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes, cfg.threads)
  local val_stats = eval.class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes, cfg.threads)
  str.printf("Evaluate\tTrain\t%4.2f\tVal\t%4.2f\tTest\t%4.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

end)
