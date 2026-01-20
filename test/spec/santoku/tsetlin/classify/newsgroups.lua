local arr = require("santoku.array")
local str = require("santoku.string")
local test = require("santoku.test")
local ds = require("santoku.tsetlin.dataset")
local eval = require("santoku.tsetlin.evaluator")
local ivec = require("santoku.ivec")
local optimize = require("santoku.tsetlin.optimize")
local tokenizer = require("santoku.tokenizer")
local utc = require("santoku.utc")

local cfg = {
  data = {
    max_per_class = nil,
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
    algo = "chi2",
    top_k = 16384,
    individualized = true,
  },
  tm = {
    classes = 20,
    clauses = { def = 72, min = 8, max = 256, round = 8 },
    clause_tolerance = { def = 61, min = 16, max = 128, int = true },
    clause_maximum = { def = 97, min = 16, max = 128, int = true },
    target = { def = 51, min = 16, max = 128, int = true },
    specificity = { def = 439, min = 400, max = 4000 },
    include_bits = { def = 2, min = 1, max = 4, int = true },
  },
  search = {
    patience = 6,
    rounds = 0,
    trials = 10,
    iterations = 20,
  },
  training = {
    patience = 20,
    iterations = 400,
  },
  threads = nil,
}

test("tsetlin", function ()

  print("Reading data")
  local train, test, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max_per_class,
    nil,
    cfg.data.tvr)

  print("Train", train.n)
  print("Validate", validate.n)
  print("Test", test.n)

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
  local use_ind = cfg.feature_selection.individualized

  if use_ind then
    str.printf("\nPer-class chi2 feature selection (individualized=%s)\n", tostring(use_ind))
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
  else
    local top_v, chi2_weights
    if cfg.feature_selection.algo == "chi2" then
      top_v, chi2_weights = train.problems0:bits_top_chi2(train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
    else
      top_v, chi2_weights = train.problems0:bits_top_mi(train.solutions, train.n, n_features, cfg.tm.classes, cfg.feature_selection.top_k)
    end
    train.solutions:add_scaled(-cfg.tm.classes)
    n_top_v = top_v:size()
    print("After top k filter", n_top_v)

    local words = tok:index()
    print("\nTop 30 by Chi2 score:")
    for i = 0, 29 do
      local id = top_v:get(i)
      local score = chi2_weights:get(i)
      str.printf("  %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
    end

    print("\nBottom 30 (of selected subset) by Chi2 score:")
    for i = 0, 29 do
      local id = top_v:get(n_top_v - i - 1)
      local score = chi2_weights:get(n_top_v - i - 1)
      str.printf("  %6d  %-24s  %.4f\n", id, words[id + 1] or "?", score)
    end

    print("\nSelecting top features and converting to cvec")
    local train_selected = ivec.create()
    train.problems0:bits_select(top_v, nil, n_features, train_selected)
    train.problems = train_selected:bits_to_cvec(train.n, n_top_v, true)
    train_selected:destroy()
    train.problems0:destroy()

    local validate_tokens = tok:tokenize(validate.problems)
    local validate_selected = ivec.create()
    validate_tokens:bits_select(top_v, nil, n_features, validate_selected)
    validate.problems = validate_selected:bits_to_cvec(validate.n, n_top_v, true)
    validate_tokens:destroy()
    validate_selected:destroy()

    local test_tokens = tok:tokenize(test.problems)
    local test_selected = ivec.create()
    test_tokens:bits_select(top_v, nil, n_features, test_selected)
    test.problems = test_selected:bits_to_cvec(test.n, n_top_v, true)
    test_tokens:destroy()
    test_selected:destroy()
  end
  tok:destroy()

  print("Optimizing Classifier")
  local stopwatch = utc.stopwatch()
  local t = optimize.classifier({

    features = n_top_v,
    individualized = use_ind,
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
    final_iterations = cfg.training.iterations,

    search_metric = function (t0, _)
      local predicted
      if validate_dim_offsets then
        predicted = t0:predict(validate.problems, validate_dim_offsets, validate.n, cfg.threads)
      else
        predicted = t0:predict(validate.problems, validate.n, cfg.threads)
      end
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
  local train_pred, val_pred, test_pred
  if use_ind then
    train_pred = t:predict(train.problems, train_dim_offsets, train.n, cfg.threads)
    val_pred = t:predict(validate.problems, validate_dim_offsets, validate.n, cfg.threads)
    test_pred = t:predict(test.problems, test_dim_offsets, test.n, cfg.threads)
  else
    train_pred = t:predict(train.problems, train.n, cfg.threads)
    val_pred = t:predict(validate.problems, validate.n, cfg.threads)
    test_pred = t:predict(test.problems, test.n, cfg.threads)
  end
  local train_stats = eval.class_accuracy(train_pred, train.solutions, train.n, cfg.tm.classes, cfg.threads)
  local val_stats = eval.class_accuracy(val_pred, validate.solutions, validate.n, cfg.tm.classes, cfg.threads)
  local test_stats = eval.class_accuracy(test_pred, test.solutions, test.n, cfg.tm.classes, cfg.threads)
  str.printf("Evaluate\tTrain\t%4.2f\tVal\t%4.2f\tTest\t%4.2f\n", train_stats.f1, val_stats.f1, test_stats.f1)

  print("\nPer-class Test Accuracy (sorted by difficulty):\n")
  local class_order = arr.range(1, cfg.tm.classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    local cat = train.categories[c] or ("class_" .. (c - 1))
    str.printf("  %-28s  F1=%.2f  P=%.2f  R=%.2f\n", cat, ts.f1, ts.precision, ts.recall)
  end

end)
