local csr = require("santoku.learn.csr")
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

io.stdout:setvbuf("line")

local cfg = {
  data = {
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
    n_selected = 8192,
  },
  tm = {
    flat = true,
    flat_evict = true,
    cost_beta = nil,
    features = 4096,
    clauses = 32,
    clause_maximum_fraction = { def = 0.03 },
    clause_tolerance_fraction = { def = 0.57 },
    specificity_fraction = { def = 0.00095 },
    absorb_threshold = { def = 9 },
    absorb_maximum_fraction = { def = 0.029 },
    absorb_insert_offset = { def = 49 },
    absorb_ranking_fraction = { def = 0.125 },
    target_fraction = { def = 0.10 },
  },
  search = {
    trials = 200,
    iterations = 20,
    subsample = 0.05,
  },
  training = {
    patience = 2,
    batch = 40,
    iterations = 800,
  },
}

test("eurlex-direct", function ()

  local stopwatch = utc.stopwatch()

  print("Loading data")
  local train, dev = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  str.printf("  Train: %d  Dev: %d  Labels: %d\n", train.n, dev.n, n_labels)

  print("\nBuilding label CSR")
  local sol_offsets, sol_neighbors = train.solutions:bits_to_csr(train.n, n_labels)
  local dev_label_off, dev_label_nbr = dev.solutions:bits_to_csr(dev.n, n_labels)

  print("\nTokenizing")
  local tok = tokenizer.create(cfg.tokenizer)
  tok:train({ corpus = train.problems })
  tok:finalize()
  local n_tokens = tok:features()
  train.tokens = tok:tokenize(train.problems)
  dev.tokens = tok:tokenize(dev.problems)
  tok = nil -- luacheck: ignore
  train.problems = nil
  dev.problems = nil
  str.printf("  Vocabulary: %d\n", n_tokens)

  print("\nFeature ranking (Chi2)")
  local n_selected = cfg.feature_selection.n_selected
  local chi2_ranking, _, class_offsets, class_feat_ids = train.tokens:bits_top_chi2(
    train.solutions, train.n, n_tokens, n_labels,
    n_selected, nil, "sum")
  train.tokens:bits_select(chi2_ranking, nil, n_tokens)
  dev.tokens:bits_select(chi2_ranking, nil, n_tokens)
  class_offsets, class_feat_ids = csr.bits_select(class_offsets, class_feat_ids, chi2_ranking)
  n_tokens = chi2_ranking:size()
  str.printf("  Selected %d features\n", n_tokens)

  print("\nBuilding CSC index")
  local csc_offsets, csc_indices = csr.to_csc(train.tokens, train.n, n_tokens)
  str.printf("  Tokens: %d  Samples: %d\n", n_tokens, train.n)

  local absorb_ranking_global = ivec.create(n_tokens):fill_indices()

  local df_ids, df_scores = train.solutions:bits_top_df(train.n, n_labels)
  local output_weights = dvec.create():copy(df_scores, df_ids, true)

  print("\nOptimizing flat regressor")
  local t = optimize.regressor({
    flat = cfg.tm.flat,
    flat_evict = cfg.tm.flat_evict,
    cost_beta = cfg.tm.cost_beta,
    outputs = n_labels,
    samples = train.n,
    features = cfg.tm.features,
    n_tokens = n_tokens,
    absorb_threshold = cfg.tm.absorb_threshold,
    absorb_maximum_fraction = cfg.tm.absorb_maximum_fraction,
    absorb_insert_offset = cfg.tm.absorb_insert_offset,
    absorb_ranking_fraction = cfg.tm.absorb_ranking_fraction,
    clauses = cfg.tm.clauses,
    clause_maximum_fraction = cfg.tm.clause_maximum_fraction,
    clause_tolerance_fraction = cfg.tm.clause_tolerance_fraction,
    target_fraction = cfg.tm.target_fraction,
    specificity_fraction = cfg.tm.specificity_fraction,
    output_weights = output_weights,
    sol_offsets = sol_offsets,
    sol_neighbors = sol_neighbors,
    tokens = train.tokens,
    csc_offsets = csc_offsets,
    csc_indices = csc_indices,
    absorb_ranking = class_feat_ids,
    absorb_ranking_offsets = class_offsets,
    absorb_ranking_global = absorb_ranking_global,
    stratify_offsets = sol_offsets,
    stratify_neighbors = sol_neighbors,
    stratify_labels = n_labels,
    search_trials = cfg.search.trials,
    search_iterations = cfg.search.iterations,
    search_subsample = cfg.search.subsample,
    final_batch = cfg.training.batch,
    final_patience = cfg.training.patience,
    final_iterations = cfg.training.iterations,
    search_metric = function (regressor)
      local input = { tokens = dev.tokens, n_samples = dev.n }
      local micro_f1, macro_f1 = regressor:label_f1(input, dev.n, dev_label_off, dev_label_nbr)
      return macro_f1, { micro_f1 = micro_f1, macro_f1 = macro_f1 }
    end,
    each = util.make_labeler_log(stopwatch),
  })

  print("\n" .. string.rep("=", 60))
  print("FINAL EVALUATION")
  print(string.rep("=", 60))

  print("\nPredicting labels (train)")
  local train_off, train_labels, train_scores = t:label(
    { tokens = train.tokens, n_samples = train.n }, train.n, 32)
  local train_oracle, train_thresh = eval.retrieval_ks({
    pred_offsets = train_off, pred_neighbors = train_labels, pred_scores = train_scores,
    expected_offsets = sol_offsets, expected_neighbors = sol_neighbors,
  })

  print("\nPredicting labels (dev)")
  local dev_off, dev_labels, dev_scores = t:label(
    { tokens = dev.tokens, n_samples = dev.n }, dev.n, 32)
  local dev_oracle, dev_thresh = eval.retrieval_ks({
    pred_offsets = dev_off, pred_neighbors = dev_labels, pred_scores = dev_scores,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })

  str.printf("\n  Labels: %d\n", n_labels)
  str.printf("  %-40s %8s %8s %8s %8s\n",
    "", "micro F1", "macro F1", "orc miF1", "orc maF1")
  str.printf("  %-40s %8s %8s %8s %8s\n",
    string.rep("-", 40), "--------", "--------", "--------", "--------")
  str.printf("  %-40s %8.4f %8.4f %8.4f %8.4f\n",
    "Train (threshold)", train_thresh.micro_f1, train_thresh.macro_f1,
    train_oracle.micro_f1, train_oracle.macro_f1)
  str.printf("  %-40s %8.4f %8.4f %8.4f %8.4f\n",
    "Dev (threshold)", dev_thresh.micro_f1, dev_thresh.macro_f1,
    dev_oracle.micro_f1, dev_oracle.macro_f1)

  str.printf("\n  Time: %.1fs\n", stopwatch())

end)
