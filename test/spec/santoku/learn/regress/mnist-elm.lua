local arr = require("santoku.array")
local csr_m = require("santoku.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { ttr = 0.8, tvr = 0.1, max = nil, features = 784 },
  emb = { n_landmarks = 8192, trace_tol = 0.01, kernel = "arccos1" },
  ridge = {
    lambda = { def = 2.3163e-05 },
    propensity_a = { def = 0.55 },
    propensity_b = { def = 1.5 },
    classes = 10,
    search_trials = 0,
    k = 1,
  },
}

test("mnist csr+pos2d", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_binary_mnist("test/res/mnist.70k.txt", cfg.data.features, cfg.data.max)
  local train, test_set, validate = ds.split_binary_mnist(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_features = cfg.data.features
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors
  str.printf("[Data] train=%d val=%d test=%d features=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_features, n_classes, sw())

  local train_p_off, train_p_nbr = csr_m.subsample(
    dataset.problem_offsets, dataset.problem_neighbors, train.ids)

  str.printf("[Spectral] Cholesky trace_tol=%s\n",
    tostring(cfg.emb.trace_tol))
  local train_codes, sp_enc = spectral.encode({
    kernel = cfg.emb.kernel, offsets = train_p_off, tokens = train_p_nbr,
    n_samples = train.n, n_tokens = n_features,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
  })
  train_p_off = nil; train_p_nbr = nil -- luacheck: ignore
  collectgarbage("collect")
  local emb_d = sp_enc:dims()
  str.printf("[Spectral] emb_d=%d %s\n", emb_d, sw())

  local function encode(ids, n)
    local p_off, p_nbr = csr_m.subsample(
      dataset.problem_offsets, dataset.problem_neighbors, ids)
    return sp_enc:encode({
      offsets = p_off, tokens = p_nbr, n_samples = n,
    })
  end

  local val_codes = encode(validate.ids, validate.n)

  str.printf("[Ridge] Training\n")
  local ridge_obj, best_params = optimize.ridge({
    train_codes = train_codes, n_samples = train.n, n_dims = emb_d,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    val_codes = val_codes, val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best_params.lambda, best_params.propensity_a, best_params.propensity_b, sw())

  str.printf("[Eval] Labeling splits\n")
  local val_off, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  local test_codes = encode(test_set.ids, test_set.n)
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
  str.printf("[Eval] Labels done %s\n", sw())

  local _, train_labels = ridge_obj:label(train_codes, train.n, 1)
  local train_stats = eval.class_accuracy(train_labels, train.sol_offsets, train.sol_neighbors, train.n, n_classes)
  local val_stats = eval.class_accuracy(val_labels, validate.sol_offsets, validate.sol_neighbors, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.sol_offsets, test_set.sol_neighbors, test_set.n, n_classes)
  str.printf("[Class] F1: train=%.2f val=%.2f test=%.2f %s\n",
    train_stats.f1, val_stats.f1, test_stats.f1, sw())

  local _, val_oracle = eval.retrieval_ks({
    pred_offsets = val_off, pred_neighbors = val_labels,
    expected_offsets = val_label_off, expected_neighbors = val_label_nbr,
  })
  str.printf("[Retrieval] val: saF1=%.4f miF1=%.4f %s\n",
    val_oracle.sample_f1, val_oracle.micro_f1, sw())

  str.printf("\n[Per-class Test Accuracy]\n")
  local class_order = arr.range(1, n_classes)
  arr.sort(class_order, function (a, b)
    return test_stats.classes[a].f1 < test_stats.classes[b].f1
  end)
  for _, c in ipairs(class_order) do
    local ts = test_stats.classes[c]
    str.printf("  digit_%-2d  F1=%.2f  P=%.2f  R=%.2f\n", c - 1, ts.f1, ts.precision, ts.recall)
  end

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
