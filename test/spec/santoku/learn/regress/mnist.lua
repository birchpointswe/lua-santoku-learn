local csr_m = require("santoku.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { ttr = 0.8, tvr = 0.1, max = nil, features = 784 },
  -- n_landmarks = 1024*24 for max (high memory)
  emb = { n_landmarks = 1024*8, trace_tol = 0.01, kernel = { "expcos", "cosine", "nngp", "ntk", "geolaplace" } },
  ridge = {
    lambda = { def = 3.82e-03 },
    propensity_a = { def = 1.01 },
    propensity_b = { def = 3.61 },
    classes = 10,
    search_trials = 0,
    k = 1,
  },
}

test("mnist classifier", function ()

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

  local val_p_off, val_p_nbr = csr_m.subsample(
    dataset.problem_offsets, dataset.problem_neighbors, validate.ids)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, val_codes, best_params = optimize.krr({
    kernel = cfg.emb.kernel, offsets = train_p_off, tokens = train_p_nbr,
    n_samples = train.n, n_tokens = n_features,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    val_offsets = val_p_off, val_tokens = val_p_nbr,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  train_p_off = nil; train_p_nbr = nil -- luacheck: ignore
  collectgarbage("collect")
  local function encode(ids, n)
    local p_off, p_nbr = csr_m.subsample(
      dataset.problem_offsets, dataset.problem_neighbors, ids)
    return sp_enc:encode({
      offsets = p_off, tokens = p_nbr, n_samples = n,
    })
  end

  str.printf("[Eval] Labeling splits\n")
  local val_off, val_nbr = ridge_obj:label(val_codes, validate.n, 1)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode(test_set.ids, test_set.n)
  local test_off, test_nbr = ridge_obj:label(test_codes, test_set.n, 1)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  local _, val_stats = eval.label_accuracy({
    pred_offsets = val_off, pred_neighbors = val_nbr,
    expected_offsets = validate.sol_offsets, expected_neighbors = validate.sol_neighbors,
    ks = 1,
  })
  local _, test_stats = eval.label_accuracy({
    pred_offsets = test_off, pred_neighbors = test_nbr,
    expected_offsets = test_set.sol_offsets, expected_neighbors = test_set.sol_neighbors,
    ks = 1,
  })
  str.printf("[Class] F1: val=%.2f test=%.2f %s\n",
    val_stats.micro_f1, test_stats.micro_f1, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
