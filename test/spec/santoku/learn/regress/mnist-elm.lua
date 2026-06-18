local csr_m = require("santoku.csr")
local csr = require("santoku.learn.csr")
local ds = require("santoku.learn.dataset")
local optimize = require("santoku.learn.optimize")
local elm = require("santoku.learn.elm")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- Reported metrics (search_trials=100; splits train=50400 val=5600 test=14000; 10 classes):
--   n_landmarks=8192:  F1 val=0.98 test=0.98  (best: matern52, lambda=5.258e-03, pa=0.45 pb=6.79)

local cfg = {
  data = { ttr = 0.8, tvr = 0.1, max = nil, features = 784 },
  emb = { n_hidden = 1024 * 8 },
  ridge = {
    mode = { "relu", "linear" },
    -- mode = { "rbf" },
    lambda = { def = 1.6772e-03 },
    propensity_a = { def = 1.9288 },
    propensity_b = { def = 7.8035 },
    gamma = { def = 1.2 },
    classes = 10,
    search_trials = 0,
    k = 1,
  },
}

test("mnist-elm classifier", function ()

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
  local train_p_val = csr.normalize(train_p_off)

  local val_p_off, val_p_nbr = csr_m.subsample(
    dataset.problem_offsets, dataset.problem_neighbors, validate.ids)
  local val_p_val = csr.normalize(val_p_off)

  str.printf("[ELM] Encoding n_hidden=%d\n", cfg.emb.n_hidden)
  local sp_enc, ridge_obj, val_codes, _, decider = optimize.elm({
    offsets = train_p_off, tokens = train_p_nbr, values = train_p_val,
    n_samples = train.n, n_tokens = n_features,
    n_hidden = cfg.emb.n_hidden,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    val_offsets = val_p_off, val_tokens = val_p_nbr, val_values = val_p_val,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    gamma = cfg.ridge.gamma,
    mode = cfg.ridge.mode,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  do  -- persist/load parity: round-tripped encoder must produce identical codes
    local p = os.tmpname()
    sp_enc:persist(p)
    local enc2 = elm.load(p)
    os.remove(p)
    local vc2 = enc2:encode({ offsets = val_p_off, tokens = val_p_nbr, values = val_p_val, n_samples = validate.n })
    local nchk = val_codes:size()
    if nchk > 100000 then nchk = 100000 end
    for i = 0, nchk - 1 do
      assert(val_codes:get(i) == vc2:get(i), "persist/load parity mismatch at " .. i)
    end
    str.printf("[Persist] load parity OK (%d codes)\n", nchk)
  end
  train_p_off = nil; train_p_nbr = nil -- luacheck: ignore
  collectgarbage("collect")
  local function encode(ids, n)
    local p_off, p_nbr = csr_m.subsample(
      dataset.problem_offsets, dataset.problem_neighbors, ids)
    local p_val = csr.normalize(p_off)
    return sp_enc:encode({
      offsets = p_off, tokens = p_nbr, values = p_val, n_samples = n,
    })
  end

  str.printf("[Eval] Labeling splits\n")
  local val_scores = ridge_obj:regress(val_codes, validate.n)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode(test_set.ids, test_set.n)
  local test_scores = ridge_obj:regress(test_codes, test_set.n)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  -- single-label decode is argmax(score - offset). `decider` carries the dev-calibrated offsets; a
  -- zero-offset decider is plain argmax. Score both decoders on both splits: decide can only improve
  -- dev (argmax is the offsets==0 special case it searched over), so test is where overfit would show.
  local decide = require("santoku.learn.decide")
  local argmax = decide.create({ n_labels = n_classes, single = true })
  local function score_split (d, scores, n, sol)
    local _, m = d:score({ scores = scores, n_samples = n,
      expected_offsets = sol.sol_offsets, expected_neighbors = sol.sol_neighbors })
    return m
  end
  str.printf("[Argmax] dev %s | test %s %s\n",
    util.fmt_metrics(score_split(argmax, val_scores, validate.n, validate)),
    util.fmt_metrics(score_split(argmax, test_scores, test_set.n, test_set)), sw())
  str.printf("[Decide] dev %s | test %s %s\n",
    util.fmt_metrics(score_split(decider, val_scores, validate.n, validate)),
    util.fmt_metrics(score_split(decider, test_scores, test_set.n, test_set)), sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
