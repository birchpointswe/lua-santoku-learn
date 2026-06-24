local ds = require("santoku.learn.dataset")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = {
    ttr = 0.8,
    tvr = 0.1,
    max = nil,
    features = 784
  },
  emb = {
    n_landmarks = 1024 * 8,
    kernel = { "matern", "cosine", "arccos" },
    nu = { def = 3 },
    gamma = { def = 2.1 }
  },
  ridge = {
    lambda = { def = 1.6260e-04 },
    propensity_a = { def = 0.9448 },
    propensity_b = { def = 8.8585 },
    classes = 10,
    search_trials = 0,
    k = 1
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
  str.printf("[Data] train=%d val=%d test=%d features=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_features, n_classes, sw())

  local function subsample (ids)
    local Y = dataset.problems:rows(ids)
    Y:normalize()
    return Y
  end

  local Xtr = subsample(train.ids)
  local Xval = subsample(validate.ids)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, val_codes, _, decider = optimize.krr({
    kernel = cfg.emb.kernel, nu = cfg.emb.nu, gamma = cfg.emb.gamma,
    x = Xtr, y = train.labels, val_x = Xval, val_y = validate.labels,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  do
    local p = os.tmpname()
    sp_enc:persist(p)
    sp_enc = spectral.load(p)
    os.remove(p)
  end
  Xtr = nil; Xval = nil -- luacheck: ignore
  collectgarbage("collect")
  local function encode(ids)
    return sp_enc:encode(subsample(ids))
  end

  str.printf("[Eval] Labeling splits\n")
  local val_scores = ridge_obj:regress(val_codes)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode(test_set.ids)
  local test_scores = ridge_obj:regress(test_codes)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  local decide = require("santoku.learn.decide")
  local argmax = decide.create({ n_labels = n_classes, single = true })
  local function score_split (d, scores, n, sol)
    local _, m = d:score({ scores = scores, n_samples = n, expected = sol.labels })
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
