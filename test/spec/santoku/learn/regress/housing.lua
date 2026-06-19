local mtx = require("santoku.mtx")
local ivec = require("santoku.ivec")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local fvec = require("santoku.fvec")
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
    max = nil
  },
  emb = {
    n_landmarks = 1024 * 8,
    kernel = { "matern", "cosine", "arccos" },
    nu = { def = 1 },
    gamma = { def = 3.872 }
  },
  ridge = {
    lambda = { def = 7.0119e-02 },
    search_trials = 0
  },
}

test("housing regressor", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_california_housing("test/res/california-housing.csv", {
    max = cfg.data.max,
  })
  local train, test_set, validate = ds.split_california_housing(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_cat = dataset.n_features
  local n_cont = train.n_continuous
  local cont_mean = mtx.create({ data = train.continuous, n_rows = train.n, n_cols = n_cont }):center()
  mtx.create({ data = validate.continuous, n_rows = validate.n, n_cols = n_cont }):center(cont_mean)
  mtx.create({ data = test_set.continuous, n_rows = test_set.n, n_cols = n_cont }):center(cont_mean)
  str.printf("[Data] train=%d val=%d test=%d cat=%d cont=%d target=%.0f-%.0f %s\n",
    train.n, validate.n, test_set.n, n_cat, n_cont,
    train.targets:min(), train.targets:max(), sw())

  local function features(bits, continuous, n)
    local cont = mtx.create({ data = continuous:to_fvec(), n_rows = n, n_cols = n_cont }):to_sparse()
    return bits:hcat(cont)
  end

  local n_tokens = n_cat + n_cont
  local X = features(train.bits, train.continuous, train.n)
  local std_scores = X:standardize()
  local ss = X:sumsq_cols(ivec.create({ 0, n_cat, n_tokens }))
  local ss_cat, ss_cont = ss:get(0), ss:get(1)
  local block = fvec.create(n_tokens)
  block:fill(ss_cat > 0 and math.sqrt(train.n / ss_cat) or 0.0, 0, n_cat)
  block:fill(ss_cont > 0 and math.sqrt(train.n / ss_cont) or 0.0, n_cat, n_tokens)
  X:standardize(block)
  std_scores:scalev(block)
  X:normalize()

  local Xv = features(validate.bits, validate.continuous, validate.n)
  Xv:standardize(std_scores)
  Xv:normalize()

  str.printf("[KRR] Encoding n_landmarks=%d n_tokens=%d\n",
    cfg.emb.n_landmarks, n_tokens)
  local sp_enc, ridge_obj, val_codes = optimize.krr({
    x = X, val_x = Xv,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    kernel = cfg.emb.kernel, nu = cfg.emb.nu, gamma = cfg.emb.gamma,
    targets = train.targets, n_targets = 1,
    val_targets = validate.targets,
    lambda = cfg.ridge.lambda,
    search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  do
    local p = os.tmpname()
    sp_enc:persist(p)
    sp_enc = spectral.load(p)   -- continue (and re-encode test) with the reloaded encoder
    os.remove(p)
  end
  X = nil; Xv = nil -- luacheck: ignore
  collectgarbage("collect")
  local function encode(bits, continuous, n)
    local Xt = features(bits, continuous, n)
    Xt:standardize(std_scores)
    Xt:normalize()
    return sp_enc:encode(Xt)
  end

  str.printf("[Eval] Scoring splits\n")
  local regress_buf = fvec.create()
  local val_stats = eval.regress_accuracy(ridge_obj:regress(val_codes, validate.n, regress_buf), validate.targets)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode(test_set.bits, test_set.continuous, test_set.n)
  local test_stats = eval.regress_accuracy(ridge_obj:regress(test_codes, regress_buf), test_set.targets)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Accuracy: val=%.1f%% test=%.1f%% %s\n",
    (1 - val_stats.nmae) * 100, (1 - test_stats.nmae) * 100, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
