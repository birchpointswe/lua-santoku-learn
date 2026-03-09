local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local hdc = require("santoku.learn.hdc")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { ttr = 0.8, tvr = 0.10, max = nil, n_thresholds = 256 },
  hdc = { d = 2^17, n_levels = 256 },
  emb = { n_landmarks = 1024, trace_tol = 0.01, cholesky = true, n_dims = nil },
  ridge = { lambda = { def = 6.3679e-05 }, search_trials = 800 },
}

test("housing regressor", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_california_housing("test/res/california-housing.csv", {
    n_thresholds = cfg.data.n_thresholds,
    max = cfg.data.max,
  })
  local train, test_set, validate = ds.split_california_housing(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_features = dataset.n_features
  local n_cont = train.n_continuous
  str.printf("[Data] train=%d val=%d test=%d features=%d cont=%d target=%.0f-%.0f %s\n",
    train.n, validate.n, test_set.n, n_features, n_cont,
    train.targets:min(), train.targets:max(), sw())

  local d = cfg.hdc.d
  local d_total = d * 2

  str.printf("\n[Cat] HDC bits d=%d\n", d)
  local cat_bmp = train.bits:bits_to_cvec(train.n, n_features)
  local cat_enc, train_cvec = hdc.create({
    bits = cat_bmp, n_dims = n_features,
    n_samples = train.n, d = d, hdc_ngram = 1,
  })
  str.printf("[Cat] HDC done %s\n", sw())

  str.printf("[Cont] HDC dense d=%d\n", d)
  local cont_enc, cont_cvec = hdc.create({
    codes = train.continuous, n_dims = n_cont,
    n_samples = train.n, d = d, hdc_ngram = 1,
    n_levels = cfg.hdc.n_levels,
  })
  str.printf("[Cont] HDC done %s\n", sw())

  str.printf("[Concat] bits_extend %d+%d=%d\n", d, d, d_total)
  train_cvec:bits_extend(cont_cvec, d, d)
  cont_cvec = nil -- luacheck: ignore
  str.printf("[Concat] done %s\n", sw())

  local function encode(bits, continuous, n)
    local bmp = bits:bits_to_cvec(n, n_features)
    local cat = cat_enc:encode({ bits = bmp, n_samples = n })
    local cont = cont_enc:encode({ codes = continuous, n_samples = n })
    cat:bits_extend(cont, d, d)
    return cat
  end

  str.printf("[Spectral] Cholesky trace_tol=%s\n", tostring(cfg.emb.trace_tol))
  local _, _, sp_enc, _, xtx, xty, col_mean, y_mean, _, pre_mean, pre_istd =
    spectral.encode({
      bits = train_cvec, n_samples = train.n, d_bits = d_total,
      n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
      cholesky = cfg.emb.cholesky, n_dims = cfg.emb.n_dims,
      targets = train.targets, n_targets = 1,
    })
  local emb_d = sp_enc:dims()
  train_cvec = nil -- luacheck: ignore
  local val_codes = sp_enc:encode(encode(validate.bits, validate.continuous, validate.n), validate.n)
  str.printf("[Spectral] emb_d=%d %s\n", emb_d, sw())

  str.printf("[Ridge] Training\n")
  local _, ridge_obj, best_params, _, _, _, std = optimize.ridge({
    XtX = xtx, XtY = xty, col_mean = col_mean, y_mean = y_mean,
    pre_mean = pre_mean, pre_istd = pre_istd,
    n_samples = train.n, n_dims = emb_d, n_targets = 1,
    targets = true,
    val_codes = val_codes, val_n_samples = validate.n,
    val_targets = validate.targets,
    lambda = cfg.ridge.lambda,
    search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  str.printf("[Ridge] lambda=%.4e %s\n", best_params.lambda, sw())

  str.printf("\n[Eval] Scoring splits\n")
  local val_stats = eval.regression_accuracy(ridge_obj:regress(val_codes, validate.n), validate.targets)
  local test_codes = sp_enc:encode(encode(test_set.bits, test_set.continuous, test_set.n), test_set.n)
  test_codes:mtx_standardize(emb_d, std.pre_mean, std.pre_istd)
  local test_stats = eval.regression_accuracy(ridge_obj:regress(test_codes, test_set.n), test_set.targets)
  str.printf("[Eval] Accuracy: val=%.1f%% test=%.1f%% %s\n",
    (1 - val_stats.nmae) * 100, (1 - test_stats.nmae) * 100, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
