local csr_m = require("santoku.csr")
local ds = require("santoku.learn.dataset")
local dvec = require("santoku.dvec")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { ttr = 0.8, tvr = 0.10, max = nil },
  cat_emb = { n_landmarks = 4096, trace_tol = 0.01, kernel = "arccos1" },
  cont_emb = { n_landmarks = 4096, trace_tol = 0.01, kernel = "arccos1" },
  ridge = { lambda = { def = 1.5612e-02 }, search_trials = 0 },
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
  str.printf("[Data] train=%d val=%d test=%d cat=%d cont=%d target=%.0f-%.0f %s\n",
    train.n, validate.n, test_set.n, n_cat, n_cont,
    train.targets:min(), train.targets:max(), sw())

  str.printf("[Spectral Cat] Cholesky kernel=%s\n", cfg.cat_emb.kernel)
  local train_cat_dv = csr_m.to_dvec(train.bit_offsets, train.bit_neighbors, nil, train.n, n_cat)
  local train_cat_codes, cat_enc = spectral.encode({
    codes = train_cat_dv, n_samples = train.n,
    kernel = cfg.cat_emb.kernel,
    n_landmarks = cfg.cat_emb.n_landmarks, trace_tol = cfg.cat_emb.trace_tol,
  })
  train_cat_dv = nil -- luacheck: ignore
  collectgarbage("collect")
  local cat_d = cat_enc:dims()
  str.printf("[Spectral Cat] emb_d=%d %s\n", cat_d, sw())

  str.printf("[Spectral Cont] Cholesky kernel=%s\n", cfg.cont_emb.kernel)
  local train_cont_codes, cont_enc = spectral.encode({
    codes = train.continuous, n_samples = train.n,
    kernel = cfg.cont_emb.kernel,
    n_landmarks = cfg.cont_emb.n_landmarks, trace_tol = cfg.cont_emb.trace_tol,
  })
  local cont_d = cont_enc:dims()
  str.printf("[Spectral Cont] emb_d=%d %s\n", cont_d, sw())

  local emb_d = cat_d + cont_d
  local function concat_codes(cat_codes, cont_codes, n)
    local out = dvec.create(n * emb_d)
    for i = 0, n - 1 do
      out:copy(cat_codes, i * cat_d, (i + 1) * cat_d, i * emb_d)
      out:copy(cont_codes, i * cont_d, (i + 1) * cont_d, i * emb_d + cat_d)
    end
    return out
  end

  local train_codes = concat_codes(train_cat_codes, train_cont_codes, train.n)
  train_cat_codes = nil; train_cont_codes = nil -- luacheck: ignore
  collectgarbage("collect")

  local xtx, xty, col_mean, y_mean = spectral.gram({
    codes = train_codes, n_samples = train.n, n_dims = emb_d,
    targets = train.targets, n_targets = 1,
  })

  local cat_sims_buf = dvec.create()
  local cat_out_buf = dvec.create()
  local cont_sims_buf = dvec.create()
  local cont_out_buf = dvec.create()
  local function encode(bit_off, bit_nbr, continuous, n)
    local cat_dv = csr_m.to_dvec(bit_off, bit_nbr, nil, n, n_cat)
    local cc = cat_enc:encode(cat_dv, n, cat_sims_buf, cat_out_buf)
    local co = cont_enc:encode(continuous, n, cont_sims_buf, cont_out_buf)
    return concat_codes(cc, co, n)
  end

  local val_codes = encode(validate.bit_offsets, validate.bit_neighbors, validate.continuous, validate.n)

  str.printf("[Ridge] Training\n")
  local _, ridge_obj, best_params = optimize.ridge({
    XtX = xtx, XtY = xty, col_mean = col_mean, y_mean = y_mean,
    n_samples = train.n, n_dims = emb_d, n_targets = 1,
    targets = true,
    val_codes = val_codes, val_n_samples = validate.n,
    val_targets = validate.targets,
    lambda = cfg.ridge.lambda,
    search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  xtx = nil; xty = nil; col_mean = nil; y_mean = nil
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e %s\n", best_params.lambda, sw())

  str.printf("\n[Eval] Scoring splits\n")
  local regress_buf = dvec.create()
  local train_stats = eval.regression_accuracy(ridge_obj:regress(train_codes, train.n, regress_buf), train.targets)
  local val_stats = eval.regression_accuracy(ridge_obj:regress(val_codes, validate.n, regress_buf), validate.targets)
  local test_codes = encode(test_set.bit_offsets, test_set.bit_neighbors, test_set.continuous, test_set.n)
  local test_stats = eval.regression_accuracy(ridge_obj:regress(test_codes, test_set.n, regress_buf), test_set.targets)
  str.printf("[Eval] Accuracy: train=%.1f%% val=%.1f%% test=%.1f%% %s\n",
    (1 - train_stats.nmae) * 100, (1 - val_stats.nmae) * 100, (1 - test_stats.nmae) * 100, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
