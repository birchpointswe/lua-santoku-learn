local csr = require("santoku.learn.csr")
local csr_m = require("santoku.csr")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local fvec = require("santoku.fvec")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  data = { ttr = 0.8, tvr = 0.1, max = nil },
  emb = { n_landmarks = 1024*32, trace_tol = 0.01, kernel = { "ntk", "cosine", "nngp", "expcos", "geolaplace" } },
  ridge = { lambda = { def = 1.33e-01 }, search_trials = 800 },
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
  local cont_mean = train.continuous:mtx_center(n_cont)
  validate.continuous:mtx_center(n_cont, cont_mean)
  test_set.continuous:mtx_center(n_cont, cont_mean)
  str.printf("[Data] train=%d val=%d test=%d cat=%d cont=%d target=%.0f-%.0f %s\n",
    train.n, validate.n, test_set.n, n_cat, n_cont,
    train.targets:min(), train.targets:max(), sw())

  local function merge_features(bit_off, bit_nbr, continuous, n)
    local cont_off, cont_nbr, cont_val = csr_m.from_dvec(continuous, n, n_cont)
    return csr.merge(bit_off, bit_nbr, nil, cont_off, cont_nbr, cont_val, n_cat)
  end

  local n_tokens = n_cat + n_cont
  local offsets, tokens, values = merge_features(
    train.bit_offsets, train.bit_neighbors, train.continuous, train.n)
  local std_scores = csr.standardize(offsets, tokens, values, nil, n_tokens)

  local val_off, val_tok, val_val = merge_features(
    validate.bit_offsets, validate.bit_neighbors, validate.continuous, validate.n)
  csr.standardize(val_off, val_tok, val_val, std_scores)

  str.printf("[KRR] Encoding n_landmarks=%d n_tokens=%d\n",
    cfg.emb.n_landmarks, n_tokens)
  local sp_enc, ridge_obj, val_codes, best_params = optimize.krr({
    offsets = offsets, tokens = tokens, values = values, n_tokens = n_tokens,
    n_samples = train.n,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    kernel = cfg.emb.kernel,
    targets = train.targets, n_targets = 1,
    val_offsets = val_off, val_tokens = val_tok, val_values = val_val,
    val_n_samples = validate.n,
    val_targets = validate.targets,
    lambda = cfg.ridge.lambda,
    search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  collectgarbage("collect")
  local emb_d = sp_enc:dims()
  str.printf("[KRR] emb_d=%d kernel=%s lambda=%.4e %s\n",
    emb_d, best_params.kernel, best_params.lambda, sw())

  local function encode(bit_off, bit_nbr, continuous, n)
    local off, tok, val = merge_features(bit_off, bit_nbr, continuous, n)
    csr.standardize(off, tok, val, std_scores)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  str.printf("\n[Eval] Scoring splits\n")
  local regress_buf = fvec.create()
  local val_stats = eval.regression_accuracy(ridge_obj:regress(val_codes, validate.n, regress_buf), validate.targets)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode(test_set.bit_offsets, test_set.bit_neighbors, test_set.continuous, test_set.n)
  local test_stats = eval.regression_accuracy(ridge_obj:regress(test_codes, test_set.n, regress_buf), test_set.targets)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Accuracy: val=%.1f%% test=%.1f%% %s\n",
    (1 - val_stats.nmae) * 100, (1 - test_stats.nmae) * 100, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
