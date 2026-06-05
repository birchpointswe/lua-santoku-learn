local csr = require("santoku.learn.csr")
local csr_m = require("santoku.csr")
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

-- Reported metrics (search_trials=100; splits train=16512 val=2064 test=2064):
--   n_landmarks=8192:  acc val=83.0% test=82.8%  (best: angular, lambda=2.31e-02)
--   n_landmarks=16384: acc val=83.1% test=82.9%  (best: geolaplace, lambda=1.28e-01; pre-kernel-expansion)

local cfg = {
  data = { ttr = 0.8, tvr = 0.1, max = nil },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01, kernel = { "angular", "cosine", "expcos", "geolaplace", "matern32", "matern52", "rq", "arccos1" } },
  ridge = { lambda = { def = 2.3081e-02 }, search_trials = 0 },
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
  -- Per-block normalization: scale each modality to unit mean per-row squared norm,
  -- MEASURED on the standardized train matrix. This counts each block's actual
  -- contribution (correct for sparse one-hot, where 1/sqrt(n_cols) is wrong since only
  -- the active bits contribute). block_sumsq groups by token-block entirely in C.
  local ss = csr.block_sumsq(tokens, values, { 0, n_cat, n_tokens })
  local ss_cat, ss_cont = ss:get(0), ss:get(1)
  local block = fvec.create(n_tokens)
  block:fill(ss_cat > 0 and math.sqrt(train.n / ss_cat) or 0.0, 0, n_cat)
  block:fill(ss_cont > 0 and math.sqrt(train.n / ss_cont) or 0.0, n_cat, n_tokens)
  csr.standardize(offsets, tokens, values, block)
  std_scores:scalev(block)
  csr.normalize(offsets, values)

  local val_off, val_tok, val_val = merge_features(
    validate.bit_offsets, validate.bit_neighbors, validate.continuous, validate.n)
  csr.standardize(val_off, val_tok, val_val, std_scores)
  csr.normalize(val_off, val_val)

  str.printf("[KRR] Encoding n_landmarks=%d n_tokens=%d\n",
    cfg.emb.n_landmarks, n_tokens)
  local sp_enc, ridge_obj, val_codes = optimize.krr({
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
  do  -- persist/load parity: round-tripped encoder must produce identical codes
    local p = os.tmpname()
    sp_enc:persist(p)
    local enc2 = spectral.load(p)
    os.remove(p)
    local vc2 = enc2:encode({ offsets = val_off, tokens = val_tok, values = val_val, n_samples = validate.n })
    local nchk = val_codes:size()
    if nchk > 100000 then nchk = 100000 end
    for i = 0, nchk - 1 do
      assert(val_codes:get(i) == vc2:get(i), "persist/load parity mismatch at " .. i)
    end
    str.printf("[Persist] load parity OK (%d codes)\n", nchk)
  end
  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  collectgarbage("collect")
  local function encode(bit_off, bit_nbr, continuous, n)
    local off, tok, val = merge_features(bit_off, bit_nbr, continuous, n)
    csr.standardize(off, tok, val, std_scores)
    csr.normalize(off, val)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  str.printf("\n[Eval] Scoring splits\n")
  local regress_buf = fvec.create()
  local val_stats = eval.regress_accuracy(ridge_obj:regress(val_codes, validate.n, regress_buf), validate.targets)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode(test_set.bit_offsets, test_set.bit_neighbors, test_set.continuous, test_set.n)
  local test_stats = eval.regress_accuracy(ridge_obj:regress(test_codes, test_set.n, regress_buf), test_set.targets)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Accuracy: val=%.1f%% test=%.1f%% %s\n",
    (1 - val_stats.nmae) * 100, (1 - test_stats.nmae) * 100, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
