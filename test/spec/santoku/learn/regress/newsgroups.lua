local csr = require("santoku.learn.csr")
local tokenizer = require("santoku.learn.tokenizer")
local function tokenize (texts, n, nmin, nmax, tok)
  local grow = tok == nil
  if grow then tok = tokenizer.create({ ngram_min = nmin, ngram_max = nmax }) end
  local o, t, v = tok:tokenize({ texts = texts, n_samples = n, grow = grow })
  return tok, o, t, v, tok:n_tokens()
end
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
    max = nil,
    tvr = 0.1
  },
  tok = {
    ngram_min = 5,
    ngram_max = 5
  },
  emb = {
    n_landmarks = 1024 * 8,
    trace_tol = 0.01,
    kernel = { "cosine", "expcos", "geolaplace", "matern52", "rq", "arccos1", "rbf" }
  },
  ridge = {
    lambda = { def = 7.8376e-02 },
    propensity_a = { def = 7.9786 },
    propensity_b = { def = 8.1669 },
    classes = 20,
    search_trials = 0,
    k = 1
  },
}

test("newsgroups classifier", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, test_set, validate = ds.read_20newsgroups_split(
    "test/res/20news-bydate-train",
    "test/res/20news-bydate-test",
    cfg.data.max, nil, cfg.data.tvr)
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors
  str.printf("[Data] train=%d val=%d test=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_classes, sw())

  local ngram_map, offsets, tokens, values, n_tokens =
    tokenize(train.problems, train.n, cfg.tok.ngram_min, cfg.tok.ngram_max)
  local bns_scores = csr.apply_bns(
    offsets, tokens, values, nil,
    label_off, label_nbr, n_tokens, n_classes)
  csr.normalize(offsets, values)
  str.printf("[Tokenize] ngram_min=%d ngram_max=%d tokens=%d %s\n",
    cfg.tok.ngram_min, cfg.tok.ngram_max, n_tokens, sw())

  local _, val_off, val_tok, val_val =
    tokenize(validate.problems, validate.n, cfg.tok.ngram_min, cfg.tok.ngram_max, ngram_map)
  csr.apply_bns(val_off, val_tok, val_val, bns_scores)
  csr.normalize(val_off, val_val)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, val_codes, _, decider = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    val_offsets = val_off, val_tokens = val_tok, val_values = val_val,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  do
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
  validate.problems = nil
  collectgarbage("collect")
  local function encode_texts(texts, n)
    local _, off, tok, val =
      tokenize(texts, n, cfg.tok.ngram_min, cfg.tok.ngram_max, ngram_map)
    csr.apply_bns(off, tok, val, bns_scores)
    csr.normalize(off, val)
    return sp_enc:encode({
      offsets = off, tokens = tok, values = val, n_samples = n,
    })
  end

  str.printf("[Eval] Labeling splits\n")
  local val_scores = ridge_obj:regress(val_codes, validate.n)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode_texts(test_set.problems, test_set.n)
  test_set.problems = nil
  local test_scores = ridge_obj:regress(test_codes, test_set.n)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

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
