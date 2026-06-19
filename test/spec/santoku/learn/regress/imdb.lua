local csr = require("santoku.learn.csr")
local tokenizer = require("santoku.learn.tokenizer")
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
    ttr = 0.5,
    tvr = 0.1
  },
  tok = {
    ngram_min = 5,
    ngram_max = 5,
    normalize = false
  },
  emb = {
    n_landmarks = 1024 * 8,
    kernel = { "matern", "cosine", "arccos" },
    nu = { def = 3 },
    gamma = { def = 0.09435 }
  },
  ridge = {
    lambda = { def = 1.0937e-01 },
    classes = 1,
    search_trials = 0,
    k = 1
  },
}

local function tokenize (texts, n, nmin, nmax, tok)
  local grow = tok == nil
  if grow then tok = tokenizer.create({
    ngram_min = nmin, ngram_max = nmax, normalize = cfg.tok.normalize })
  end
  local o, t, v = tok:tokenize({ texts = texts, n_samples = n, grow = grow })
  return tok, o, t, v, tok:n_tokens()
end

test("imdb classifier", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)
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
  local sp_enc, ridge_obj, val_codes, _, decider, dec_metrics = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = train.n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel, nu = cfg.emb.nu, gamma = cfg.emb.gamma,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = label_off, label_neighbors = label_nbr, n_labels = n_classes,
    val_offsets = val_off, val_tokens = val_tok, val_values = val_val,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off, val_expected_neighbors = val_label_nbr,
    lambda = cfg.ridge.lambda,
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
  val_codes = nil -- luacheck: ignore
  local test_codes = encode_texts(test_set.problems, test_set.n)
  test_set.problems = nil
  local test_off, test_nbr, test_sco = ridge_obj:label(test_codes, test_set.n, 1)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  local _, test_stats = decider:score({
    offsets = test_off, neighbors = test_nbr, scores = test_sco,
    expected_offsets = test_set.sol_offsets, expected_neighbors = test_set.sol_neighbors,
    n_samples = test_set.n,
  })
  str.printf("[Decide] val %s | test %s %s\n",
    util.fmt_metrics(dec_metrics), util.fmt_metrics(test_stats), sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
