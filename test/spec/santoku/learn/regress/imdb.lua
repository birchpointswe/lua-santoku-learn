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
    kernel = { "matern", "cosine" },
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

local function tokenize (texts, nmin, nmax, tok)
  local grow = tok == nil
  if grow then tok = tokenizer.create({
    ngram_min = nmin, ngram_max = nmax, normalize = cfg.tok.normalize })
  end
  local X = grow and tok:fit({ texts = texts }) or tok:tokenize({ texts = texts })
  return tok, X
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
  str.printf("[Data] train=%d val=%d test=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_classes, sw())

  local ngram_map, X = tokenize(train.problems, cfg.tok.ngram_min, cfg.tok.ngram_max)
  local _, n_tokens = X:shape()
  local bns_scores = X:bns(train.labels)
  X:normalize()
  str.printf("[Tokenize] ngram_min=%d ngram_max=%d tokens=%d %s\n",
    cfg.tok.ngram_min, cfg.tok.ngram_max, n_tokens, sw())

  local _, Xv = tokenize(validate.problems, cfg.tok.ngram_min, cfg.tok.ngram_max, ngram_map)
  Xv:bns(bns_scores)
  Xv:normalize()

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, _, _, decider, dec_metrics = optimize.krr({
    x = X, y = train.labels, val_x = Xv, val_y = validate.labels,
    kernel = cfg.emb.kernel, nu = cfg.emb.nu, gamma = cfg.emb.gamma,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    lambda = cfg.ridge.lambda,
    k = cfg.ridge.k, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  do
    local p = os.tmpname()
    sp_enc:persist(p)
    sp_enc = spectral.load(p)   -- continue (and re-encode test) with the reloaded encoder
    os.remove(p)
  end
  X = nil; Xv = nil -- luacheck: ignore
  validate.problems = nil
  collectgarbage("collect")
  local function encode_texts(texts)
    local _, Xt = tokenize(texts, cfg.tok.ngram_min, cfg.tok.ngram_max, ngram_map)
    Xt:bns(bns_scores)
    Xt:normalize()
    return sp_enc:encode(Xt)
  end

  str.printf("[Eval] Labeling splits\n")
  local test_codes = encode_texts(test_set.problems)
  test_set.problems = nil
  local P = ridge_obj:label(test_codes, 1)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  local _, test_stats = decider:score({ pred = P, expected = test_set.labels, n_samples = test_set.n })
  str.printf("[Decide] val %s | test %s %s\n",
    util.fmt_metrics(dec_metrics), util.fmt_metrics(test_stats), sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
