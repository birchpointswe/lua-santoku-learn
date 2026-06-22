local tokenizer = require("santoku.learn.tokenizer")
local boundary = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
local function tokenize (texts, nmin, nmax, bundle)
  local grow = bundle == nil
  if grow then bundle = {
    byte = tokenizer.create({ ngram_min = nmin, ngram_max = nmax, boundary = boundary }),
    word = tokenizer.create({ ngram_min = 1, ngram_max = 3, boundary = boundary, words = true }),
  } end
  local Xb = grow and bundle.byte:fit({ texts = texts }) or bundle.byte:tokenize({ texts = texts })
  local Xw = grow and bundle.word:fit({ texts = texts }) or bundle.word:tokenize({ texts = texts })
  return bundle, Xb:hcat(Xw)
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
    kernel = { "cosine", "matern" }
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
  local sp_enc, ridge_obj, val_codes, _, decider = optimize.krr({
    x = X, y = train.labels, val_x = Xv, val_y = validate.labels,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    lambda = cfg.ridge.lambda, propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
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
  local val_scores = ridge_obj:regress(val_codes)
  val_codes = nil -- luacheck: ignore
  local test_codes = encode_texts(test_set.problems)
  test_set.problems = nil
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
