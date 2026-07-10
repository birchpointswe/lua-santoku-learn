local ds = require("santoku.learn.dataset")
local optimize = require("santoku.learn.optimize")
local csr = require("santoku.csr")
local spans = require("santoku.spans")
local aho = require("santoku.learn.aho")
local ivec = require("santoku.ivec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local cfg = {
  verbose = false,
  search_landmarks = 1024 * 2,
  landmark_rounds = 32,
  search_landmark_rounds = 1,
  data = { dir = "test/res/conll2003", max = nil },
  blocks = {
    { ngram_min = 1, ngram_max = 5, normalize = false },
    { ngram_min = 1, ngram_max = 3, words = true, word_characters = util.WORD_CHARACTERS, normalize = false },
  },
  emb = { n_landmarks = 1024 * 8 },
  head = {
    kernel = { "matern" },
    nu = { def = 3 },
    gamma = { def = 0.908231 },
    lambda = { def = 1.20122e-07 },
    relevance = { "bns", "bns" },
    scales = { def = { 0.186851, 5.35186 } },
    exponent = { def = { 1.74648, 7.99336 } },
    decode_offset = { def = -0.575133 },
    search_trials = 0,
    folds = 5,
  },
}

local N_TYPES = 4
local word_characters = util.WORD_CHARACTERS

local function merge_splits (a, b)
  local m = { n = a.n + b.n, texts = {}, sent_tokens = {}, sent_ents = {},
    n_pos = a.n_pos, pos_names = a.pos_names, n_types = a.n_types, type_names = a.type_names }
  for i = 1, a.n do m.texts[i] = a.texts[i]; m.sent_tokens[i] = a.sent_tokens[i]; m.sent_ents[i] = a.sent_ents[i] end
  for i = 1, b.n do local j = a.n + i
    m.texts[j] = b.texts[i]; m.sent_tokens[j] = b.sent_tokens[i]; m.sent_ents[j] = b.sent_ents[i] end
  return m
end

local function build_gaz (splits)
  local counts = {}
  for _, split in ipairs(splits) do
    for d = 1, split.n do
      local text = split.texts[d]
      for _, e in ipairs(split.sent_ents[d]) do
        local surf = text:sub(e.s + 1, e.e)
        local c = counts[surf]
        if not c then c = {}; for ty = 0, N_TYPES - 1 do c[ty] = 0 end; counts[surf] = c end
        c[e.t] = c[e.t] + 1
      end
    end
  end
  local patterns, pat_type = {}, ivec.create()
  for surf, c in pairs(counts) do
    patterns[#patterns + 1] = surf
    local bt, bc = 0, -1
    for ty = 0, N_TYPES - 1 do if c[ty] > bc then bc, bt = c[ty], ty end end
    pat_type:push(bt)
  end
  return aho.create({ patterns = patterns, normalize = false }), pat_type
end

local function gold_spans (split)
  local eoff, es, ee, ety = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  eoff:push(0)
  for d = 1, split.n do
    for _, e in ipairs(split.sent_ents[d]) do es:push(e.s); ee:push(e.e); ety:push(e.t) end
    eoff:push(es:size())
  end
  return spans.create({ offsets = eoff, s = es, e = ee, ty = ety })
end

local function candidates (ac, pat_type, split)
  local S = ac:predict({ texts = split.texts, longest = true, word_characters = word_characters })
  local id = S:col("id")
  local ty = ivec.create(id:size()):copy(pat_type, id)
  return spans.create({ offsets = S:offsets(), s = S:col("s"), e = S:col("e"), ty = ty })
end

local function cand_labels (Scand, Sgold)
  return csr.from_mask(Scand:match_labels(Sgold))
end

test("conll-gaz CV", function ()
  local stopwatch = utc.stopwatch()
  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  local ac, pat_type = build_gaz({ train, dev, test_set })
  local pool = merge_splits(train, dev)
  local Gtr, Gte = gold_spans(pool), gold_spans(test_set)
  local Ctr, Cte = candidates(ac, pat_type, pool), candidates(ac, pat_type, test_set)
  local n_pool, n_test = Ctr:col("s"):size(), Cte:col("s"):size()
  str.printf("[Cands] pool=%d test=%d | test coverage=%.4f folds=%d trials=%d\n",
    n_pool, n_test, Cte:coverage(Gte), cfg.head.folds, cfg.head.search_trials)

  local toks, Xtr = util.tokenize_focus_blocks(cfg.blocks, pool.texts, Ctr)
  local _, Xte = util.tokenize_focus_blocks(cfg.blocks, test_set.texts, Cte, toks)
  local Ytr = cand_labels(Ctr, Gtr)

  local _, rg, deploy, best, decider = optimize.krr({
    pool_blocks = Xtr,
    pool_labels = Ytr,
    pool_n = n_pool,
    n_labels = 1,
    reject = N_TYPES,
    folds = cfg.head.folds,
    cand = Ctr,
    gold = Gtr,
    relevance = cfg.head.relevance,
    scales = cfg.head.scales,
    exponent = cfg.head.exponent,
    kernel = cfg.head.kernel,
    nu = cfg.head.nu,
    gamma = cfg.head.gamma,
    lambda = cfg.head.lambda,
    n_landmarks = cfg.emb.n_landmarks,
    search_landmarks = cfg.search_landmarks,
    landmark_rounds = cfg.landmark_rounds,
    search_landmark_rounds = cfg.search_landmark_rounds,
    k = 1,
    decode_offset = cfg.head.decode_offset,
    search_trials = cfg.head.search_trials,
    verbose = cfg.verbose,
    each = util.make_ridge_log(stopwatch),
  })

  local _, test_scores = util.predict_tiled({ deploy = deploy, ridge = rg,
    blocks = Xte, n = n_test, scores = true, n_labels = 1 })
  local _, m = decider:score({ scores = test_scores,
    n_samples = test_set.n, cand = Cte, gold = Gte })
  local _, total = stopwatch()
  str.printf("[Span] lambda=%.4g offset=%.6g | test %s\nTotal: %.1fs\n",
    best.lambda or 0, decider:offset(), util.fmt_metrics(m), total)
end)
