-- SPDX-License-Identifier: MIT
-- SPDX-FileCopyrightText: 2024 Birch Point SWE
local env = require("santoku.env")
if env.var("TK_LEARN_REGRESS", nil) ~= "1" then
  print("TK_LEARN_REGRESS not set. Skipping.")
  return
end

local ds = require("santoku.learn.dataset")
local retrieval = require("santoku.learn.retrieval")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local num = require("santoku.num")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")
local fs = require("santoku.fs")

fs.stdout:setvbuf("line")

local cfg = {
  depth = 100,
  bm25_ndcg = 0.6663,
  n_landmarks = 256,
}

local function codes (enc, X, mu)
  local C = enc:encode(X)
  C:normalize()
  if mu then C:center(mu) else mu = C:center() end
  C:normalize()
  return C, mu
end

test("scifact retrieval", function ()
  local stopwatch = utc.stopwatch()
  local d = ds.read_beir("test/res/scifact", "test")
  str.printf("[Data] corpus=%d queries=%d\n", d.n_corpus, d.n_queries)
  assert(d.n_corpus == 5183 and d.n_queries == 300)

  local X, Q = retrieval.lexical({ corpus_texts = d.corpus_texts, query_texts = d.query_texts })
  local rank = retrieval.bm25_ranker(X, Q)
  local R = rank(1.2, 0.75, cfg.depth)
  local nd, m = R:ndcg(d.qrels, 10)
  local _, rc = R:recall(d.qrels, cfg.depth)
  local _, mr = R:mrr(d.qrels, 10)
  str.printf("[Lexical] ndcg@10=%.4f recall@%d=%.4f mrr@10=%.4f\n", m, cfg.depth, rc, mr)
  assert(num.abs(m - cfg.bm25_ndcg) < 1e-4, "bm25 ndcg drifted from the FTS5 pin")
  local _, m2 = rank(1.2, 0.75, cfg.depth):ndcg(d.qrels, 10)
  assert(m2 == m, "bm25_ranker is not repeatable")

  local lex = optimize.lexical({
    datasets = { { name = "scifact", corpus_texts = d.corpus_texts, query_texts = d.query_texts, qrels = d.qrels } },
  })
  str.printf("[Lexical] defaults k1=%.4g b=%.4g ngram=%d\n", lex.k1, lex.b, lex.ngram)
  assert(lex.k1 == 1.2 and lex.b == 0.75 and lex.ngram == 1)

  local f = retrieval.featurizer({ ngram_min = 4, ngram_max = 4, texts = d.corpus_texts })
  local _, enc = spectral.encode({ x = f.X, n_landmarks = cfg.n_landmarks })
  local D, mu = codes(enc, f.X)
  local Qc = codes(enc, f.transform(d.query_texts), mu)

  local R0 = retrieval.rerank({ candidates = R, query_codes = Qc, doc_codes = D, alpha = 0 })
  local n0, m0 = R0:ndcg(d.qrels, 10)
  local delta, p = nd:paired_test(n0, 1000, 1)
  str.printf("[Rerank] alpha=0 ndcg@10=%.4f delta=%+.4f p=%.4f\n", m0, delta, p)
  assert(m0 == m and delta == 0, "alpha 0 must keep the lexical order")

  local set = { name = "scifact", candidates = R, qrels = d.qrels, query_codes = Qc, doc_codes = D }
  local best = optimize.retrieval({ datasets = { set } })
  assert(best.alpha == 0.96, "published alpha default changed")
  local R1 = retrieval.rerank({ candidates = R, query_codes = Qc, doc_codes = D, alpha = best.alpha })
  local o = R:overlap(R1, cfg.depth)
  assert(o:sum() == o:size(), "rerank must keep the candidate set")
  local _, m1 = R1:ndcg(d.qrels, 10)
  local _, total = stopwatch()
  str.printf("[Rerank] alpha=%.2f ndcg@10=%.4f (spectral codes)\nTotal: %.1fs\n", best.alpha, m1, total)
end)
