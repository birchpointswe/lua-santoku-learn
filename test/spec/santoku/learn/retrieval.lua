-- SPDX-License-Identifier: MIT
-- SPDX-FileCopyrightText: 2024 Birch Point SWE
local tokenizer = require("santoku.learn.tokenizer")
local spectral = require("santoku.learn.spectral")
local ds = require("santoku.learn.dataset")
local retrieval = require("santoku.learn.retrieval")
local csr = require("santoku.csr")
local ivec = require("santoku.ivec")
local fvec = require("santoku.fvec")
local num = require("santoku.num")
local mtx = require("santoku.mtx")
local str = require("santoku.string")
local test = require("santoku.test")
local fs = require("santoku.fs")

fs.stdout:setvbuf("line")

local function recall (Pa, Pd, nq, k)
  local oa, na = Pa:offsets(), Pa:neighbors()
  local od, nd = Pd:offsets(), Pd:neighbors()
  local tot, hit = 0, 0
  for q = 0, nq - 1 do
    local want = {}
    local lo = od:get(q)
    local hi = lo + k < od:get(q + 1) and lo + k or od:get(q + 1)
    for j = lo, hi - 1 do want[nd:get(j)] = true end
    tot = tot + (hi - lo)
    local la = oa:get(q)
    local ha = la + k < oa:get(q + 1) and la + k or oa:get(q + 1)
    for j = la, ha - 1 do
      if want[na:get(j)] then hit = hit + 1 end
    end
  end
  return tot > 0 and hit / tot or 0
end

local function bits (M)
  local _, c = M:shape()
  return M:sign(), c
end

test("retrieval: bm25 spectral codes, itq bits, exhaustive hamming", function ()

  local dataset = ds.read_imdb("test/res/imdb.50k", 1000)
  local tok = tokenizer.create({ ngram_min = 4, ngram_max = 4, normalize = true })
  local X = tok:fit({ texts = dataset.problems })
  local w, avgdl = X:bm25()
  X:normalize()
  local _, enc = spectral.encode({ x = X, n_landmarks = 256 })
  local C = enc:encode(X)
  C:normalize()
  local mu = C:center()
  C:normalize()
  local n, dim = C:shape()
  local k = 10

  local P_exact = C:topk(C, k + 1)

  local W, obj = C:itq({ iterations = 30 })
  for i = 1, obj:size() - 1 do
    assert(obj:get(i) <= obj:get(i - 1) + 1e-6 * obj:get(i - 1))
  end
  local B_itq, nb_itq = bits(C:multiply(W))
  local B_sign, nb_sign = bits(C)
  local r_itq = recall(B_itq:bits_topk(B_itq, nb_itq, k + 1), P_exact, n, k + 1)
  local r_sign = recall(B_sign:bits_topk(B_sign, nb_sign, k + 1), P_exact, n, k + 1)

  local W64, _, _, kept64 = C:itq({ bits = 64, iterations = 30 })
  local B64, nb64 = bits(C:multiply(W64))
  local r_64 = recall(B64:bits_topk(B64, nb64, k + 1), P_exact, n, k + 1)

  str.printf("[Retrieval] docs=%d dim=%d recall@%d itq=%.4f sign=%.4f itq64=%.4f kept64=%.4f\n",
    n, dim, k + 1, r_itq, r_sign, r_64, kept64)

  assert(r_itq > r_sign + 0.1)
  assert(r_itq > 0.35)
  assert(r_64 > 0.2 and r_64 < r_itq)
  assert(kept64 > 0 and kept64 < 1)

  local Y = tok:tokenize({ texts = { dataset.problems[1], dataset.problems[2] } })
  Y:bm25(w, avgdl)
  Y:normalize()
  local Q = enc:encode(Y)
  Q:normalize()
  Q:center(mu)
  Q:normalize()
  local Pq = B_itq:bits_topk((bits(Q:multiply(W))), nb_itq, 1)
  assert(Pq:neighbors():get(0) == 0 and Pq:neighbors():get(1) == 1)

  local tmp = fs.tmpname() .. ".mtx"
  W:persist(tmp)
  local W2 = mtx.load(tmp)
  fs.rm(tmp, true)
  local B2 = bits(C:multiply(W2))
  local P1, P2 = B_itq:bits_topk(B_itq, nb_itq, k), B2:bits_topk(B2, nb_itq, k)
  assert(recall(P1, P2, n, k) == 1)

end)

test("retrieval: rerank without embeddings and with missing embedding rows", function ()

  local R = csr.create({
    offsets = ivec.create({ 0, 3 }),
    neighbors = ivec.create({ 0, 1, 2 }),
    values = fvec.create({ 1, 3, 2 }),
    n_cols = 3,
  })
  assert(retrieval.rerank({ candidates = R, alpha = 0.96 }) == R, "no embeddings must return the bm25 candidates")

  local Qc = mtx.create({ data = fvec.create({ 1, 0 }), n_rows = 1, n_cols = 2 })
  local D = mtx.create({ data = fvec.create({ 1, 0, 0, 1 }), n_rows = 2, n_cols = 2 })
  local F = retrieval.rerank({ candidates = R, query_codes = Qc, doc_codes = D, alpha = 0.96 })
  local nb, vs = F:neighbors(), F:values()
  assert(nb:size() == 3, "every candidate must survive rerank")
  local got = {}
  for j = 0, nb:size() - 1 do got[nb:get(j)] = vs:get(j) end
  assert(num.abs(got[0] - (0.04 / 3 + 0.96)) < 1e-6)
  assert(num.abs(got[1] - 0.04) < 1e-6)
  assert(num.abs(got[2] - 0.04 * 2 / 3) < 1e-6, "a candidate with no embedding row keeps its bm25 term")

end)
