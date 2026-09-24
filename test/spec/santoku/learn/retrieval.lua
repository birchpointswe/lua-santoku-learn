local tokenizer = require("santoku.learn.tokenizer")
local spectral = require("santoku.learn.spectral")
local ds = require("santoku.learn.dataset")
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
  local r, c = M:shape()
  return mtx.create({ data = M:sign(), n_rows = r, n_cols = c, bits = true })
end

test("retrieval: bm25 spectral codes, itq bits, exhaustive hamming", function ()

  local dataset = ds.read_imdb("test/res/imdb.50k", 1000)
  local tok = tokenizer.create({ ngram_min = 4, ngram_max = 4, normalize = true })
  local X = tok:fit({ texts = dataset.problems })
  local w, avgdl = X:bm25()
  X:normalize()
  local _, enc = spectral.encode({ x = X, n_landmarks = 256, kernel = "cosine" })
  local C = enc:encode(X)
  C:normalize("row")
  local mu = C:center()
  C:normalize("row")
  local n, dim = C:shape()
  local k = 10

  local P_exact = C:topk(C, k + 1)

  local W, obj = C:itq({ iterations = 30 })
  for i = 1, obj:size() - 1 do
    assert(obj:get(i) <= obj:get(i - 1) + 1e-6 * obj:get(i - 1))
  end
  local B_itq = bits(C:multiply(W))
  local B_sign = bits(C)
  local r_itq = recall(B_itq:topk(B_itq, k + 1), P_exact, n, k + 1)
  local r_sign = recall(B_sign:topk(B_sign, k + 1), P_exact, n, k + 1)

  local W64, _, _, kept64 = C:itq({ bits = 64, iterations = 30 })
  local B64 = bits(C:multiply(W64))
  local r_64 = recall(B64:topk(B64, k + 1), P_exact, n, k + 1)

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
  Q:normalize("row")
  Q:center(mu)
  Q:normalize("row")
  local Pq = B_itq:topk(bits(Q:multiply(W)), 1)
  assert(Pq:neighbors():get(0) == 0 and Pq:neighbors():get(1) == 1)

  local tmp = fs.tmpname() .. ".mtx"
  W:persist(tmp)
  local W2 = mtx.load(tmp)
  fs.rm(tmp, true)
  local B2 = bits(C:multiply(W2))
  local P1, P2 = B_itq:topk(B_itq, k), B2:topk(B2, k)
  assert(recall(P1, P2, n, k) == 1)

end)
