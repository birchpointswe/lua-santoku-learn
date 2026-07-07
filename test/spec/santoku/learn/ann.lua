local tokenizer = require("santoku.learn.tokenizer")
local spectral = require("santoku.learn.spectral")
local ann = require("santoku.learn.ann")
local ds = require("santoku.learn.dataset")
local str = require("santoku.string")
local test = require("santoku.test")

io.stdout:setvbuf("line")


local function recall (Pa, Pd, nq)
  local oa, na = Pa:offsets(), Pa:neighbors()
  local od, nd = Pd:offsets(), Pd:neighbors()
  local tot, hit = 0, 0
  for q = 0, nq - 1 do
    local want = {}
    for j = od:get(q), od:get(q + 1) - 1 do want[nd:get(j)] = true end
    tot = tot + (od:get(q + 1) - od:get(q))
    for j = oa:get(q), oa:get(q + 1) - 1 do
      if want[na:get(j)] then hit = hit + 1 end
    end
  end
  return tot > 0 and hit / tot or 0
end

test("ann spectral idf retrieval", function ()


  local dataset = ds.read_imdb("test/res/imdb.50k", 500)
  local texts = dataset.problems
  local tok = tokenizer.create({ ngram_min = 4, ngram_max = 4, normalize = true })
  local X = tok:fit({ texts = texts })
  X:idf()
  X:normalize()
  local _, enc = spectral.encode({ x = X, n_landmarks = 256, kernel = "cosine" })
  local C = enc:encode(X)
  C:normalize("row")
  local nq, dim = C:shape()
  str.printf("[ANN] docs=%d dim=%d\n", nq, dim)

  local k = 10


  local radius = 6


  local P_exact = C:topk(C, k)


  local idx = ann.create({ codes = C })
  local P_rr = idx:neighborhoods_by_vecs(C, k, radius)
  local r_rr = recall(P_rr, P_exact, nq)


  local idx_bin = ann.create({ codes = C, rerank = false })
  local P_bin = idx_bin:neighborhoods_by_vecs(C, k, radius)
  local r_bin = recall(P_bin, P_exact, nq)


  local P_self = idx:neighborhoods(k)

  local P_self_rr = idx:neighborhoods(k, true)
  local P_self_bin = idx:neighborhoods(k, false)

  str.printf("[ANN] recall@%d  rerank=%.4f  hamming=%.4f\n", k, r_rr, r_bin)


  assert(P_rr:offsets():size() == nq + 1)
  assert(P_bin:offsets():size() == nq + 1)
  assert(P_self:offsets():size() == nq + 1)
  assert(P_self_rr:offsets():size() == nq + 1)
  assert(P_self_bin:offsets():size() == nq + 1)
  assert(recall(P_self_rr, P_self, nq) == 1 and recall(P_self, P_self_rr, nq) == 1)

  assert(r_rr >= r_bin)

  assert(r_rr > 0.3)

end)
