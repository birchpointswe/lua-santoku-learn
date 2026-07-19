local tokenizer = require("santoku.learn.tokenizer")
local spectral = require("santoku.learn.spectral")
local ann = require("santoku.learn.ann")
local ds = require("santoku.learn.dataset")
local str = require("santoku.string")
local test = require("santoku.test")

io.stdout:setvbuf("line")

-- mean recall@k of an approximate P csr against the exact (mtx:topk) P csr, per query row.
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

  -- ~1000 docs (500 pos + 500 neg) -> idf-weighted char-ngram csr -> spectral codes (mtx)
  local dataset = ds.read_imdb("test/res/imdb.50k", 500)
  local texts = dataset.problems
  local tok = tokenizer.create({ ngram_min = 4, ngram_max = 4, normalize = true })
  local X = tok:fit({ texts = texts })
  X:idf()
  X:normalize()
  local _, enc = spectral.encode({ x = X, n_landmarks = 256, kernel = "cosine" })
  local C = enc:encode(X)
  C:normalize("row")   -- row L2: dot == cosine, matching sign-LSH's angular nature
  local nq, dim = C:shape()
  str.printf("[ANN] docs=%d dim=%d\n", nq, dim)

  local k = 10
  -- radius governs the candidate pool; rerank needs a pool > k to beat Hamming-only, since the
  -- Hamming path early-terminates at k while the rerank path keeps probing to the radius cap.
  local radius = 6

  -- exact brute-force top-k (ground truth)
  local P_exact = C:topk(C, k)

  -- mode 1: sign-LSH index + exact-dot rerank (codes retained)
  local idx = ann.create({ codes = C })
  local P_rr = idx:neighborhoods_by_vecs(C, k, radius)
  local r_rr = recall(P_rr, P_exact, nq)

  -- mode 2: binarize-only index (Hamming ranking, no rerank)
  local idx_bin = ann.create({ codes = C, rerank = false })
  local P_bin = idx_bin:neighborhoods_by_vecs(C, k, radius)
  local r_bin = recall(P_bin, P_exact, nq)

  -- mode 3: self-neighborhoods (corpus vs itself, self excluded) -> csr smoke check
  local P_self = idx:neighborhoods(k)
  -- explicit boolean rerank arg: true matches the default (codes retained), false is Hamming-only
  local P_self_rr = idx:neighborhoods(k, true)
  local P_self_bin = idx:neighborhoods(k, false)

  str.printf("[ANN] recall@%d  rerank=%.4f  hamming=%.4f\n", k, r_rr, r_bin)

  -- shapes: each search returns a P csr with one row per query
  assert(P_rr:offsets():size() == nq + 1)
  assert(P_bin:offsets():size() == nq + 1)
  assert(P_self:offsets():size() == nq + 1)
  assert(P_self_rr:offsets():size() == nq + 1)
  assert(P_self_bin:offsets():size() == nq + 1)
  assert(recall(P_self_rr, P_self, nq) == 1 and recall(P_self, P_self_rr, nq) == 1)
  -- rerank pulls a wider Hamming pool then keeps the exact-dot top-k, so it dominates Hamming-only
  assert(r_rr >= r_bin)
  -- sanity floor: cosine-aligned codes recover well above random (random recall@10 ~ 0.01)
  assert(r_rr > 0.3)

  -- persist -> reload reproduces the index exactly (raw mmap-compatible sidecars)
  local tmp = os.tmpname() .. ".ann"
  idx:persist(tmp)
  -- mmap sidecars + external rerank codes: identical to the in-memory rerank index
  local idx_m = ann.load(tmp, C)
  local P_m = idx_m:neighborhoods_by_vecs(C, k, radius)
  assert(recall(P_m, P_rr, nq) == 1 and recall(P_rr, P_m, nq) == 1)
  -- RAM sidecars (mmap = false): same result, internally-managed buffers
  local idx_r = ann.load(tmp, C, false)
  local P_r = idx_r:neighborhoods_by_vecs(C, k, radius)
  assert(recall(P_r, P_rr, nq) == 1 and recall(P_rr, P_r, nq) == 1)
  -- no codes passed -> Hamming-only, matches the create-time binarize path
  local idx_h = ann.load(tmp)
  local P_h = idx_h:neighborhoods_by_vecs(C, k, radius)
  assert(recall(P_h, P_bin, nq) == 1 and recall(P_bin, P_h, nq) == 1)
  for _, sfx in ipairs({ "", ".sids", ".buckets", ".bits" }) do os.remove(tmp .. sfx) end

end)
