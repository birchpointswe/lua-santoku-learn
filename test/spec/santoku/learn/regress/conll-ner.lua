local ds = require("santoku.learn.dataset")
local csr = require("santoku.learn.csr")
local optimize = require("santoku.learn.optimize")
local spectral = require("santoku.learn.spectral")
local eval = require("santoku.learn.evaluator")
local ivec = require("santoku.ivec")
local util = require("santoku.learn.util")
local str = require("santoku.string")
local test = require("santoku.test")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

-- End-to-end NER as per-token BIO x type tagging + deterministic run-collapse (no gazetteer, so
-- recall is not coverage-capped). Classes: O=0, B-<t>=1+t, I-<t>=1+n_types+t over the 4 conll
-- types (PER/ORG/LOC/MISC) => 2*n_types+1 classes. csr.bio_encode/bio_decode do the span<->token
-- conversion; eval.span_f1 scores. Each token is rendered with full-sentence context
-- (collapse="none" marks the focus token inside the whole sentence).

local cfg = {
  data = { dir = "test/res/conll2003", max = nil },
  tok = { ngram_min = 3, ngram_max = 5, collapse = "none", normalize = true },
  emb = { n_landmarks = 1024 * 8, trace_tol = 0.01,
    kernel = { "cosine", "expcos", "geolaplace", "angular", "matern32", "matern52", "rq", "arccos1" } },
  ridge = { lambda = { min = 1e-4, max = 1e1, log = true, def = 1e-1 }, search_trials = 100 },
}

local N_TYPES = 4                   -- PER/ORG/LOC/MISC
local N_CLASSES = 2 * N_TYPES + 1   -- O + B/I x types

-- marshal per-token spans + gold entity spans into ivecs; BIO x type labels via csr.bio_encode
local function build_tokens (split)
  local off, starts, ends, types = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  local eoff, es, ee, ety = ivec.create(), ivec.create(), ivec.create(), ivec.create()
  off:push(0); eoff:push(0)
  for d = 1, split.n do
    for _, t in ipairs(split.sent_tokens[d]) do
      starts:push(t.s); ends:push(t.e); types:push(0)
    end
    off:push(starts:size())
    for _, e in ipairs(split.sent_ents[d]) do
      es:push(e.s); ee:push(e.e); ety:push(e.t)
    end
    eoff:push(es:size())
  end
  local lab_off, lab_nbr = csr.bio_encode(off, starts, ends, eoff, es, ee, ety, N_TYPES)
  return off, starts, ends, types, lab_off, lab_nbr, starts:size(), eoff, es, ee, ety
end

local function tokenize (split, off, starts, ends, types, ngram_map)
  return csr.tokenize_annotated({
    texts = split.texts, doc_span_offsets = off,
    span_starts = starts, span_ends = ends, span_types = types,
    collapse = cfg.tok.collapse,
    ngram_min = cfg.tok.ngram_min, ngram_max = cfg.tok.ngram_max,
    normalize = cfg.tok.normalize, terminals = true, ngram_map = ngram_map,
  })
end

test("conll ner", function ()

  local stopwatch = utc.stopwatch()
  local function sw ()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_conll2003(cfg.data.dir, cfg.data.max)
  str.printf("[Data] train=%d dev=%d test=%d %s\n", train.n, dev.n, test_set.n, sw())

  local tr_off, tr_s, tr_e, tr_ty, tr_loff, tr_lnbr, tr_n = build_tokens(train)
  local dv_off, dv_s, dv_e, dv_ty, dv_loff, dv_lnbr, dv_n = build_tokens(dev)
  local te_off, te_s, te_e, te_ty, te_loff, te_lnbr, te_n, te_eoff, te_es, te_ee, te_ety =
    build_tokens(test_set)
  str.printf("[Tokens] train=%d dev=%d test=%d %s\n", tr_n, dv_n, te_n, sw())

  local ngram_map, offsets, tokens, values, n_tokens =
    tokenize(train, tr_off, tr_s, tr_e, tr_ty, nil)
  local bns = csr.apply_bns(offsets, tokens, values, nil, tr_loff, tr_lnbr, n_tokens, N_CLASSES)
  csr.normalize(offsets, values)
  str.printf("[Tokenize] tokens=%d %s\n", n_tokens, sw())

  local _, dv_o, dv_t, dv_v = tokenize(dev, dv_off, dv_s, dv_e, dv_ty, ngram_map)
  csr.apply_bns(dv_o, dv_t, dv_v, bns)
  csr.normalize(dv_o, dv_v)

  str.printf("[KRR] Encoding n_landmarks=%d\n", cfg.emb.n_landmarks)
  local sp_enc, ridge_obj, dv_codes, best_params = optimize.krr({
    offsets = offsets, tokens = tokens, values = values,
    n_samples = tr_n, n_tokens = n_tokens,
    kernel = cfg.emb.kernel,
    n_landmarks = cfg.emb.n_landmarks, trace_tol = cfg.emb.trace_tol,
    label_offsets = tr_loff, label_neighbors = tr_lnbr, n_labels = N_CLASSES,
    val_offsets = dv_o, val_tokens = dv_t, val_values = dv_v, val_n_samples = dv_n,
    val_expected_offsets = dv_loff, val_expected_neighbors = dv_lnbr,
    lambda = cfg.ridge.lambda, k = 1, search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })

  do  -- persist/load parity: round-tripped encoder must produce identical codes
    local pth = os.tmpname()
    sp_enc:persist(pth)
    local enc2 = spectral.load(pth)
    os.remove(pth)
    local dvc2 = enc2:encode({ offsets = dv_o, tokens = dv_t, values = dv_v, n_samples = dv_n })
    local nchk = dv_codes:size(); if nchk > 100000 then nchk = 100000 end
    for i = 0, nchk - 1 do
      assert(dv_codes:get(i) == dvc2:get(i), "persist/load parity mismatch at " .. i)
    end
    str.printf("[Persist] load parity OK (%d codes)\n", nchk)
  end

  local dvp_off, dvp_nbr = ridge_obj:label(dv_codes, dv_n, 1)
  local _, dv_stats = eval.label_accuracy({
    pred_offsets = dvp_off, pred_neighbors = dvp_nbr,
    expected_offsets = dv_loff, expected_neighbors = dv_lnbr, ks = 1,
  })
  str.printf("[Params] kernel=%s lambda=%.4e\n", best_params.kernel, best_params.lambda)
  str.printf("[Dev] token-acc=%.4f %s\n", dv_stats.micro_f1, sw())

  offsets = nil; tokens = nil; values = nil -- luacheck: ignore
  collectgarbage("collect")

  local _, te_o, te_t, te_v = tokenize(test_set, te_off, te_s, te_e, te_ty, ngram_map)
  csr.apply_bns(te_o, te_t, te_v, bns)
  csr.normalize(te_o, te_v)
  local te_codes = sp_enc:encode({ offsets = te_o, tokens = te_t, values = te_v, n_samples = te_n })
  local tep_off, tep_nbr = ridge_obj:label(te_codes, te_n, 1)
  local _, te_tok = eval.label_accuracy({
    pred_offsets = tep_off, pred_neighbors = tep_nbr,
    expected_offsets = te_loff, expected_neighbors = te_lnbr, ks = 1,
  })
  -- deterministic collapse of predicted token classes -> spans, scored vs gold entity spans
  local po, ps, pe, pty = csr.bio_decode(te_off, te_s, te_e, tep_nbr, N_TYPES)
  local f1, p, r = eval.span_f1(po, ps, pe, pty, te_eoff, te_es, te_ee, te_ety)
  str.printf("[Test] token-acc=%.4f | span: f1=%.4f p=%.4f r=%.4f %s\n",
    te_tok.micro_f1, f1, p, r, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

  -- TODO: learned collapse on top of the deterministic decode. Instead of committing to the argmax
  -- BIO run-merge, over-generate candidate spans (e.g. spans between tokens carrying non-trivial
  -- B/I mass, up to a max length) and run the L1/L2 accept-reject stack (cf. conll-ner-aho:
  -- optimize.oof OOF backdrop + spans-collapse doc context) to LEARN which segmentation to keep.
  -- This is where span-level + doc-context features can beat per-token BIO on ambiguous boundaries
  -- (e.g. [New York][Times] vs [New York Times], adjacent same-type entities).

end)
