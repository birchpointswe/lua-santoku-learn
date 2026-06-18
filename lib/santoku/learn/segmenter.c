#include <santoku/lua/utils.h>
#include <santoku/learn/mathlibs.h>
#include <santoku/ivec.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

// Marker-enriched byte dendrogram with a JOINT lexicographic merge objective. train() builds a
// PPMI co-occurrence embedding over 258 symbols (256 bytes + gold span markers [B]/[E]); the embedding
// direction is set by create({context}): "both" = left(+)right halves, "left" = predecessors only.
// then agglomerates the occurring bytes by a lexicographic cost: PRIMARY = gold boundaries the merge
// would break (cluster boundary-count, summed), TIE-BREAK = cosine complete-linkage distance (the PPMI
// shape-quality regularizer). Boundary-preserving (BB==0) merges run first, so the coarsest
// PERFECT-RECALL cut sits exactly before the first boundary-breaking merge -> coarse_k. One tree, two
// consumers: segment({drop_sep=true}) = the segmentation units (coarse_k, separators dropped),
// segment({k=ortho_k}) = the ortho-class stream for the shape block.

#define TK_SEGMENTER_MT "tk_segmenter_t"

typedef struct {
  int trained;
  int left_only;         // embedding direction: 1 = predecessor distribution only, 0 = left(+)right
  uint8_t seen[256];     // bytes occurring in train
  int nseen;
  int coarse_k;          // perfect-recall cut (segmentation)
  int sep_class;         // learned separator class at coarse_k (lowest gold-insideness)
  int *coph;             // 256*256 cophenetic merge steps (malloc'd)
} tk_segmenter_t;

static inline tk_segmenter_t *tk_segmenter_peek (lua_State *L, int i) {
  return (tk_segmenter_t *) luaL_checkudata(L, i, TK_SEGMENTER_MT);
}

static luaL_Reg tk_segmenter_mt_fns[];
static int tk_segmenter_gc (lua_State *L);

// create({ context = "both" | "left" }) -- embedding direction (default "both": left(+)right halves).
static int tk_segmenter_create_lua (lua_State *L) {
  luaL_checktype(L, 1, LUA_TTABLE);
  tk_segmenter_t *s = tk_lua_newuserdata(L, tk_segmenter_t, TK_SEGMENTER_MT,
    tk_segmenter_mt_fns, tk_segmenter_gc);
  memset(s, 0, sizeof(*s));
  lua_getfield(L, 1, "context");
  if (!lua_isnil(L, -1)) {
    const char *cx = lua_tostring(L, -1);
    if (cx && strcmp(cx, "left") == 0) s->left_only = 1;
    else if (cx && strcmp(cx, "both") == 0) s->left_only = 0;
    else return luaL_error(L, "segmenter: context must be 'both'|'left'");
  }
  lua_pop(L, 1);
  return 1;
}

// class map at the cut after S merges: two seen bytes co-cluster iff coph<=S; rep = smallest such
// member; clusters labelled by ascending representative; absent bytes share one catch-all class.
static int tk_cut_classmap (tk_segmenter_t *s, int S, uint8_t *out) {
  int rep[256];
  for (int v = 0; v < 256; v++) {
    if (!s->seen[v]) continue;
    int r = v;
    for (int u = 0; u < v; u++) { if (!s->seen[u]) continue; int c = s->coph[u * 256 + v]; if (c >= 1 && c <= S) { r = u; break; } }
    rep[v] = r;
  }
  int id[256], k = 0;
  for (int v = 0; v < 256; v++) if (s->seen[v] && rep[v] == v) id[v] = k++;
  int has_absent = s->nseen < 256;
  for (int v = 0; v < 256; v++) out[v] = s->seen[v] ? (uint8_t) id[rep[v]] : (uint8_t) k;
  return k + (has_absent ? 1 : 0);
}

// ---------------------------------------------------------------------------
// train: marker embedding + lexicographic (boundary-broken, PPMI) agglomeration +
// perfect-recall cut
// ---------------------------------------------------------------------------
static int tk_segmenter_train_lua (lua_State *L) {
  tk_segmenter_t *s = tk_segmenter_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int n = (int) tk_lua_fcheckunsigned(L, 2, "train", "n");
  lua_getfield(L, 2, "texts"); luaL_checktype(L, -1, LUA_TTABLE);
  int ti = lua_gettop(L);
  lua_getfield(L, 2, "gold_offsets"); tk_ivec_t *go = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "gold_starts");  tk_ivec_t *gst = tk_ivec_peekopt(L, -1); lua_pop(L, 1);
  lua_getfield(L, 2, "gold_ends");    tk_ivec_t *gen = tk_ivec_peekopt(L, -1); lua_pop(L, 1);

  const char **tp = (const char **) malloc((size_t) n * sizeof(char *));
  size_t *tl = (size_t *) malloc((size_t) n * sizeof(size_t));
  for (int d = 0; d < n; d++) { lua_rawgeti(L, ti, d + 1); tp[d] = lua_tolstring(L, -1, &tl[d]); lua_pop(L, 1); }

  const size_t NSYM = 258, B_MARK = 256, E_MARK = 257;
  double *Cr = (double *) calloc(NSYM * NSYM, sizeof(double));
  memset(s->seen, 0, sizeof(s->seen));
  size_t maxlen = 1; for (int d = 0; d < n; d++) if (tl[d] > maxlen) maxlen = tl[d];
  uint8_t *smark = (uint8_t *) malloc(maxlen), *emark = (uint8_t *) malloc(maxlen);
  int64_t *aug = (int64_t *) malloc((maxlen * 3 + 4) * sizeof(int64_t));
  int64_t ngold = (go && gst && gen) ? go->a[n] : 0;
  int *su = (int *) malloc((size_t) (ngold > 0 ? ngold : 1) * sizeof(int));
  int *sv = (int *) malloc((size_t) (ngold > 0 ? ngold : 1) * sizeof(int));
  int *eu = (int *) malloc((size_t) (ngold > 0 ? ngold : 1) * sizeof(int));
  int *ev = (int *) malloc((size_t) (ngold > 0 ? ngold : 1) * sizeof(int));
  int nspan = 0;

  for (int d = 0; d < n; d++) {
    const unsigned char *b = (const unsigned char *) tp[d]; size_t len = tl[d];
    if (len == 0) continue;
    for (size_t i = 0; i < len; i++) s->seen[b[i]] = 1;
    memset(smark, 0, len); memset(emark, 0, len);
    if (go && gst && gen)
      for (int64_t gi = go->a[d]; gi < go->a[d + 1]; gi++) {
        int64_t gs = gst->a[gi], ge = gen->a[gi];
        if (ge <= gs || (size_t) ge > len) continue;
        smark[gs] = 1; emark[ge - 1] = 1;
        su[nspan] = gs > 0 ? b[gs - 1] : 256; sv[nspan] = (size_t) gs < len ? b[gs] : 256;
        eu[nspan] = ge > 0 ? b[ge - 1] : 256; ev[nspan] = (size_t) ge < len ? b[ge] : 256;
        nspan++;
      }
    size_t m = 0;
    for (size_t i = 0; i < len; i++) {
      if (smark[i]) aug[m++] = (int64_t) B_MARK;
      aug[m++] = (int64_t) b[i];
      if (emark[i]) aug[m++] = (int64_t) E_MARK;
    }
    for (size_t j = 0; j + 1 < m; j++)
      Cr[(size_t) aug[j] * NSYM + (size_t) aug[j + 1]] += 1.0;
  }

  // PPMI reweight
  double *rm = (double *) calloc(NSYM, sizeof(double)), *cmg = (double *) calloc(NSYM, sizeof(double)), tot = 0;
  for (size_t u = 0; u < NSYM; u++) for (size_t w = 0; w < NSYM; w++) { double c = Cr[u * NSYM + w]; rm[u] += c; cmg[w] += c; tot += c; }
  for (size_t u = 0; u < NSYM; u++) for (size_t w = 0; w < NSYM; w++) {
    size_t idx = u * NSYM + w; double c = Cr[idx];
    double pmi = (c > 0 && rm[u] > 0 && cmg[w] > 0) ? log(c * tot / (rm[u] * cmg[w])) : 0.0;
    Cr[idx] = pmi > 0 ? pmi : 0.0;
  }
  free(rm); free(cmg);

  // position-aware embedding + cosine distance over the 256 byte rows: row[u] = Cr[u][v] = count of u
  // preceding v (LEFT/predecessor distribution); when not left_only, row[NSYM+u] = Cr[v][u] = successors
  // (RIGHT). left_only halves D and drops the right context (better case separation for the shape cut).
  size_t D = s->left_only ? NSYM : 2 * NSYM;
  double *E = (double *) calloc(256 * D, sizeof(double));
  for (size_t v = 0; v < 256; v++) {
    double *row = E + v * D;
    for (size_t u = 0; u < NSYM; u++) {
      row[u] = Cr[u * NSYM + v];
      if (!s->left_only) row[NSYM + u] = Cr[v * NSYM + u];
    }
    double nn = cblas_dnrm2((int) D, row, 1);
    if (nn > 0) for (size_t j = 0; j < D; j++) row[j] /= nn;
  }
  double *G = (double *) malloc(256 * 256 * sizeof(double));
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasNoTrans, 256, (int) D, 1.0, E, (int) D, 0.0, G, 256);
  double *dist = (double *) malloc(256 * 256 * sizeof(double));
  for (size_t a = 0; a < 256; a++) for (size_t b = 0; b < 256; b++) {
    size_t lo = a <= b ? a : b, hi = a <= b ? b : a;
    dist[a * 256 + b] = 1.0 - G[lo * 256 + hi];
  }

  // boundary-breaking matrix BB[a][b] = # gold edges whose two flanking byte VALUES are a and b.
  // self (su==sv) and doc-edge (==256) flanks are excluded: a byte->class map can never separate a
  // byte from itself, so those boundaries are inherently unpreservable and must not gate the cut.
  double *BB = (double *) calloc(256 * 256, sizeof(double));
  for (int i = 0; i < nspan; i++) {
    if (su[i] != 256 && sv[i] != 256 && su[i] != sv[i]) { BB[su[i] * 256 + sv[i]] += 1.0; BB[sv[i] * 256 + su[i]] += 1.0; }
    if (eu[i] != 256 && ev[i] != 256 && eu[i] != ev[i]) { BB[eu[i] * 256 + ev[i]] += 1.0; BB[ev[i] * 256 + eu[i]] += 1.0; }
  }

  // lexicographic complete-linkage agglomeration: PRIMARY cost = gold boundaries broken by the merge
  // (cluster BB, summed additively like Lance-Williams); TIE-BREAK = cosine complete-linkage distance
  // (max). Boundary-preserving merges (BB==0) are exhausted first, so the coarsest perfect-recall cut
  // sits exactly before the first BB>0 merge. PPMI/cosine only orders the boundary-indifferent merges.
  int nseen = 0; for (int i = 0; i < 256; i++) if (s->seen[i]) nseen++;
  s->nseen = nseen;
  free(s->coph); s->coph = (int *) calloc(256 * 256, sizeof(int));
  int *coph = s->coph;
  double *step_bb = (double *) malloc((size_t) (nseen > 0 ? nseen : 1) * sizeof(double));
  int alive[256], memn[256]; int *mem_store = (int *) malloc(256 * 256 * sizeof(int)), *mem[256];
  for (int i = 0; i < 256; i++) { alive[i] = s->seen[i] ? 1 : 0; mem[i] = mem_store + (size_t) i * 256; mem[i][0] = i; memn[i] = 1; }
  for (int step = 0; step < nseen - 1; step++) {
    double bbest = INFINITY, dbest = INFINITY; int ba = -1, bx = -1;
    for (int a = 0; a < 256; a++) if (alive[a]) for (int b = a + 1; b < 256; b++) if (alive[b]) {
      double bv = BB[(size_t) a * 256 + (size_t) b], dd = dist[(size_t) a * 256 + (size_t) b];
      if (bv < bbest - 1e-9 || (bv <= bbest + 1e-9 && dd < dbest)) { bbest = bv; dbest = dd; ba = a; bx = b; }
    }
    step_bb[step] = bbest;
    for (int x = 0; x < memn[ba]; x++) for (int y = 0; y < memn[bx]; y++) { int u = mem[ba][x], w = mem[bx][y]; coph[u * 256 + w] = coph[w * 256 + u] = step + 1; }
    for (int c = 0; c < 256; c++) if (alive[c] && c != ba && c != bx) {
      double da = dist[(size_t) ba * 256 + (size_t) c], db = dist[(size_t) bx * 256 + (size_t) c], mx = da > db ? da : db;
      dist[(size_t) ba * 256 + (size_t) c] = dist[(size_t) c * 256 + (size_t) ba] = mx;
      BB[(size_t) ba * 256 + (size_t) c] += BB[(size_t) bx * 256 + (size_t) c];
      BB[(size_t) c * 256 + (size_t) ba] += BB[(size_t) c * 256 + (size_t) bx];
    }
    memcpy(mem[ba] + memn[ba], mem[bx], (size_t) memn[bx] * sizeof(int)); memn[ba] += memn[bx]; alive[bx] = 0;
  }

  // coarsest perfect-recall cut = before the first boundary-breaking merge
  int bestS = 0;
  for (int step = 0; step < nseen - 1; step++) { if (step_bb[step] <= 1e-9) bestS = step + 1; else break; }
  s->coarse_k = nseen - bestS;
  s->trained = 1;

  // both-edges span-recall ceiling at the cut: a span is recoverable unless one edge is a self-flank
  // (same byte value both sides, unseparable by any byte->class map). Doc-edge flanks are preserved.
  int n_span_ok = 0;
  for (int i = 0; i < nspan; i++) {
    int s_bad = (su[i] != 256 && su[i] == sv[i]);
    int e_bad = (eu[i] != 256 && eu[i] == ev[i]);
    if (!s_bad && !e_bad) n_span_ok++;
  }
  double recall = nspan > 0 ? (double) n_span_ok / nspan : 1.0;

  // diagnostic: segments-per-gold at the coarse cut
  int maxc = 0, p95 = 0;
  {
    uint8_t cm[256]; tk_cut_classmap(s, bestS, cm);
    int hist[1024]; memset(hist, 0, sizeof(hist)); int nsp = 0;
    for (int d = 0; d < n; d++) {
      const unsigned char *b = (const unsigned char *) tp[d]; size_t len = tl[d];
      if (!go || !gst || !gen) break;
      for (int64_t gi = go->a[d]; gi < go->a[d + 1]; gi++) {
        int64_t g0 = gst->a[gi], g1 = gen->a[gi];
        if (g1 <= g0 || (size_t) g1 > len) continue;
        int cc = 1; for (int64_t i = g0 + 1; i < g1; i++) if (cm[b[i]] != cm[b[i - 1]]) cc++;
        if (cc > 1023) cc = 1023;
        hist[cc]++; nsp++;
      }
    }
    for (int i = 0; i < 1024; i++) if (hist[i]) maxc = i;
    int cum = 0, thr = (int) (0.95 * nsp);
    for (int i = 0; i < 1024; i++) { cum += hist[i]; if (cum >= thr) { p95 = i; break; } }
  }

  // learned separator class: the coarse-cut class with the lowest gold-insideness fraction (its byte
  // occurrences sit mostly between/around entities). drop_sep omits its cells -- the byte-native,
  // charclass-free replacement for a whitespace test. (smark reused as the inside-gold mask.)
  int sep_class = 0;
  {
    uint8_t cm[256]; tk_cut_classmap(s, bestS, cm);
    int ncl = 0; for (int v = 0; v < 256; v++) if (cm[v] + 1 > ncl) ncl = cm[v] + 1;
    double *ins = (double *) calloc((size_t) (ncl > 0 ? ncl : 1), sizeof(double));
    double *tot = (double *) calloc((size_t) (ncl > 0 ? ncl : 1), sizeof(double));
    if (go && gst && gen)
      for (int d = 0; d < n; d++) {
        const unsigned char *b = (const unsigned char *) tp[d]; size_t len = tl[d];
        if (len == 0) continue;
        memset(smark, 0, len);
        for (int64_t gi = go->a[d]; gi < go->a[d + 1]; gi++) {
          int64_t g0 = gst->a[gi], g1 = gen->a[gi];
          if (g1 <= g0 || (size_t) g1 > len) continue;
          for (int64_t i = g0; i < g1; i++) smark[i] = 1;
        }
        for (size_t i = 0; i < len; i++) { int c = cm[b[i]]; tot[c] += 1.0; if (smark[i]) ins[c] += 1.0; }
      }
    double best = 1e30;
    for (int c = 0; c < ncl; c++) { double fr = tot[c] > 0 ? ins[c] / tot[c] : 1e30; if (fr < best) { best = fr; sep_class = c; } }
    free(ins); free(tot);
  }
  s->sep_class = sep_class;

  free(tp); free(tl); free(Cr); free(E); free(G); free(dist); free(smark); free(emark); free(aug);
  free(su); free(sv); free(eu); free(ev); free(BB); free(step_bb); free(mem_store);
  lua_pushinteger(L, s->coarse_k);
  lua_pushnumber(L, recall);
  lua_pushinteger(L, maxc);
  lua_pushinteger(L, p95);
  lua_pushinteger(L, s->sep_class);
  return 5;
}

// ---------------------------------------------------------------------------
// shared emit: cut at k, run-length-encode each doc's byte-class stream into (offsets, starts, ends,
// classes). When drop_sep, cells whose bytes belong to the learned separator class AT THE COARSE CUT are
// omitted (the dendrogram is hierarchical, so a finer k-class sits within one coarse class -> sampling
// the run's first byte's coarse class suffices). At k == coarse_k this equals the old rc == sep_class
// test. Reads texts/n from the table at stack index 2; returns the 4 ivecs.
static int tk_segmenter_emit (lua_State *L, tk_segmenter_t *s, int k, int drop_sep) {
  luaL_checktype(L, 2, LUA_TTABLE);
  int n = (int) tk_lua_fcheckunsigned(L, 2, "segment", "n");
  if (k < 1) k = 1;
  if (k > s->nseen) k = s->nseen;
  lua_getfield(L, 2, "texts"); luaL_checktype(L, -1, LUA_TTABLE);
  int ti = lua_gettop(L);

  const char **tp = (const char **) malloc((size_t) n * sizeof(char *));
  size_t *tl = (size_t *) malloc((size_t) n * sizeof(size_t));
  for (int d = 0; d < n; d++) { lua_rawgeti(L, ti, d + 1); tp[d] = lua_tolstring(L, -1, &tl[d]); lua_pop(L, 1); }

  uint8_t cm[256]; tk_cut_classmap(s, s->nseen - k, cm);
  uint8_t cmc[256]; if (drop_sep) tk_cut_classmap(s, s->nseen - s->coarse_k, cmc);

  tk_ivec_t *off = tk_ivec_create(L, 0), *st = tk_ivec_create(L, 0),
            *en = tk_ivec_create(L, 0), *cl = tk_ivec_create(L, 0);
  tk_ivec_push(off, 0);
  for (int d = 0; d < n; d++) {
    const unsigned char *b = (const unsigned char *) tp[d]; size_t len = tl[d];
    if (len > 0) {
      int64_t run_s = 0; int rc = cm[b[0]];
      for (size_t i = 1; i < len; i++) {
        if (cm[b[i]] != rc) {
          if (!(drop_sep && cmc[b[run_s]] == s->sep_class)) {
            tk_ivec_push(st, run_s); tk_ivec_push(en, (int64_t) i); tk_ivec_push(cl, rc);
          }
          run_s = (int64_t) i; rc = cm[b[i]];
        }
      }
      if (!(drop_sep && cmc[b[run_s]] == s->sep_class)) {
        tk_ivec_push(st, run_s); tk_ivec_push(en, (int64_t) len); tk_ivec_push(cl, rc);
      }
    }
    tk_ivec_push(off, (int64_t) st->n);
  }
  free(tp); free(tl);
  return 4;
}

// segment({texts, n, k, drop_sep}) -> offsets, starts, ends, classes  (cut at k; default coarse_k;
// drop_sep omits the coarse separator-class cells, default keep all). drop_sep at the default cut =
// the byte-native content segmentation for L1.
static int tk_segmenter_segment_lua (lua_State *L) {
  tk_segmenter_t *s = tk_segmenter_peek(L, 1);
  if (!s->trained) return luaL_error(L, "segmenter: segment before train");
  int k = (int) tk_lua_foptunsigned(L, 2, "segment", "k", (unsigned) s->coarse_k);
  int drop_sep = tk_lua_foptboolean(L, 2, "segment", "drop_sep", 0);
  return tk_segmenter_emit(L, s, k, drop_sep);
}

// compression_curve({texts, n}) -> ratios (Lua array, ratios[k] = avg over docs of bytes/RLE-cells at the
// k-cluster cut), nseen. No re-segmentation: adjacent bytes share a run at the cut after S=nseen-k merges
// iff their cophenetic level <= S, so per-doc RLE cells = 1 + #{adjacent pairs with coph > S}. One pass
// per doc builds a local merge-level histogram; a suffix sum gives cells (hence ratio) at every cut.
static int tk_segmenter_compression_curve_lua (lua_State *L) {
  tk_segmenter_t *s = tk_segmenter_peek(L, 1);
  if (!s->trained) return luaL_error(L, "segmenter: compression_curve before train");
  luaL_checktype(L, 2, LUA_TTABLE);
  int n = (int) tk_lua_fcheckunsigned(L, 2, "compression_curve", "n");
  lua_getfield(L, 2, "texts"); luaL_checktype(L, -1, LUA_TTABLE);
  int ti = lua_gettop(L);
  int nseen = s->nseen; int *coph = s->coph;
  double *rsum = (double *) calloc((size_t) (nseen + 1), sizeof(double));
  int64_t *Hd = (int64_t *) malloc((size_t) (nseen + 1) * sizeof(int64_t));
  int64_t *suf = (int64_t *) malloc((size_t) (nseen + 1) * sizeof(int64_t));
  int64_t ndoc = 0;
  for (int d = 0; d < n; d++) {
    lua_rawgeti(L, ti, d + 1);
    size_t len; const unsigned char *b = (const unsigned char *) lua_tolstring(L, -1, &len);
    lua_pop(L, 1);
    if (len < 1) continue;
    ndoc++;
    for (int l = 0; l <= nseen; l++) Hd[l] = 0;
    for (size_t i = 1; i < len; i++) {
      int a = b[i - 1], c = b[i], sa = s->seen[a], sc = s->seen[c], lvl;
      if (sa && sc) lvl = (a == c) ? 0 : coph[a * 256 + c];   // same byte = same class at every cut
      else if (!sa && !sc) lvl = 0;                            // both absent = the shared catch-all class
      else lvl = nseen;                                        // absent vs seen = always a boundary
      Hd[lvl]++;
    }
    suf[nseen] = 0;
    for (int S = nseen - 1; S >= 0; S--) suf[S] = suf[S + 1] + Hd[S + 1];   // suf[S] = #pairs with coph > S
    for (int k = 1; k <= nseen; k++) {
      int64_t cells = 1 + suf[nseen - k];
      rsum[k] += (double) len / (double) cells;
    }
  }
  lua_newtable(L);
  for (int k = 1; k <= nseen; k++) {
    lua_pushnumber(L, ndoc > 0 ? rsum[k] / (double) ndoc : 0.0);
    lua_rawseti(L, -2, k);
  }
  free(rsum); free(Hd); free(suf);
  lua_pushinteger(L, nseen);
  return 2;
}

static int tk_segmenter_persist_lua (lua_State *L) {
  tk_segmenter_t *s = tk_segmenter_peek(L, 1);
  if (!s->trained) return luaL_error(L, "segmenter: persist before train");
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "SGd2", 1, 4, fh);
  tk_lua_fwrite(L, &s->nseen, sizeof(int), 1, fh);
  tk_lua_fwrite(L, &s->coarse_k, sizeof(int), 1, fh);
  tk_lua_fwrite(L, &s->sep_class, sizeof(int), 1, fh);
  tk_lua_fwrite(L, s->seen, 1, 256, fh);
  tk_lua_fwrite(L, s->coph, sizeof(int), 256 * 256, fh);
  tk_lua_fclose(L, fh);
  return 0;
}

static int tk_segmenter_load_lua (lua_State *L) {
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 1), "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "SGd2", 4) != 0) { tk_lua_fclose(L, fh); return luaL_error(L, "segmenter.load: bad magic"); }
  tk_segmenter_t *s = tk_lua_newuserdata(L, tk_segmenter_t, TK_SEGMENTER_MT,
    tk_segmenter_mt_fns, tk_segmenter_gc);
  memset(s, 0, sizeof(*s));
  tk_lua_fread(L, &s->nseen, sizeof(int), 1, fh);
  tk_lua_fread(L, &s->coarse_k, sizeof(int), 1, fh);
  tk_lua_fread(L, &s->sep_class, sizeof(int), 1, fh);
  tk_lua_fread(L, s->seen, 1, 256, fh);
  s->coph = (int *) malloc(256 * 256 * sizeof(int));
  tk_lua_fread(L, s->coph, sizeof(int), 256 * 256, fh);
  tk_lua_fclose(L, fh);
  s->trained = 1;
  return 1;
}

static int tk_segmenter_gc (lua_State *L) {
  tk_segmenter_t *s = (tk_segmenter_t *) luaL_checkudata(L, 1, TK_SEGMENTER_MT);
  free(s->coph); s->coph = NULL;
  return 0;
}

static luaL_Reg tk_segmenter_mt_fns[] = {
  { "train", tk_segmenter_train_lua },
  { "segment", tk_segmenter_segment_lua },
  { "compression_curve", tk_segmenter_compression_curve_lua },
  { "persist", tk_segmenter_persist_lua },
  { NULL, NULL }
};

static luaL_Reg tk_segmenter_fns[] = {
  { "create", tk_segmenter_create_lua },
  { "load", tk_segmenter_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_segmenter (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_segmenter_fns, 0);
  return 1;
}
