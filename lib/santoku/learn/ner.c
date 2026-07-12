#include <santoku/lua/utils.h>
#include <santoku/spans.h>
#include <santoku/ivec.h>
#include <santoku/svec.h>
#include <santoku/fvec.h>
#include <santoku/csr.h>
#include <santoku/klib.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

static inline void tk_ner_cols3 (
  lua_State *L, tk_spans_t *S, int64_t *s, int64_t *e, int64_t *ty, const char *msg
) {
  *s = tk_spans_colidx(S, "s");
  *e = tk_spans_colidx(S, "e");
  if (ty) *ty = tk_spans_colidx(S, "ty");
  if (*s < 0 || *e < 0 || (ty && *ty < 0))
    tk_lua_verror(L, 2, "ner", msg);
}

static inline void tk_ner_push_int_array (
  lua_State *L, const int64_t *a, int64_t n, const char *field
) {
  lua_newtable(L);
  for (int64_t i = 0; i < n; i ++) {
    lua_pushinteger(L, (lua_Integer) a[i]);
    lua_rawseti(L, -2, (int) i);
  }
  lua_setfield(L, -2, field);
}

static int tk_ner_type_labels (lua_State *L)
{
  lua_settop(L, 3);
  tk_spans_t *C = tk_spans_peek(L, 1, "cand");
  tk_spans_t *G = tk_spans_peek(L, 2, "gold");
  int64_t n_types = luaL_checkinteger(L, 3);
  int64_t cs, ce, gs, ge, gt;
  tk_ner_cols3(L, C, &cs, &ce, NULL, "type_labels requires cand{s,e} gold{s,e,ty}");
  tk_ner_cols3(L, G, &gs, &ge, &gt, "type_labels requires cand{s,e} gold{s,e,ty}");
  int64_t n_docs = (int64_t) tk_spans_docs(C);
  int64_t ncand = (int64_t) tk_spans_n(C);
  tk_ivec_t *out = tk_ivec_create(L, (uint64_t) ncand);
  out->n = (uint64_t) ncand;
  int64_t *Cs = C->cols[cs]->a, *Ce = C->cols[ce]->a;
  int64_t *Gs = G->cols[gs]->a, *Ge = G->cols[ge]->a, *Gt = G->cols[gt]->a;
  for (int64_t d = 0; d < n_docs; d ++)
    for (int64_t c = C->offsets->a[d]; c < C->offsets->a[d + 1]; c ++) {
      int64_t a = Cs[c], b = Ce[c], lab = n_types;
      for (int64_t g = G->offsets->a[d]; g < G->offsets->a[d + 1]; g ++)
        if (Gs[g] == a && Ge[g] == b) { lab = Gt[g]; break; }
      out->a[c] = lab;
    }
  return 1;
}

static inline tk_spans_t *tk_ner_field_spans (lua_State *L, int t, const char *name)
{
  lua_getfield(L, t, name);
  tk_spans_t *S = tk_spans_peek(L, -1, name);
  lua_pop(L, 1);
  return S;
}

static int tk_ner_miss_report (lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  tk_spans_t *GZ = tk_ner_field_spans(L, 1, "gaz");
  tk_spans_t *BI = tk_ner_field_spans(L, 1, "bio");
  tk_spans_t *GO = tk_ner_field_spans(L, 1, "gold");
  lua_getfield(L, 1, "n_types");
  int64_t n_types = tk_lua_checkinteger(L, -1, "n_types");
  lua_pop(L, 1);
  tk_spans_t *src[2] = { GZ, BI };
  int64_t scs[2], sce[2], sct[2];
  for (int s = 0; s < 2; s ++)
    tk_ner_cols3(L, src[s], &scs[s], &sce[s], &sct[s], "miss_report sources require {s,e,ty}");
  int64_t gcs, gce, gct;
  tk_ner_cols3(L, GO, &gcs, &gce, &gct, "miss_report gold requires {s,e,ty}");
  int64_t *go = GO->offsets->a, *gs = GO->cols[gcs]->a, *ge = GO->cols[gce]->a, *gt = GO->cols[gct]->a;
  int64_t n_docs = (int64_t) tk_spans_docs(GO);
  int64_t n_gold = 0, covered = 0, wrong = 0, over = 0, under = 0, cross = 0, none = 0;
  int64_t under_gaz = 0, under_bio = 0, under_both = 0;
  int64_t *under_ty = (int64_t *) calloc((size_t) (n_types > 0 ? n_types : 1), sizeof(int64_t));
  if (!under_ty) return tk_lua_verror(L, 2, "ner", "miss_report: alloc failed");
  for (int64_t d = 0; d < n_docs; d ++) {
    for (int64_t g = go[d]; g < go[d + 1]; g ++) {
      int64_t a = gs[g], b = ge[g], t = gt[g];
      n_gold ++;
      int exact = 0, fwrong = 0, fover = 0, fcross = 0;
      int funder_s[2] = { 0, 0 };
      for (int s = 0; s < 2 && !exact; s ++) {
        int64_t *so = src[s]->offsets->a, *ss = src[s]->cols[scs[s]]->a;
        int64_t *se = src[s]->cols[sce[s]]->a, *st = src[s]->cols[sct[s]]->a;
        for (int64_t c = so[d]; c < so[d + 1]; c ++) {
          int64_t x = ss[c], y = se[c];
          if (x >= b || a >= y) continue;
          if (x == a && y == b) {
            if (st[c] == t) { exact = 1; break; }
            fwrong = 1;
          } else if (x <= a && b <= y) {
            fover = 1;
          } else if (a <= x && y <= b) {
            funder_s[s] = 1;
          } else {
            fcross = 1;
          }
        }
      }
      int funder = funder_s[0] || funder_s[1];
      if (exact) covered ++;
      else if (fwrong) wrong ++;
      else if (fover) over ++;
      else if (funder) {
        under ++;
        if (funder_s[0] && funder_s[1]) under_both ++;
        else if (funder_s[0]) under_gaz ++;
        else under_bio ++;
        if (t >= 0 && t < n_types) under_ty[t] ++;
      }
      else if (fcross) cross ++;
      else none ++;
    }
  }
  lua_newtable(L);
  lua_pushinteger(L, (lua_Integer) n_gold); lua_setfield(L, -2, "gold");
  lua_pushinteger(L, (lua_Integer) covered); lua_setfield(L, -2, "covered");
  lua_pushinteger(L, (lua_Integer) wrong); lua_setfield(L, -2, "wrong_type");
  lua_pushinteger(L, (lua_Integer) over); lua_setfield(L, -2, "over");
  lua_pushinteger(L, (lua_Integer) under); lua_setfield(L, -2, "under");
  lua_pushinteger(L, (lua_Integer) cross); lua_setfield(L, -2, "cross");
  lua_pushinteger(L, (lua_Integer) none); lua_setfield(L, -2, "none");
  lua_pushinteger(L, (lua_Integer) under_gaz); lua_setfield(L, -2, "under_gaz");
  lua_pushinteger(L, (lua_Integer) under_bio); lua_setfield(L, -2, "under_bio");
  lua_pushinteger(L, (lua_Integer) under_both); lua_setfield(L, -2, "under_both");
  tk_ner_push_int_array(L, under_ty, n_types, "under_by_type");
  free(under_ty);
  return 1;
}

static int tk_ner_decode_report (lua_State *L)
{
  luaL_checktype(L, 1, LUA_TTABLE);
  tk_spans_t *C = tk_ner_field_spans(L, 1, "cand");
  tk_spans_t *GO = tk_ner_field_spans(L, 1, "gold");
  lua_getfield(L, 1, "pred");
  tk_ivec_t *cp = tk_ivec_peek(L, -1, "pred");
  lua_pop(L, 1);
  lua_getfield(L, 1, "pred_stride");
  int64_t stride = tk_lua_checkinteger(L, -1, "pred_stride");
  lua_pop(L, 1);
  lua_getfield(L, 1, "n_types");
  int64_t n_types = tk_lua_checkinteger(L, -1, "n_types");
  lua_pop(L, 1);
  int64_t cs, ce, gs, ge, gt;
  tk_ner_cols3(L, C, &cs, &ce, NULL, "decode_report requires cand{s,e} gold{s,e,ty}");
  tk_ner_cols3(L, GO, &gs, &ge, &gt, "decode_report requires cand{s,e} gold{s,e,ty}");
  int64_t *co = C->offsets->a, *Cs = C->cols[cs]->a, *Ce = C->cols[ce]->a;
  int64_t *go = GO->offsets->a, *Gs = GO->cols[gs]->a, *Ge = GO->cols[ge]->a, *Gt = GO->cols[gt]->a;
  int64_t n_docs = (int64_t) tk_spans_docs(GO);
  int64_t n_gold = 0, in_pool = 0, not_in_pool = 0, correct = 0, freject = 0, mistype = 0;
  size_t nt = (size_t) (n_types > 0 ? n_types : 1);
  int64_t *corr_ty = (int64_t *) calloc(nt, sizeof(int64_t));
  int64_t *rej_ty = (int64_t *) calloc(nt, sizeof(int64_t));
  int64_t *mis_ty = (int64_t *) calloc(nt, sizeof(int64_t));
  int64_t *conf = (int64_t *) calloc(nt * nt, sizeof(int64_t));
  if (!corr_ty || !rej_ty || !mis_ty || !conf) {
    free(corr_ty); free(rej_ty); free(mis_ty); free(conf);
    return tk_lua_verror(L, 2, "ner", "decode_report: alloc failed");
  }
  for (int64_t d = 0; d < n_docs; d ++) {
    for (int64_t g = go[d]; g < go[d + 1]; g ++) {
      int64_t a = Gs[g], b = Ge[g], t = Gt[g];
      n_gold ++;
      int64_t cmatch = -1;
      for (int64_t c = co[d]; c < co[d + 1]; c ++)
        if (Cs[c] == a && Ce[c] == b) { cmatch = c; break; }
      if (cmatch < 0) { not_in_pool ++; continue; }
      in_pool ++;
      int64_t pred = cp->a[cmatch * stride];
      if (pred == n_types) { freject ++; if (t >= 0 && t < n_types) rej_ty[t] ++; }
      else if (pred == t) { correct ++; if (t >= 0 && t < n_types) corr_ty[t] ++; }
      else {
        mistype ++;
        if (t >= 0 && t < n_types) {
          mis_ty[t] ++;
          if (pred >= 0 && pred < n_types) conf[t * n_types + pred] ++;
        }
      }
    }
  }
  lua_newtable(L);
  lua_pushinteger(L, (lua_Integer) n_gold); lua_setfield(L, -2, "gold");
  lua_pushinteger(L, (lua_Integer) in_pool); lua_setfield(L, -2, "in_pool");
  lua_pushinteger(L, (lua_Integer) not_in_pool); lua_setfield(L, -2, "not_in_pool");
  lua_pushinteger(L, (lua_Integer) correct); lua_setfield(L, -2, "correct");
  lua_pushinteger(L, (lua_Integer) freject); lua_setfield(L, -2, "false_reject");
  lua_pushinteger(L, (lua_Integer) mistype); lua_setfield(L, -2, "mistype");
  tk_ner_push_int_array(L, corr_ty, n_types, "correct_by_type");
  tk_ner_push_int_array(L, rej_ty, n_types, "reject_by_type");
  tk_ner_push_int_array(L, mis_ty, n_types, "mistype_by_type");
  tk_ner_push_int_array(L, conf, n_types * n_types, "confusion");
  free(corr_ty); free(rej_ty); free(mis_ty); free(conf);
  return 1;
}

KHASH_MAP_INIT_STR(tk_gazmap, int64_t)

#define TK_GAZ_MT "tk_gaz_t"

typedef struct {
  int64_t n_types, is_char, nmin, nmax, nkeys, cap;
  int64_t *data;
  khash_t(tk_gazmap) *map;
} tk_gaz_t;

static inline tk_gaz_t *tk_gaz_peek (lua_State *L, int i) {
  return (tk_gaz_t *) luaL_checkudata(L, i, TK_GAZ_MT);
}

static int tk_gaz_gc (lua_State *L) {
  tk_gaz_t *G = (tk_gaz_t *) luaL_checkudata(L, 1, TK_GAZ_MT);
  if (G->map != NULL) {
    for (khint_t k = kh_begin(G->map); k != kh_end(G->map); k ++)
      if (kh_exist(G->map, k)) free((void *) kh_key(G->map, k));
    kh_destroy(tk_gazmap, G->map);
    free(G->map);
    G->map = NULL;
  }
  free(G->data); G->data = NULL;
  return 0;
}

static luaL_Reg tk_gaz_mt_fns[];

static inline int64_t *tk_gaz_rowptr (tk_gaz_t *G, int64_t idx) {
  return G->data + idx * (G->n_types + 1);
}

static inline char *tk_gaz_lower (char *tmp, const char *key, size_t len) {
  for (size_t i = 0; i < len; i ++) tmp[i] = (char) tolower((unsigned char) key[i]);
  tmp[len] = 0;
  return tmp;
}

static int64_t tk_gaz_row (tk_gaz_t *G, const char *key, size_t len) {
  char *tmp = tk_gaz_lower((char *) malloc(len + 1), key, len);
  int ret;
  khint_t k = kh_put(tk_gazmap, G->map, tmp, &ret);
  if (ret == 0) { free(tmp); return kh_val(G->map, k); }
  int64_t idx = G->nkeys ++;
  if (G->nkeys > G->cap) {
    G->cap = G->cap ? G->cap * 2 : 1024;
    G->data = (int64_t *) realloc(G->data,
      (size_t) G->cap * (size_t) (G->n_types + 1) * sizeof(int64_t));
  }
  memset(tk_gaz_rowptr(G, idx), 0, (size_t) (G->n_types + 1) * sizeof(int64_t));
  kh_val(G->map, k) = idx;
  return idx;
}

static int64_t tk_gaz_find (tk_gaz_t *G, const char *key, size_t len, char *sb, size_t sbcap) {
  char *tmp = tk_gaz_lower((len + 1 <= sbcap) ? sb : (char *) malloc(len + 1), key, len);
  khint_t k = kh_get(tk_gazmap, G->map, tmp);
  int64_t idx = (k == kh_end(G->map)) ? -1 : kh_val(G->map, k);
  if (tmp != sb) free(tmp);
  return idx;
}

static int tk_gaz_build (lua_State *L, int is_char) {
  luaL_checktype(L, 1, LUA_TTABLE);
  lua_getfield(L, 1, "texts");
  luaL_checktype(L, -1, LUA_TTABLE);
  int texts = lua_gettop(L);
  tk_spans_t *G = tk_ner_field_spans(L, 1, "gold");
  lua_getfield(L, 1, "n_types");
  int64_t n_types = tk_lua_checkinteger(L, -1, "n_types");
  lua_pop(L, 1);
  int64_t nmin = 0, nmax = 0;
  if (is_char) {
    lua_getfield(L, 1, "ngram_min"); nmin = tk_lua_checkinteger(L, -1, "ngram_min"); lua_pop(L, 1);
    lua_getfield(L, 1, "ngram_max"); nmax = tk_lua_checkinteger(L, -1, "ngram_max"); lua_pop(L, 1);
  }
  int64_t gs, ge, gt;
  tk_ner_cols3(L, G, &gs, &ge, &gt, "build gaz requires gold{s,e,ty}");
  tk_gaz_t *Z = tk_lua_newuserdata(L, tk_gaz_t, TK_GAZ_MT, tk_gaz_mt_fns, tk_gaz_gc);
  memset(Z, 0, sizeof(*Z));
  Z->n_types = n_types; Z->is_char = is_char; Z->nmin = nmin; Z->nmax = nmax;
  Z->map = (khash_t(tk_gazmap) *) malloc(sizeof(khash_t(tk_gazmap)));
  kh_init(tk_gazmap, Z->map, 0);
  int64_t *Gs = G->cols[gs]->a, *Ge = G->cols[ge]->a, *Gt = G->cols[gt]->a, *Go = G->offsets->a;
  int64_t nd = (int64_t) tk_spans_docs(G);
  for (int64_t d = 0; d < nd; d ++) {
    lua_rawgeti(L, texts, (int) (d + 1));
    size_t tl; const char *t = lua_tolstring(L, -1, &tl);
    for (int64_t g = Go[d]; g < Go[d + 1]; g ++) {
      int64_t a = Gs[g], b = Ge[g], ty = Gt[g];
      const char *surf = t + a; size_t sl = (size_t) (b - a);
      if (ty < 0 || ty >= n_types) continue;
      if (!is_char) {
        int64_t *row = tk_gaz_rowptr(Z, tk_gaz_row(Z, surf, sl));
        row[0] ++; row[1 + ty] ++;
      } else {
        for (int64_t n = nmin; n <= nmax; n ++)
          for (size_t i = 0; i + (size_t) n <= sl; i ++) {
            int64_t *row = tk_gaz_rowptr(Z, tk_gaz_row(Z, surf + i, (size_t) n));
            row[0] ++; row[1 + ty] ++;
          }
      }
    }
    lua_pop(L, 1);
  }
  return 1;
}

static int tk_gaz_build_typed_lua (lua_State *L) { return tk_gaz_build(L, 0); }
static int tk_gaz_build_char_lua (lua_State *L) { return tk_gaz_build(L, 1); }

static int tk_gaz_block (lua_State *L) {
  lua_settop(L, 4);
  tk_gaz_t *Z = tk_gaz_peek(L, 1);
  luaL_checktype(L, 2, LUA_TTABLE);
  int texts = 2;
  tk_spans_t *S = tk_spans_peek(L, 3, "cand");
  tk_ivec_t *tlab = lua_isnil(L, 4) ? NULL : tk_ivec_peek(L, 4, "tlab");
  int64_t nt = Z->n_types;
  int64_t cs, ce;
  tk_ner_cols3(L, S, &cs, &ce, NULL, "gaz:block requires cand{s,e}");
  int64_t *Cs = S->cols[cs]->a, *Ce = S->cols[ce]->a, *Co = S->offsets->a;
  int64_t nd = (int64_t) tk_spans_docs(S);
  tk_ivec_t *off = tk_ivec_create(L, 0); int off_idx = lua_gettop(L); tk_ivec_push(off, 0);
  tk_svec_t *nbr = tk_svec_create(L, 0); int nbr_idx = lua_gettop(L);
  tk_fvec_t *val = tk_fvec_create(L, 0); int val_idx = lua_gettop(L);
  double *acc = (double *) malloc((size_t) (nt > 0 ? nt : 1) * sizeof(double));
  char sb[256];
  for (int64_t d = 0; d < nd; d ++) {
    lua_rawgeti(L, texts, (int) (d + 1));
    size_t tl; const char *t = lua_tolstring(L, -1, &tl);
    for (int64_t ci = Co[d]; ci < Co[d + 1]; ci ++) {
      int64_t a = Cs[ci], b = Ce[ci];
      const char *surf = t + a; size_t sl = (size_t) (b - a);
      int64_t g = tlab ? tlab->a[ci] : nt;
      int64_t own = (g >= 0 && g < nt) ? 1 : 0;
      if (!Z->is_char) {
        int64_t r = tk_gaz_find(Z, surf, sl, sb, sizeof sb);
        if (r >= 0) {
          int64_t *row = tk_gaz_rowptr(Z, r);
          int64_t den = row[0] - own;
          if (den > 0)
            for (int64_t ty = 0; ty < nt; ty ++) {
              int64_t cnt = row[1 + ty] - ((ty == g) ? own : 0);
              if (cnt > 0) { tk_svec_push(nbr, (int32_t) ty); tk_fvec_push(val, (float) ((double) cnt / (double) den)); }
            }
        }
      } else {
        for (int64_t ty = 0; ty < nt; ty ++) acc[ty] = 0.0;
        for (int64_t n = Z->nmin; n <= Z->nmax; n ++)
          for (size_t i = 0; i + (size_t) n <= sl; i ++) {
            int64_t r = tk_gaz_find(Z, surf + i, (size_t) n, sb, sizeof sb);
            if (r < 0) continue;
            int64_t *row = tk_gaz_rowptr(Z, r);
            int64_t den = row[0] - own;
            if (den <= 0) continue;
            for (int64_t ty = 0; ty < nt; ty ++) {
              int64_t cnt = row[1 + ty] - ((ty == g) ? own : 0);
              if (cnt > 0) acc[ty] += (double) cnt / (double) den;
            }
          }
        for (int64_t ty = 0; ty < nt; ty ++)
          if (acc[ty] > 0.0) { tk_svec_push(nbr, (int32_t) ty); tk_fvec_push(val, (float) acc[ty]); }
      }
      tk_ivec_push(off, (int64_t) nbr->n);
    }
    lua_pop(L, 1);
  }
  free(acc);
  tk_csr_push(L, TK_TAG_F32, TK_TAG_I32, (uint64_t) nt,
    off_idx, off, nbr_idx, (void *) nbr, val_idx, (void *) val);
  return 1;
}

// Embedded persist: header scalars, then data rows (nkeys x n_types+1 int64) in row order, then each
// key (int64 len + bytes) in row order. Load rebuilds the string map by re-inserting key i at row i.
static int tk_gaz_persist (lua_State *L) {
  tk_gaz_t *Z = tk_gaz_peek(L, 1);
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 2), "w");
  tk_lua_fwrite(L, "TKgz", 1, 4, fh);
  uint8_t version = 1;
  tk_lua_fwrite(L, &version, sizeof(uint8_t), 1, fh);
  int64_t hdr[5] = { Z->n_types, Z->is_char, Z->nmin, Z->nmax, Z->nkeys };
  tk_lua_fwrite(L, hdr, sizeof(int64_t), 5, fh);
  int64_t width = Z->n_types + 1;
  if (Z->nkeys > 0)
    tk_lua_fwrite(L, Z->data, sizeof(int64_t), (size_t) (Z->nkeys * width), fh);
  const char **keys = (const char **) calloc((size_t) (Z->nkeys > 0 ? Z->nkeys : 1), sizeof(char *));
  for (khint_t k = kh_begin(Z->map); k != kh_end(Z->map); k ++)
    if (kh_exist(Z->map, k)) keys[kh_val(Z->map, k)] = kh_key(Z->map, k);
  for (int64_t i = 0; i < Z->nkeys; i ++) {
    int64_t len = (int64_t) strlen(keys[i]);
    tk_lua_fwrite(L, &len, sizeof(int64_t), 1, fh);
    if (len > 0) tk_lua_fwrite(L, (void *) keys[i], 1, (size_t) len, fh);
  }
  free(keys);
  tk_lua_fclose(L, fh);
  return 0;
}

static int tk_gaz_load_lua (lua_State *L) {
  FILE *fh = tk_lua_fopen(L, luaL_checkstring(L, 1), "r");
  char magic[4];
  tk_lua_fread(L, magic, 1, 4, fh);
  if (memcmp(magic, "TKgz", 4) != 0) { tk_lua_fclose(L, fh); return luaL_error(L, "ner.load_gaz: bad magic"); }
  uint8_t version;
  tk_lua_fread(L, &version, sizeof(uint8_t), 1, fh);
  if (version != 1) { tk_lua_fclose(L, fh);
    return luaL_error(L, "ner.load_gaz: unsupported version %d", (int) version); }
  int64_t hdr[5];
  tk_lua_fread(L, hdr, sizeof(int64_t), 5, fh);
  tk_gaz_t *Z = tk_lua_newuserdata(L, tk_gaz_t, TK_GAZ_MT, tk_gaz_mt_fns, tk_gaz_gc);
  memset(Z, 0, sizeof(*Z));
  Z->n_types = hdr[0]; Z->is_char = hdr[1]; Z->nmin = hdr[2]; Z->nmax = hdr[3]; Z->nkeys = hdr[4];
  Z->cap = Z->nkeys;
  int64_t width = Z->n_types + 1;
  if (Z->nkeys > 0) {
    Z->data = (int64_t *) malloc((size_t) (Z->nkeys * width) * sizeof(int64_t));
    tk_lua_fread(L, Z->data, sizeof(int64_t), (size_t) (Z->nkeys * width), fh);
  }
  Z->map = (khash_t(tk_gazmap) *) malloc(sizeof(khash_t(tk_gazmap)));
  kh_init(tk_gazmap, Z->map, 0);
  for (int64_t i = 0; i < Z->nkeys; i ++) {
    int64_t len;
    tk_lua_fread(L, &len, sizeof(int64_t), 1, fh);
    char *key = (char *) malloc((size_t) len + 1);
    if (len > 0) tk_lua_fread(L, key, 1, (size_t) len, fh);
    key[len] = 0;
    int ret;
    khint_t k = kh_put(tk_gazmap, Z->map, key, &ret);
    kh_val(Z->map, k) = i;
  }
  tk_lua_fclose(L, fh);
  return 1;
}

static luaL_Reg tk_gaz_mt_fns[] = {
  { "block", tk_gaz_block },
  { "persist", tk_gaz_persist },
  { NULL, NULL }
};

static luaL_Reg tk_ner_spans_mt_fns[] = {
  { "type_labels", tk_ner_type_labels },
  { NULL, NULL }
};

static luaL_Reg tk_ner_module_fns[] = {
  { "miss_report", tk_ner_miss_report },
  { "decode_report", tk_ner_decode_report },
  { "build_typed_gaz", tk_gaz_build_typed_lua },
  { "build_char_gaz", tk_gaz_build_char_lua },
  { "load_gaz", tk_gaz_load_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_ner (lua_State *L)
{
  tk_lua_require_mod(L, "santoku.spans");
  tk_lua_extend_mt(L, TK_SPANS_MT, tk_ner_spans_mt_fns);
  lua_newtable(L);
  luaL_register(L, NULL, tk_ner_module_fns);
  return 1;
}
