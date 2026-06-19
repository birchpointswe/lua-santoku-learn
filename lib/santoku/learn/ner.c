#include <santoku/spans.h>
#include <santoku/ivec.h>
#include <stdlib.h>

// Span-level NER diagnostics + the type-stage label assignment, registered against the
// matrix-owned spans type. type_labels is a spans method; the two reports are module fns
// (they consume multiple span sets). All spans are [start,end) with columns "s","e"[,"ty"].

// cand:type_labels(gold, n_types) -> ivec[ncand]: per candidate, the gold type if its
// (s,e) exactly matches a gold span in the same doc, else reject (= n_types).
static int tk_ner_type_labels (lua_State *L)
{
  lua_settop(L, 3);
  tk_spans_t *C = tk_spans_peek(L, 1, "cand");
  tk_spans_t *G = tk_spans_peek(L, 2, "gold");
  int64_t n_types = luaL_checkinteger(L, 3);
  int64_t cs = tk_spans_colidx(C, "s"), ce = tk_spans_colidx(C, "e");
  int64_t gs = tk_spans_colidx(G, "s"), ge = tk_spans_colidx(G, "e"), gt = tk_spans_colidx(G, "ty");
  if (cs < 0 || ce < 0 || gs < 0 || ge < 0 || gt < 0)
    return tk_lua_verror(L, 2, "ner", "type_labels requires cand{s,e} gold{s,e,ty}");
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

// ner.miss_report{ gaz, bio, gold, n_types } -> table: decomposes why each gold span is/isn't
// recovered by the candidate pool (gaz UNION bio). Per missed gold the highest-priority relation
// to any same-doc candidate is counted: covered (exact s,e,ty) / wrong_type (exact s,e) / over
// (gold inside a candidate) / under (candidate inside gold) / cross (overlap) / none. For under,
// which source(s) supplied the inside candidate, plus per-gold-type counts.
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
  for (int s = 0; s < 2; s ++) {
    scs[s] = tk_spans_colidx(src[s], "s");
    sce[s] = tk_spans_colidx(src[s], "e");
    sct[s] = tk_spans_colidx(src[s], "ty");
    if (scs[s] < 0 || sce[s] < 0 || sct[s] < 0)
      return tk_lua_verror(L, 2, "ner", "miss_report sources require {s,e,ty}");
  }
  int64_t gcs = tk_spans_colidx(GO, "s"), gce = tk_spans_colidx(GO, "e"), gct = tk_spans_colidx(GO, "ty");
  if (gcs < 0 || gce < 0 || gct < 0)
    return tk_lua_verror(L, 2, "ner", "miss_report gold requires {s,e,ty}");
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
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i ++) {
    lua_pushinteger(L, (lua_Integer) under_ty[i]);
    lua_rawseti(L, -2, (int) i);
  }
  lua_setfield(L, -2, "under_by_type");
  free(under_ty);
  return 1;
}

// ner.decode_report{ cand, pred, pred_stride, gold, n_types } -> table: for each gold span finds the
// candidate with exact (s,e) and inspects the TYPE head's top-1 (pred[c*pred_stride]; == n_types means
// REJECT). Splits conversion loss (golds in the pool not emitted correctly) into false_reject vs mistype,
// with a gold->pred confusion matrix. Golds with no exact candidate are not_in_pool.
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
  int64_t cs = tk_spans_colidx(C, "s"), ce = tk_spans_colidx(C, "e");
  int64_t gs = tk_spans_colidx(GO, "s"), ge = tk_spans_colidx(GO, "e"), gt = tk_spans_colidx(GO, "ty");
  if (cs < 0 || ce < 0 || gs < 0 || ge < 0 || gt < 0)
    return tk_lua_verror(L, 2, "ner", "decode_report requires cand{s,e} gold{s,e,ty}");
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
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i ++) { lua_pushinteger(L, (lua_Integer) corr_ty[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "correct_by_type");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i ++) { lua_pushinteger(L, (lua_Integer) rej_ty[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "reject_by_type");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types; i ++) { lua_pushinteger(L, (lua_Integer) mis_ty[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "mistype_by_type");
  lua_newtable(L);
  for (int64_t i = 0; i < n_types * n_types; i ++) { lua_pushinteger(L, (lua_Integer) conf[i]); lua_rawseti(L, -2, (int) i); }
  lua_setfield(L, -2, "confusion");
  free(corr_ty); free(rej_ty); free(mis_ty); free(conf);
  return 1;
}

static luaL_Reg tk_ner_spans_mt_fns[] = {
  { "type_labels", tk_ner_type_labels },
  { NULL, NULL }
};

static luaL_Reg tk_ner_module_fns[] = {
  { "miss_report", tk_ner_miss_report },
  { "decode_report", tk_ner_decode_report },
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
