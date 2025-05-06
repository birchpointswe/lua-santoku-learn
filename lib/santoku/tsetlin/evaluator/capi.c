#include "lua.h"
#include "lauxlib.h"
#include "../conf.h"

static inline unsigned int hamming (
  tk_bits_t *a,
  tk_bits_t *b,
  unsigned int chunks
) {
  unsigned int t = 0;
  for (unsigned int i = 0; i < chunks; i ++)
    t += (unsigned int) popcount(a[i] ^ b[i]);
  return t;
}

static inline void tk_lua_callmod (
  lua_State *L,
  int nargs,
  int nret,
  const char *smod,
  const char *sfn
) {
  lua_getglobal(L, "require"); // arg req
  lua_pushstring(L, smod); // arg req smod
  lua_call(L, 1, 1); // arg mod
  lua_pushstring(L, sfn); // args mod sfn
  lua_gettable(L, -2); // args mod fn
  lua_remove(L, -2); // args fn
  lua_insert(L, - nargs - 1); // fn args
  lua_call(L, nargs, nret); // results
}

static inline int tk_error (
  lua_State *L,
  const char *label,
  int err
) {
  lua_pushstring(L, label);
  lua_pushstring(L, strerror(err));
  tk_lua_callmod(L, 2, 0, "santoku.error", "error");
  return 1;
}

static inline void *tk_realloc (
  lua_State *L,
  void *p,
  size_t s
) {
  p = realloc(p, s);
  if (!p) {
    tk_error(L, "realloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline void *tk_malloc (
  lua_State *L,
  size_t s
) {
  void *p = malloc(s);
  if (!p) {
    tk_error(L, "malloc failed", ENOMEM);
    return NULL;
  } else {
    return p;
  }
}

static inline int tk_lua_verror (lua_State *L, int n, ...) {
  va_list args;
  va_start(args, n);
  for (int i = 0; i < n; i ++) {
    const char *str = va_arg(args, const char *);
    lua_pushstring(L, str);
  }
  va_end(args);
  tk_lua_callmod(L, n, 0, "santoku.error", "error");
  return 0;
}

static inline const char *tk_lua_checkstring (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checkstring(L, i);
}

static inline const char *tk_lua_checklstring (lua_State *L, int i, size_t *lenp, char *name)
{
  if (lua_type(L, i) != LUA_TSTRING)
    tk_lua_verror(L, 3, name, "value is not a string");
  return luaL_checklstring(L, i, lenp);
}

static inline unsigned int tk_lua_checkunsigned (lua_State *L, int i, char *name)
{
  if (lua_type(L, i) != LUA_TNUMBER)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  lua_Integer l = luaL_checkinteger(L, i);
  if (l < 0)
    tk_lua_verror(L, 2, name, "value is not a positive integer");
  if (l > UINT_MAX)
    luaL_error(L, "value is too large");
  return (unsigned int) l;
}

static inline void tk_lua_register (lua_State *L, luaL_Reg *regs, int nup)
{
  while (true) {
    if ((*regs).name == NULL)
      break;
    for (int i = 0; i < nup; i ++)
      lua_pushvalue(L, -nup); // t upsa upsb
    lua_pushcclosure(L, (*regs).func, nup); // t upsa fn
    lua_setfield(L, -nup - 2, (*regs).name); // t
    regs ++;
  }
  lua_pop(L, nup);
}

static inline int tm_class_accuracy (lua_State *L)
{
  size_t predicted_len, expected_len;
  unsigned int *predicted = (unsigned int *) tk_lua_checklstring(L, 1, &predicted_len, "predicted");
  unsigned int *expected = (unsigned int *) tk_lua_checklstring(L, 2, &expected_len, "expected");
  unsigned int n_classes = tk_lua_checkunsigned(L, 3, "n_classes");
  unsigned int n_samples = tk_lua_checkunsigned(L, 4, "n_samples");

  if (predicted_len != n_samples * sizeof(unsigned int))
    tk_lua_verror(L, 3, "class_accuracy", "predicted", "invalid data length");

  if (expected_len != predicted_len)
    tk_lua_verror(L, 3, "class_accuracy", "expected", "invalid data length");

  if (n_classes)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "number of classes must be greater than 0");

  double f1[n_classes];
  double precision[n_classes];
  double recall[n_classes];

  double f1_avg;
  double precision_avg;
  double recall_avg;

  // TODO: calculate per-class and overall precision, recall, and f1

  lua_newtable(L);

  lua_newtable(L);
  for (unsigned int i = 0; i < n_classes; i ++) {
    lua_pushinteger(L, i + 1);
    lua_newtable(L);
    lua_pushnumber(L, f1[i]);
    lua_setfield(L, -2, "f1");
    lua_pushnumber(L, precision[i]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, recall[i]);
    lua_setfield(L, -2, "recall");
    lua_settable(L, -3);
  }
  lua_setfield(L, -2, "classes");

  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");

  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  size_t predicted_len, expected_len;
  tk_bits_t *codes_predicted = (tk_bits_t *) tk_lua_checklstring(L, 1, &predicted_len, "predicted");
  tk_bits_t *codes_expected = (tk_bits_t *) tk_lua_checklstring(L, 2, &expected_len, "expected");
  unsigned int n_features = tk_lua_checkunsigned(L, 3, "n_features");
  unsigned int n_codes = tk_lua_checkunsigned(L, 4, "n_codes");

  if (predicted_len != n_codes * BITS_DIV(n_features) * n_codes)
    tk_lua_verror(L, 3, "encoding_accuracy", "predicted", "invalid data length");

  if (expected_len != n_codes * BITS_DIV(n_features) * n_codes)
    tk_lua_verror(L, 3, "encoding_accuracy", "expected", "invalid data length");

  double f1[n_features];
  double precision[n_features];
  double recall[n_features];

  double f1_avg;
  double precision_avg;
  double recall_avg;

  // TODO: calculate per-bit and overall precision, recall, and f1

  lua_newtable(L);

  lua_newtable(L);
  for (unsigned int i = 0; i < n_features; i ++) {
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, f1[i]);
    lua_setfield(L, -3, "f1");
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, precision[i]);
    lua_setfield(L, -3, "precision");
    lua_pushinteger(L, i + 1);
    lua_pushnumber(L, recall[i]);
    lua_setfield(L, -3, "recall");
  }
  lua_setfield(L, -2, "classes");

  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");

  return 1;
}

typedef struct {
  unsigned int dist;
  unsigned char label;
} tm_dl_t;

static int tm_cmp_dist (const void *a, const void *b)
{
  return (((tm_dl_t *) a)->dist < ((tm_dl_t *) b)->dist)
    ? -1 : (((tm_dl_t *) a)->dist > ((tm_dl_t *) b)->dist);
}

static inline int tm_encoding_similarity (lua_State *L)
{
  size_t codes_len, pairs_len, labels_len;
  tk_bits_t *codes = (tk_bits_t *) tk_lua_checklstring(L, 1, &codes_len, "codes");
  unsigned int *pairs = (unsigned int *) tk_lua_checklstring(L, 2, &pairs_len, "pairs");
  tk_bits_t *labels = (tk_bits_t *) tk_lua_checklstring(L, 3, &labels_len, "labels");
  unsigned int n_features = tk_lua_checkunsigned(L, 4, "n_features");
  unsigned int n_pairs = tk_lua_checkunsigned(L, 5, "n_pairs");
  unsigned int n_codes = tk_lua_checkunsigned(L, 6, "n_codes");

  if (codes_len != n_codes * BITS_DIV(n_features) * sizeof(tk_bits_t))
    tk_lua_verror(L, 3, "encoding_similarity", "codes",  "invalid data length");
  if (pairs_len != n_pairs * 2 * sizeof(unsigned int))
    tk_lua_verror(L, 3, "encoding_similarity", "pairs",  "invalid data length");
  if (labels_len < BYTES_DIV(n_pairs))
    tk_lua_verror(L, 3, "encoding_similarity", "labels", "invalid data length");

  /* — build (distance, label) array — */
  tm_dl_t *pl = malloc(n_pairs * sizeof(tm_dl_t));
  unsigned int chunks = BITS_DIV(n_features);

  for (unsigned int k = 0; k < n_pairs; ++k) {
    unsigned int i = pairs[2*k], j = pairs[2*k+1];
    /* compute Hamming distance */
    unsigned int d = 0;
    for (unsigned int c = 0; c < chunks; ++c) {
      tk_bits_t x = codes[i*chunks + c] ^ codes[j*chunks + c];
      d += popcount(x);
    }
    pl[k].dist  = d;
    pl[k].label = (labels[BITS_DIV(k)] & ((tk_bits_t)1 << BITS_MOD(k))) ? 1 : 0;
  }

  /* — compute ROC-AUC — */
  /* sort by ascending distance */
  qsort(pl, n_pairs, sizeof(tm_dl_t), tm_cmp_dist);
  unsigned int n_pos = 0, n_neg = 0;
  for (unsigned int k = 0; k < n_pairs; ++k) {
    if (pl[k].label) ++n_pos; else ++n_neg;
  }
  double auc = 0.5;
  if (n_pos > 0 && n_neg > 0) {
    double sum_ranks = 0.0;
    for (unsigned int k = 0; k < n_pairs; ++k) {
      if (pl[k].label) sum_ranks += (double)(k + 1);
    }
    auc = (sum_ranks - (double)n_pos*(n_pos+1)/2) / ((double)n_pos * n_neg);
  }

  /* — find optimal Hamming margin for max F1 — */
  double best_f1 = -1.0, best_prec = 0, best_rec = 0;
  unsigned int best_margin = 0;
  for (unsigned int m = 0; m <= n_features; ++m) {
    unsigned int TP=0, FP=0, FN=0;
    for (unsigned int k = 0; k < n_pairs; ++k) {
      int pred = (pl[k].dist <= m);
      int lab  = pl[k].label;
      if (pred) { if (lab) ++TP; else ++FP; }
      else      { if (lab) ++FN; }
    }
    double prec = (TP+FP) ? (double)TP/(TP+FP) : 0.0;
    double rec  = (TP+FN) ? (double)TP/(TP+FN) : 0.0;
    double f1   = (prec+rec) ? 2*prec*rec/(prec+rec) : 0.0;
    if (f1 > best_f1) {
      best_f1     = f1;
      best_prec   = prec;
      best_rec    = rec;
      best_margin = m;
    }
  }

  /* — per-bit precision/recall/F1 — */
  double f1[n_features], precision[n_features], recall[n_features];
  double sum_f1 = 0.0, sum_prec = 0.0, sum_rec = 0.0;
  for (unsigned int f = 0; f < n_features; ++f) {
    unsigned int TP=0, FP=0, FN=0;
    unsigned int chunk = f >> 6, pos = f & 63;
    for (unsigned int k = 0; k < n_pairs; ++k) {
      unsigned int i = pairs[2*k], j = pairs[2*k+1];
      int bi = (codes[i*chunks + chunk] >> pos) & 1;
      int bj = (codes[j*chunks + chunk] >> pos) & 1;
      int pred = (bi == bj);
      int lab  = pl[k].label;
      if (pred) { if (lab) ++TP; else ++FP; }
      else      { if (lab) ++FN; }
    }
    double prec = (TP+FP) ? (double)TP/(TP+FP) : 0.0;
    double rec  = (TP+FN) ? (double)TP/(TP+FN) : 0.0;
    double f1v  = (prec+rec) ? 2*prec*rec/(prec+rec) : 0.0;
    precision[f] = prec;
    recall[f]    = rec;
    f1[f]        = f1v;
    sum_prec    += prec;
    sum_rec     += rec;
    sum_f1      += f1v;
  }
  double precision_avg = sum_prec / n_features;
  double recall_avg    = sum_rec  / n_features;
  double f1_avg        = sum_f1   / n_features;

  free(pl);

  /* — build return table — */
  lua_newtable(L);

  /* per-bit ("classes") */
  lua_newtable(L);
  for (unsigned int f = 0; f < n_features; ++f) {
    lua_pushinteger(L, f+1);
    lua_newtable(L);
    lua_pushnumber(L, f1[f]);         lua_setfield(L, -2, "f1");
    lua_pushnumber(L, precision[f]);  lua_setfield(L, -2, "precision");
    lua_pushnumber(L, recall[f]);     lua_setfield(L, -2, "recall");
    lua_settable(L, -3);
  }
  lua_setfield(L, -2, "classes");

  /* overall metrics */
  lua_pushinteger(L, best_margin);     lua_setfield(L, -2, "margin");
  lua_pushnumber(L, precision_avg);    lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);       lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);           lua_setfield(L, -2, "f1");
  lua_pushnumber(L, auc);              lua_setfield(L, -2, "auc");

  return 1;
}

// static inline int tm_encoding_similarity (lua_State *L)
// {
//   size_t codes_len, pairs_len, labels_len;
//   tk_bits_t *codes = (tk_bits_t *) tk_lua_checklstring(L, 1, &codes_len, "codes");
//   unsigned int *pairs = (unsigned int *) tk_lua_checklstring(L, 2, &pairs_len, "pairs");
//   tk_bits_t *labels = (tk_bits_t *) tk_lua_checklstring(L, 3, &labels_len, "labels");
//   unsigned int n_features = tk_lua_checkunsigned(L, 4, "n_features");
//   unsigned int n_pairs = tk_lua_checkunsigned(L, 5, "n_pairs");
//   unsigned int n_codes = tk_lua_checkunsigned(L, 6, "n_codes");

//   if (codes_len != n_codes * BITS_DIV(n_features) * sizeof(tk_bits_t))
//     tk_lua_verror(L, 3, "encoding_similarity", "codes", "invalid data length");

//   if (pairs_len != n_pairs * 2 * sizeof(unsigned int))
//     tk_lua_verror(L, 3, "encoding_similarity", "pairs", "invalid data length");

//   if (labels_len < BYTES_DIV(n_pairs))
//     tk_lua_verror(L, 3, "encoding_similarity", "labels", "invalid data length");

//   double auc;

//   // TODO: calculate ROC-AUC

//   unsigned int margin;

//   // TODO: find optimal margin

//   double f1[n_features];
//   double precision[n_features];
//   double recall[n_features];

//   double f1_avg;
//   double precision_avg;
//   double recall_avg;

//   // TODO: calculate per-bit and averaged precision, recall, and f1

//   lua_newtable(L);

//   lua_newtable(L);
//   for (unsigned int i = 0; i < n_features; i ++) {
//     lua_pushinteger(L, i + 1);
//     lua_newtable(L);
//     lua_pushnumber(L, f1[i]);
//     lua_setfield(L, -2, "f1");
//     lua_pushnumber(L, precision[i]);
//     lua_setfield(L, -2, "precision");
//     lua_pushnumber(L, recall[i]);
//     lua_setfield(L, -2, "recall");
//     lua_settable(L, -3);
//   }
//   lua_setfield(L, -2, "classes");

//   lua_pushinteger(L, margin);
//   lua_setfield(L, -2, "margin");
//   lua_pushnumber(L, precision_avg);
//   lua_setfield(L, -2, "precision");
//   lua_pushnumber(L, recall_avg);
//   lua_setfield(L, -2, "recall");
//   lua_pushnumber(L, f1_avg);
//   lua_setfield(L, -2, "f1");

//   return 1;
// }

static luaL_Reg tm_evaluator_fns[] =
{
  { "class_accuracy", tm_class_accuracy },
  { "encoding_accuracy", tm_encoding_accuracy },
  { "encoding_similarity", tm_encoding_similarity },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}
