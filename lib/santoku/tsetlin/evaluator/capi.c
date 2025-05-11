#include "lua.h"
#include "lauxlib.h"
#include "../conf.h"

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

static inline void *tk_lua_checkuserdata (lua_State *L, int i, char *mt)
{
  if (mt == NULL && (lua_islightuserdata(L, i) || lua_isuserdata(L, i)))
    return lua_touserdata(L, i);
  void *p = luaL_checkudata(L, -1, mt);
  lua_pop(L, 1);
  return p;
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
  memset(f1, 0, sizeof(double) * n_classes);

  double precision[n_classes];
  memset(precision, 0, sizeof(double) * n_classes);

  double recall[n_classes];
  memset(recall, 0, sizeof(double) * n_classes);

  double f1_avg = 0.0;
  double precision_avg = 0.0;
  double recall_avg = 0.0;

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
  unsigned int n_hidden = tk_lua_checkunsigned(L, 3, "n_hidden");
  unsigned int n_codes = tk_lua_checkunsigned(L, 4, "n_codes");

  if (predicted_len != n_codes * BITS_DIV(n_hidden) * n_codes)
    tk_lua_verror(L, 3, "encoding_accuracy", "predicted", "invalid data length");

  if (expected_len != n_codes * BITS_DIV(n_hidden) * n_codes)
    tk_lua_verror(L, 3, "encoding_accuracy", "expected", "invalid data length");

  double f1[n_hidden];
  memset(f1, 0, sizeof(double) * n_hidden);

  double precision[n_hidden];
  memset(precision, 0, sizeof(double) * n_hidden);

  double recall[n_hidden];
  memset(recall, 0, sizeof(double) * n_hidden);

  double f1_avg = 0.0;
  double precision_avg = 0.0;
  double recall_avg = 0.0;

  // TODO: calculate per-bit and overall precision, recall, and f1

  lua_newtable(L);

  lua_newtable(L);
  for (unsigned int i = 0; i < n_hidden; i ++) {
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

static inline void tm_print_hamming_histogram (
  const tk_bits_t *codes,
  const tm_pair_t *pos,
  const tm_pair_t *neg,
  uint64_t n_pos,
  uint64_t n_neg,
  uint64_t n_sentences,
  uint64_t n_hidden
) {
  const uint64_t chunks = BITS_DIV(n_hidden);
  const uint64_t max_dist = n_hidden + 1;
  uint64_t *hist_pos = calloc(max_dist, sizeof(uint64_t));
  uint64_t *hist_neg = calloc(max_dist, sizeof(uint64_t));
  for (uint64_t k = 0; k < n_pos; k ++) {
    int64_t u = pos[k].u, v = pos[k].v;
    uint64_t d = hamming(codes + u * chunks, codes + v * chunks, chunks);
    if (d <= n_hidden) hist_pos[d] ++;
  }
  for (uint64_t k = 0; k < n_neg; k ++) {
    int64_t u = neg[k].u, v = neg[k].v;
    uint64_t d = hamming(codes + u * chunks, codes + v * chunks, chunks);
    if (d <= n_hidden) hist_neg[d] ++;
  }
  printf("Hamming Distance Histogram (Positive vs. Negative):\n");
  for (uint64_t d = 0; d <= n_hidden; d ++)
    if (hist_pos[d] || hist_neg[d])
      printf("Distance %3lu: Pos %6lu  |  Neg %6lu\n", d, hist_pos[d], hist_neg[d]);
  free(hist_pos);
  free(hist_neg);
  for (uint64_t i = 0; i < ((n_sentences < 16) ? n_sentences : 16); i ++) {
    printf("Sample %lu: ", i);
    for (uint64_t j = 0; j < n_hidden; j ++) {
      uint64_t chunk = BITS_DIV(j);
      uint64_t bit = BITS_MOD(j);
      printf("%s", (codes[i * BITS_DIV(n_hidden) + chunk] & ((tk_bits_t) 1 << bit)) ? "1" : "0");
    }
    printf("\n");
  }
}

static inline int tm_encoding_similarity (lua_State *L)
{
  lua_settop(L, 5);

  size_t codes_len;
  tk_bits_t *codes = (tk_bits_t *) tk_lua_checklstring(L, 1, &codes_len, "codes");

  lua_pushvalue(L, 2);
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *pos = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_pos = (uint64_t) luaL_checkinteger(L, -1) / 2;

  lua_pushvalue(L, 3);
  tk_lua_callmod(L, 1, 4, "santoku.matrix.integer", "view");
  tm_pair_t *neg = (tm_pair_t *) tk_lua_checkuserdata(L, -4, NULL);
  uint64_t n_neg = (uint64_t) luaL_checkinteger(L, -1) / 2;

  uint64_t n_sentences = tk_lua_checkunsigned(L, 4, "n_sentences");
  uint64_t n_hidden = tk_lua_checkunsigned(L, 5, "n_hidden");

  tm_print_hamming_histogram(codes, pos, neg, n_pos, n_neg, n_sentences, n_hidden);

  if (codes_len != n_sentences * BITS_DIV(n_hidden) * sizeof(tk_bits_t))
    tk_lua_verror(L, 3, "encoding_similarity", "codes",  "invalid data length");

  tm_dl_t *pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  uint64_t chunks = BITS_DIV(n_hidden);

  // Calculate AUC
  for (uint64_t k = 0; k < n_pos + n_neg; k ++) {
    tm_pair_t *pairs = k < n_pos ? pos : neg;
    uint64_t offset = k < n_pos ? 0 : n_pos;
    int64_t u = pairs[k - offset].u;
    int64_t v = pairs[k - offset].v;
    pl[k].sim = (uint64_t) n_hidden - hamming(codes + u * chunks, codes + v * chunks, chunks);
    pl[k].label = k < n_pos ? 1 : 0;
  }
  ks_introsort(dl, n_pos + n_neg, pl);
  double sum_ranks = 0.0;
  unsigned int rank = 1;
  for (uint64_t k = 0; k < n_pos + n_neg; k ++, rank ++)
    if (pl[k].label)
      sum_ranks += rank;
  double auc = (sum_ranks - ((double) n_pos * (n_pos + 1) / 2)) / ((double) n_pos * n_neg);

  // Calculate total for best margin calculation
  uint64_t hist_pos[n_hidden + 1];
  uint64_t hist_neg[n_hidden + 1];
  memset(hist_pos, 0, (n_hidden + 1) * sizeof(uint64_t));
  memset(hist_neg, 0, (n_hidden + 1) * sizeof(uint64_t));
  for (uint64_t k = 0; k < n_pos + n_neg; k ++) {
    uint64_t s = pl[k].sim;
    if (s > n_hidden) s = n_hidden;
    if (pl[k].label) hist_pos[s]++;
    else hist_neg[s] ++;
  }

  // Find best margin for f1
  double best_f1 = -1.0, best_prec = 0.0, best_rec = 0.0;
  uint64_t best_margin = 0, cum_tp = 0, cum_fp = 0;
  for (int64_t m = (int64_t) n_hidden; m >= 0; m --) {
    cum_tp += hist_pos[m];
    cum_fp += hist_neg[m];
    double prec = (cum_tp + cum_fp) > 0 ? (double) cum_tp / (cum_tp + cum_fp) : 0.0;
    double rec = (double) cum_tp / (double) n_pos;
    double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
    if (f1 > best_f1) {
      best_f1 = f1;
      best_prec = prec;
      best_rec = rec;
      best_margin = (uint64_t) m;
    }
  }

  // Cleanup
  free(pl);

  // Output
  lua_newtable(L);
  lua_pushinteger(L, (int64_t) best_margin);
  lua_setfield(L, -2, "margin");
  lua_pushnumber(L, best_prec);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, best_rec);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, best_f1);
  lua_setfield(L, -2, "f1");
  lua_pushnumber(L, auc);
  lua_setfield(L, -2, "auc");
  return 1;
}

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
