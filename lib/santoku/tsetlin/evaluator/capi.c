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

  if (n_classes == 0)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "must be > 0");

  uint64_t TP[n_classes], FP[n_classes], FN[n_classes];
  memset(TP, 0, sizeof(TP));
  memset(FP, 0, sizeof(FP));
  memset(FN, 0, sizeof(FN));

  for (unsigned int i = 0; i < n_samples; i ++) {
    unsigned int y_pred = predicted[i];
    unsigned int y_true = expected[i];
    if (y_pred >= n_classes || y_true >= n_classes)
      continue;
    if (y_pred == y_true)
      TP[y_true] ++;
    else {
      FP[y_pred] ++;
      FN[y_true] ++;
    }
  }

  double precision[n_classes], recall[n_classes], f1[n_classes];
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (unsigned int c = 0; c < n_classes; c ++) {
    uint64_t tp = TP[c], fp = FP[c], fn = FN[c];
    precision[c] = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
    recall[c] = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
    f1[c] = (precision[c] + recall[c]) > 0 ? 2.0 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0.0;
    precision_avg += precision[c];
    recall_avg += recall[c];
    f1_avg += f1[c];
  }

  precision_avg /= n_classes;
  recall_avg /= n_classes;
  f1_avg /= n_classes;

  // Lua output
  lua_newtable(L);
  lua_newtable(L);
  for (unsigned int c = 0; c < n_classes; c ++) {
    lua_pushinteger(L, c + 1);
    lua_newtable(L);
    lua_pushnumber(L, precision[c]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, recall[c]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, f1[c]);
    lua_setfield(L, -2, "f1");
    lua_settable(L, -3);  // result.classes[c+1] = {...}
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

static inline int tm_codebook_stats (lua_State *L)
{
  lua_settop(L, 3);
  size_t codes_len;
  tk_bits_t *codes = (tk_bits_t *) tk_lua_checklstring(L, 1, &codes_len, "expected");
  unsigned int n_codes = tk_lua_checkunsigned(L, 2, "n_codes");
  unsigned int n_hidden = tk_lua_checkunsigned(L, 3, "n_hidden");
  uint64_t chunks = BITS_DIV(n_hidden);
  if (codes_len != n_codes * chunks * sizeof(tk_bits_t))
    tk_lua_verror(L, 3, "codebook_stats", "codes", "invalid data length");
  uint64_t *bit_counts = tk_malloc(L, n_hidden * sizeof(uint64_t));
  memset(bit_counts, 0, n_hidden * sizeof(uint64_t));
  // Count number of 1s per bit across all codes
  for (uint64_t i = 0; i < n_codes; i++) {
    for (uint64_t j = 0; j < n_hidden; j++) {
      uint64_t word = j / BITS;
      uint64_t bit  = j % BITS;
      if ((codes[i * chunks + word] >> bit) & 1)
        bit_counts[j]++;
    }
  }
  // Compute per-bit entropy
  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_hidden; j++) {
    double p = (double) bit_counts[j] / (double) n_codes;
    double entropy = 0.0;
    if (p > 0.0 && p < 1.0)
      entropy = -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    lua_pushinteger(L, (int64_t) j + 1);
    lua_pushnumber(L, entropy);
    lua_settable(L, -3);
    if (entropy < min_entropy)
      min_entropy = entropy;
    if (entropy > max_entropy)
      max_entropy = entropy;
    sum_entropy += entropy;
  }
  lua_setfield(L, -2, "bits");
  // Aggregate stats
  double mean = sum_entropy / n_hidden;
  double variance = 0.0;
  for (uint64_t j = 0; j < n_hidden; j++) {
    double p = (double) bit_counts[j] / (double) n_codes;
    double entropy = 0.0;
    if (p > 0.0 && p < 1.0)
      entropy = -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    variance += (entropy - mean) * (entropy - mean);
  }
  variance /= n_hidden;
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, min_entropy);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_entropy);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, sqrt(variance));
  lua_setfield(L, -2, "std");
  free(bit_counts);
  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  lua_settop(L, 4);
  size_t predicted_len, expected_len;
  tk_bits_t *codes_predicted = (tk_bits_t *) tk_lua_checklstring(L, 1, &predicted_len, "predicted");
  tk_bits_t *codes_expected = (tk_bits_t *) tk_lua_checklstring(L, 2, &expected_len, "expected");
  unsigned int n_codes = tk_lua_checkunsigned(L, 3, "n_codes");
  unsigned int n_hidden = tk_lua_checkunsigned(L, 4, "n_hidden");

  uint64_t chunks = BITS_DIV(n_hidden);
  if (predicted_len != n_codes * chunks * sizeof(tk_bits_t))
    tk_lua_verror(L, 1, "encoding_accuracy", "predicted", "invalid data length");

  if (expected_len != n_codes * chunks * sizeof(tk_bits_t))
    tk_lua_verror(L, 2, "encoding_accuracy", "expected", "invalid data length");

  uint64_t TP[n_hidden], FP[n_hidden], FN[n_hidden];
  memset(TP, 0, sizeof(TP));
  memset(FP, 0, sizeof(FP));
  memset(FN, 0, sizeof(FN));

  for (uint64_t i = 0; i < n_codes; i ++) {
    for (uint64_t j = 0; j < n_hidden; j ++) {
      uint64_t word = j / (sizeof(tk_bits_t) * CHAR_BIT);
      uint64_t bit = j % (sizeof(tk_bits_t) * CHAR_BIT);
      bool y_true = (codes_expected[i * chunks + word] >> bit) & 1;
      bool y_pred = (codes_predicted[i * chunks + word] >> bit) & 1;
      if (y_pred && y_true) TP[j] ++;
      else if (y_pred && !y_true) FP[j] ++;
      else if (!y_pred && y_true) FN[j] ++;
    }
  }

  double f1[n_hidden], precision[n_hidden], recall[n_hidden];
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;

  for (uint64_t j = 0; j < n_hidden; j ++) {
    uint64_t tp = TP[j], fp = FP[j], fn = FN[j];
    precision[j] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
    recall[j] = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
    f1[j] = (precision[j] + recall[j]) > 0 ? 2.0 * precision[j] * recall[j] / (precision[j] + recall[j]) : 0.0;
    precision_avg += precision[j];
    recall_avg += recall[j];
    f1_avg += f1[j];
  }

  precision_avg /= n_hidden;
  recall_avg /= n_hidden;
  f1_avg /= n_hidden;

  // Output to Lua
  lua_newtable(L);
  lua_newtable(L); // classes
  for (uint64_t j = 0; j < n_hidden; j ++) {
    lua_pushinteger(L, (int64_t) j + 1);
    lua_newtable(L);
    lua_pushnumber(L, precision[j]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, recall[j]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, f1[j]);
    lua_setfield(L, -2, "f1");
    lua_settable(L, -3);
  }
  lua_setfield(L, -2, "classes");
  lua_pushnumber(L, precision_avg);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, recall_avg);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, f1_avg);
  lua_setfield(L, -2, "f1");
  double f1_var = 0.0;
  double min_f1 = 1.0, max_f1 = 0.0;
  for (uint64_t j = 0; j < n_hidden; j++) {
    double f = f1[j];
    f1_var += (f - f1_avg) * (f - f1_avg);
    if (f < min_f1) min_f1 = f;
    if (f > max_f1) max_f1 = f;
  }
  f1_var /= n_hidden;
  lua_pushnumber(L, min_f1);
  lua_setfield(L, -2, "f1_min");
  lua_pushnumber(L, max_f1);
  lua_setfield(L, -2, "f1_max");
  lua_pushnumber(L, sqrt(f1_var));
  lua_setfield(L, -2, "f1_std");
  return 1;
}

// static inline void tm_print_hamming_histogram (
//   const tk_bits_t *codes,
//   const tm_pair_t *pos,
//   const tm_pair_t *neg,
//   uint64_t n_pos,
//   uint64_t n_neg,
//   uint64_t n_sentences,
//   uint64_t n_hidden
// ) {
//   const uint64_t chunks = BITS_DIV(n_hidden);
//   const uint64_t max_dist = n_hidden + 1;
//   uint64_t *hist_pos = calloc(max_dist, sizeof(uint64_t));
//   uint64_t *hist_neg = calloc(max_dist, sizeof(uint64_t));
//   for (uint64_t k = 0; k < n_pos; k ++) {
//     int64_t u = pos[k].u, v = pos[k].v;
//     uint64_t d = hamming(codes + u * chunks, codes + v * chunks, chunks);
//     if (d <= n_hidden) hist_pos[d] ++;
//   }
//   for (uint64_t k = 0; k < n_neg; k ++) {
//     int64_t u = neg[k].u, v = neg[k].v;
//     uint64_t d = hamming(codes + u * chunks, codes + v * chunks, chunks);
//     if (d <= n_hidden) hist_neg[d] ++;
//   }
//   printf("Hamming Distance Histogram (Positive vs. Negative):\n");
//   for (uint64_t d = 0; d <= n_hidden; d ++)
//     if (hist_pos[d] || hist_neg[d])
//       printf("Distance %3lu: Pos %6lu  |  Neg %6lu\n", d, hist_pos[d], hist_neg[d]);
//   free(hist_pos);
//   free(hist_neg);
//   for (uint64_t i = 0; i < ((n_sentences < 16) ? n_sentences : 16); i ++) {
//     printf("Sample %lu: ", i);
//     for (uint64_t j = 0; j < n_hidden; j ++) {
//       uint64_t chunk = BITS_DIV(j);
//       uint64_t bit = BITS_MOD(j);
//       printf("%s", (codes[i * BITS_DIV(n_hidden) + chunk] & ((tk_bits_t) 1 << bit)) ? "1" : "0");
//     }
//     printf("\n");
//   }
// }

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

  // tm_print_hamming_histogram(codes, pos, neg, n_pos, n_neg, n_sentences, n_hidden);

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
    pl[k].sim = (uint64_t) n_hidden - hamming(codes + (uint64_t) u * chunks, codes + (uint64_t) v * chunks, chunks);
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
  { "codebook_stats", tm_codebook_stats },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}
