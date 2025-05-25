#define _GNU_SOURCE

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/threads.h>

typedef enum {
  TK_EVAL_CLASS_ACCURACY,
} tk_eval_stage_t;

typedef struct {
  atomic_ulong *TP, *FP, *FN;
  double *f1, *precision, *recall;
  unsigned int *predicted, *expected;
  unsigned int n_classes;
} tk_eval_t;

typedef struct {
  tk_eval_t *state;
  uint64_t sfirst, slast;
} tk_eval_thread_t;

static void tk_eval_worker (void *dp, int sig)
{
  tk_eval_thread_t *data = (tk_eval_thread_t *) dp;
  tk_eval_t *state = data->state;
  switch ((tk_eval_stage_t) sig) {
    case TK_EVAL_CLASS_ACCURACY:
      for (unsigned int i = data->sfirst; i <= data->slast; i ++) {
        unsigned int y_pred = state->predicted[i];
        unsigned int y_true = state->expected[i];
        if (y_pred >= state->n_classes || y_true >= state->n_classes)
          continue;
        if (y_pred == y_true) {
          atomic_fetch_add(state->TP + y_true, 1.0);
        } else {
          atomic_fetch_add(state->FP + y_pred, 1.0);
          atomic_fetch_add(state->FN + y_true, 1.0);
        }
      }
      break;
  }
}

static inline int tm_class_accuracy (lua_State *L)
{
  unsigned int *predicted = (unsigned int *) tk_lua_checkustring(L, 1, "predicted");
  unsigned int *expected = (unsigned int *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_classes = tk_lua_checkunsigned(L, 3, "n_classes");
  unsigned int n_samples = tk_lua_checkunsigned(L, 4, "n_samples");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  if (n_classes == 0)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "must be > 0");

  tk_eval_t state;
  state.n_classes = n_classes;
  state.expected = expected;
  state.predicted = predicted;
  state.TP = tk_malloc(L, n_classes * sizeof(atomic_ulong));
  state.FP = tk_malloc(L, n_classes * sizeof(atomic_ulong));
  state.FN = tk_malloc(L, n_classes * sizeof(atomic_ulong));
  state.precision = tk_malloc(L, n_classes * sizeof(double));
  state.recall = tk_malloc(L, n_classes * sizeof(double));
  state.f1 = tk_malloc(L, n_classes * sizeof(double));
  for (uint64_t i = 0; i < n_classes; i ++) {
    atomic_init(state.TP + i, 0);
    atomic_init(state.FP + i, 0);
    atomic_init(state.FN + i, 0);
  }

  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Eval
  tk_threads_signal(pool, TK_EVAL_CLASS_ACCURACY);
  tk_threads_signal(pool, -1);
  tk_threads_destroy(pool);

  // Reduce
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (unsigned int c = 0; c < n_classes; c ++) {
    uint64_t tp = state.TP[c], fp = state.FP[c], fn = state.FN[c];
    state.precision[c] = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0.0;
    state.recall[c] = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0.0;
    state.f1[c] = (state.precision[c] + state.recall[c]) > 0 ?
      2.0 * state.precision[c] * state.recall[c] / (state.precision[c] + state.recall[c]) : 0.0;
    precision_avg += state.precision[c];
    recall_avg += state.recall[c];
    f1_avg += state.f1[c];
  }

  free(state.TP);
  free(state.FP);
  free(state.FN);

  precision_avg /= n_classes;
  recall_avg /= n_classes;
  f1_avg /= n_classes;

  // Lua output
  lua_newtable(L);
  lua_newtable(L);
  for (unsigned int c = 0; c < n_classes; c ++) {
    lua_pushinteger(L, c + 1);
    lua_newtable(L);
    lua_pushnumber(L, state.precision[c]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, state.recall[c]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, state.f1[c]);
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

  free(state.precision);
  free(state.recall);
  free(state.f1);

  return 1;
}

static inline int tm_codebook_stats (lua_State *L)
{
  lua_settop(L, 3);
  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "expected");
  unsigned int n_codes = tk_lua_checkunsigned(L, 2, "n_codes");
  unsigned int n_hidden = tk_lua_checkunsigned(L, 3, "n_hidden");
  uint64_t chunks = BITS_DIV(n_hidden);
  uint64_t *bit_counts = tk_malloc(L, n_hidden * sizeof(uint64_t));
  memset(bit_counts, 0, n_hidden * sizeof(uint64_t));
  // Count number of 1s per bit across all codes
  #pragma omp parallel for
  for (uint64_t i = 0; i < n_codes; i ++) {
    for (uint64_t j = 0; j < n_hidden; j ++) {
      uint64_t word = j / BITS;
      uint64_t bit  = j % BITS;
      if ((codes[i * chunks + word] >> bit) & 1) {
        #pragma omp atomic
        bit_counts[j] ++;
      }
    }
  }
  // Compute per-bit entropy
  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_hidden; j ++) {
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
  for (uint64_t j = 0; j < n_hidden; j ++) {
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
  tk_bits_t *codes_predicted = (tk_bits_t *) tk_lua_checkustring(L, 1, "predicted");
  tk_bits_t *codes_expected = (tk_bits_t *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_codes = tk_lua_checkunsigned(L, 3, "n_codes");
  unsigned int n_hidden = tk_lua_checkunsigned(L, 4, "n_hidden");

  uint64_t chunks = BITS_DIV(n_hidden);

  uint64_t *TP = tk_malloc(L, n_hidden * sizeof(uint64_t));
  uint64_t *FP = tk_malloc(L, n_hidden * sizeof(uint64_t));
  uint64_t *FN = tk_malloc(L, n_hidden * sizeof(uint64_t));
  memset(TP, 0, n_hidden * sizeof(uint64_t));
  memset(FP, 0, n_hidden * sizeof(uint64_t));
  memset(FN, 0, n_hidden * sizeof(uint64_t));

  #pragma omp parallel for
  for (uint64_t i = 0; i < n_codes; i ++) {
    for (uint64_t j = 0; j < n_hidden; j ++) {
      uint64_t word = j / (sizeof(tk_bits_t) * CHAR_BIT);
      uint64_t bit = j % (sizeof(tk_bits_t) * CHAR_BIT);
      bool y_true = (codes_expected[i * chunks + word] >> bit) & 1;
      bool y_pred = (codes_predicted[i * chunks + word] >> bit) & 1;
      if (y_pred && y_true) {
        #pragma omp atomic
        TP[j] ++;
      } else if (y_pred && !y_true) {
        #pragma omp atomic
        FP[j] ++;
      } else if (!y_pred && y_true) {
        #pragma omp atomic
        FN[j] ++;
      }
    }
  }

  double *precision = tk_malloc(L, n_hidden * sizeof(double));
  double *recall = tk_malloc(L, n_hidden * sizeof(double));
  double *f1 = tk_malloc(L, n_hidden * sizeof(double));
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

  free(TP);
  free(FP);
  free(FN);

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
  for (uint64_t j = 0; j < n_hidden; j ++) {
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

  free(precision);
  free(recall);
  free(f1);

  return 1;
}

static inline double _tm_auc (
  lua_State *L,
  tk_bits_t *codes,
  tk_bits_t *mask,
  tm_pair_t *pos,
  tm_pair_t *neg,
  uint64_t n_pos,
  uint64_t n_neg,
  uint64_t n_sentences,
  uint64_t n_hidden,
  tm_dl_t **plp
) {
  tm_dl_t *pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  uint64_t chunks = BITS_DIV(n_hidden);

  // Calculate AUC
  #pragma omp parallel for
  for (uint64_t k = 0; k < n_pos + n_neg; k ++) {
    tm_pair_t *pairs = k < n_pos ? pos : neg;
    uint64_t offset = k < n_pos ? 0 : n_pos;
    int64_t u = pairs[k - offset].u;
    int64_t v = pairs[k - offset].v;
    if (mask != NULL)
      pl[k].sim = (uint64_t) n_hidden - hamming_mask(codes + (uint64_t) u * chunks, codes + (uint64_t) v * chunks, mask, chunks);
    else
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
  if (plp != NULL)
    *plp = pl;
  else
    free(pl);
  return auc;
}

static inline int tm_auc (lua_State *L)
{
  lua_settop(L, 6);

  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "codes");

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

  tk_bits_t *mask =
    lua_type(L, 6) == LUA_TSTRING ? (tk_bits_t *) luaL_checkstring(L, 6) :
    lua_type(L, 6) == LUA_TLIGHTUSERDATA ? (tk_bits_t *) lua_touserdata(L, 6) : NULL;

  double auc = _tm_auc(L, codes, mask, pos, neg, n_pos, n_neg, n_sentences, n_hidden, NULL);

  lua_pushnumber(L, auc);
  return 1;
}

static inline int tm_encoding_similarity (lua_State *L)
{
  lua_settop(L, 5);

  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "codes");

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

  tm_dl_t *pl;
  double auc = _tm_auc(L, codes, NULL, pos, neg, n_pos, n_neg, n_sentences, n_hidden, &pl);

  // Calculate total for best margin calculation
  uint64_t hist_pos[n_hidden + 1];
  uint64_t hist_neg[n_hidden + 1];
  memset(hist_pos, 0, (n_hidden + 1) * sizeof(uint64_t));
  memset(hist_neg, 0, (n_hidden + 1) * sizeof(uint64_t));
  #pragma omp parallel for
  for (uint64_t k = 0; k < n_pos + n_neg; k ++) {
    uint64_t s = pl[k].sim;
    if (s > n_hidden) s = n_hidden;
    if (pl[k].label) {
      #pragma omp atomic
      hist_pos[s] ++;
    } else {
      #pragma omp atomic
      hist_neg[s] ++;
    }
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
  { "auc", tm_auc },
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
