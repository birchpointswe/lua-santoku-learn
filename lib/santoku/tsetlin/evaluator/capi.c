#define _GNU_SOURCE

#include <santoku/tsetlin/conf.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>

typedef enum {
  TK_EVAL_CLASS_ACCURACY,
  TK_EVAL_ENCODING_ACCURACY,
  TK_EVAL_ENCODING_SIMILARITY,
  TK_EVAL_ENCODING_AUC,
  TK_EVAL_CODEBOOK_STATS
} tk_eval_stage_t;

typedef struct {
  tm_dl_t *pl;
  atomic_ulong *TP, *FP, *FN, *bit_counts, *hist_pos, *hist_neg;
  tk_ivec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  double *f1, *precision, *recall;
  unsigned int *predicted, *expected;
  tk_bits_t *codes, *codes_predicted, *codes_expected, *mask;
  unsigned int n_classes, chunks;
} tk_eval_t;

typedef struct {
  tk_eval_t *state;
  uint64_t sfirst, slast;
  uint64_t pfirst, plast;
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
        if (y_pred == y_true)
          atomic_fetch_add(state->TP + y_true, 1);
        else {
          atomic_fetch_add(state->FP + y_pred, 1);
          atomic_fetch_add(state->FN + y_true, 1);
        }
      }
      break;

    case TK_EVAL_ENCODING_ACCURACY:
      for (uint64_t i = data->sfirst; i <= data->slast; i ++) {
        for (uint64_t j = 0; j < state->n_classes; j ++) {
          uint64_t word = j / (sizeof(tk_bits_t) * CHAR_BIT);
          uint64_t bit = j % (sizeof(tk_bits_t) * CHAR_BIT);
          bool y_true = (state->codes_expected[i * state->chunks + word] >> bit) & 1;
          bool y_pred = (state->codes_predicted[i * state->chunks + word] >> bit) & 1;
          if (y_pred && y_true)
            atomic_fetch_add(state->TP + j, 1);
          else if (y_pred && !y_true)
            atomic_fetch_add(state->FP + j, 1);
          else if (!y_pred && y_true)
            atomic_fetch_add(state->FN + j, 1);
        }
      }
      break;

    case TK_EVAL_ENCODING_SIMILARITY:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        uint64_t s = state->pl[k].sim;
        if (s > state->n_classes)
          s = state->n_classes;
        if (state->pl[k].label)
          atomic_fetch_add(state->hist_pos + s, 1);
        else
          atomic_fetch_add(state->hist_neg + s, 1);
      }
      break;

    case TK_EVAL_ENCODING_AUC:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        tk_ivec_t *pairs = k < state->n_pos ? state->pos : state->neg;
        uint64_t offset = k < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(k - offset) * 2 + 0];
        int64_t v = pairs->a[(k - offset) * 2 + 1];
        if (state->mask != NULL)
          state->pl[k].sim = (uint64_t) state->n_classes - hamming_mask(
            state->codes + (uint64_t) u * state->chunks,
            state->codes + (uint64_t) v * state->chunks, state->mask, state->chunks);
        else
          state->pl[k].sim = (uint64_t) state->n_classes - hamming(
            state->codes + (uint64_t) u * state->chunks,
            state->codes + (uint64_t) v * state->chunks, state->chunks);
        state->pl[k].label = k < state->n_pos ? 1 : 0;
      }
      break;

    case TK_EVAL_CODEBOOK_STATS:
      for (uint64_t i = data->sfirst; i <= data->slast; i ++) {
        for (uint64_t j = 0; j < state->n_classes; j ++) {
          uint64_t word = j / BITS;
          uint64_t bit = j % BITS;
          if ((state->codes[i * state->chunks + word] >> bit) & 1)
            atomic_fetch_add(state->bit_counts + j, 1);
        }
      }
      break;

  }
}

static inline int tm_class_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  unsigned int *predicted = (unsigned int *) tk_lua_checkustring(L, 1, "predicted");
  unsigned int *expected = (unsigned int *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_classes = tk_lua_checkunsigned(L, 4, "n_classes");
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

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_CLASS_ACCURACY);
  tk_threads_destroy(pool);

  // Reduce
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (unsigned int c = 0; c < n_classes; c ++) {
    uint64_t tp = state.TP[c], fp = state.FP[c], fn = state.FN[c];
    state.precision[c] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
    state.recall[c] = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
    state.f1[c] = (state.precision[c] + state.recall[c]) > 0 ?
      2.0 * state.precision[c] * state.recall[c] / (state.precision[c] + state.recall[c]) : 0.0;
    precision_avg += state.precision[c];
    recall_avg += state.recall[c];
    f1_avg += state.f1[c];
  }

  // Cleanup
  free(state.TP);
  free(state.FP);
  free(state.FN);

  // Lua output
  precision_avg /= n_classes;
  recall_avg /= n_classes;
  f1_avg /= n_classes;
  lua_newtable(L);
  lua_newtable(L);
  for (uint64_t c = 0; c < n_classes; c ++) {
    lua_pushinteger(L, (int64_t) c + 1);
    lua_newtable(L);
    lua_pushnumber(L, state.precision[c]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, state.recall[c]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, state.f1[c]);
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

  // Cleanup
  free(state.precision);
  free(state.recall);
  free(state.f1);

  return 1;
}

static inline int tm_codebook_stats (lua_State *L)
{
  lua_settop(L, 4);
  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 2, "n_samples");
  unsigned int n_classes = tk_lua_checkunsigned(L, 3, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 4, "n_threads", NULL);
  uint64_t chunks = BITS_DIV(n_classes);

  tk_eval_t state;
  state.n_classes = n_classes;
  state.codes = codes;
  state.chunks = chunks;
  state.bit_counts = tk_malloc(L, n_classes * sizeof(atomic_ulong));
  for (uint64_t i = 0; i < n_classes; i ++)
    atomic_init(state.bit_counts + i, 0);

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Run counts via pool
  tk_threads_signal(pool, TK_EVAL_CODEBOOK_STATS);
  tk_threads_destroy(pool);

  // Compute per-bit entropy
  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_classes; j ++) {
    double p = (double) state.bit_counts[j] / (double) n_samples;
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
  double mean = sum_entropy / n_classes;
  double variance = 0.0;
  for (uint64_t j = 0; j < n_classes; j ++) {
    double p = (double) state.bit_counts[j] / (double) n_samples;
    double entropy = 0.0;
    if (p > 0.0 && p < 1.0)
      entropy = -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    variance += (entropy - mean) * (entropy - mean);
  }

  free(state.bit_counts);

  variance /= n_classes;
  lua_pushnumber(L, mean);
  lua_setfield(L, -2, "mean");
  lua_pushnumber(L, min_entropy);
  lua_setfield(L, -2, "min");
  lua_pushnumber(L, max_entropy);
  lua_setfield(L, -2, "max");
  lua_pushnumber(L, sqrt(variance));
  lua_setfield(L, -2, "std");

  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  tk_bits_t *codes_predicted = (tk_bits_t *) tk_lua_checkustring(L, 1, "predicted");
  tk_bits_t *codes_expected = (tk_bits_t *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_classes = tk_lua_checkunsigned(L, 4, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);
  uint64_t chunks = BITS_DIV(n_classes);

  tk_eval_t state;
  state.n_classes = n_classes;
  state.chunks = chunks;
  state.codes_expected = codes_expected;
  state.codes_predicted = codes_predicted;
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

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_ENCODING_ACCURACY);
  tk_threads_destroy(pool);

  // Reduce
  double precision_avg = 0.0, recall_avg = 0.0, f1_avg = 0.0;
  for (uint64_t j = 0; j < n_classes; j ++) {
    uint64_t tp = state.TP[j], fp = state.FP[j], fn = state.FN[j];
    state.precision[j] = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
    state.recall[j] = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
    state.f1[j] = (state.precision[j] + state.recall[j]) > 0 ?
      2.0 * state.precision[j] * state.recall[j] / (state.precision[j] + state.recall[j]) : 0.0;
    precision_avg += state.precision[j];
    recall_avg += state.recall[j];
    f1_avg += state.f1[j];
  }

  // Cleanup
  free(state.TP);
  free(state.FP);
  free(state.FN);

  // Lua output
  precision_avg /= n_classes;
  recall_avg /= n_classes;
  f1_avg /= n_classes;
  lua_newtable(L);
  lua_newtable(L); // classes
  for (uint64_t j = 0; j < n_classes; j ++) {
    lua_pushinteger(L, (int64_t) j + 1);
    lua_newtable(L);
    lua_pushnumber(L, state.precision[j]);
    lua_setfield(L, -2, "precision");
    lua_pushnumber(L, state.recall[j]);
    lua_setfield(L, -2, "recall");
    lua_pushnumber(L, state.f1[j]);
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

  // F1 stats
  double f1_var = 0.0;
  double min_f1 = 1.0, max_f1 = 0.0;
  for (uint64_t j = 0; j < n_classes; j ++) {
    double f = state.f1[j];
    f1_var += (f - f1_avg) * (f - f1_avg);
    if (f < min_f1) min_f1 = f;
    if (f > max_f1) max_f1 = f;
  }
  f1_var /= n_classes;
  lua_pushnumber(L, min_f1);
  lua_setfield(L, -2, "f1_min");
  lua_pushnumber(L, max_f1);
  lua_setfield(L, -2, "f1_max");
  lua_pushnumber(L, sqrt(f1_var));
  lua_setfield(L, -2, "f1_std");

  // Cleanup
  free(state.precision);
  free(state.recall);
  free(state.f1);

  return 1;
}

static inline double _tm_auc (
  lua_State *L,
  tk_eval_t *state,
  tk_threadpool_t *pool
) {
  tk_threads_signal(pool, TK_EVAL_ENCODING_AUC);
  ks_introsort(dl, state->n_pos + state->n_neg, state->pl);
  double sum_ranks = 0.0;
  unsigned int rank = 1;
  // TODO: Parallelize?
  for (uint64_t k = 0; k < state->n_pos + state->n_neg; k ++, rank ++)
    if (state->pl[k].label)
      sum_ranks += rank;
  double auc = (sum_ranks - ((double) state->n_pos * (state->n_pos + 1) / 2)) / ((double) state->n_pos * state->n_neg);
  return auc;
}

static inline int tm_auc (lua_State *L)
{
  lua_settop(L, 6);

  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "codes");
  tk_ivec_t *pos = tk_ivec_peek(L, 2);
  tk_ivec_t *neg = tk_ivec_peek(L, 3);
  uint64_t n_classes = tk_lua_checkunsigned(L, 4, "n_hidden");
  uint64_t n_pos = pos->n / 2;
  uint64_t n_neg = neg->n / 2;

  tk_bits_t *mask =
    lua_type(L, 5) == LUA_TSTRING ? (tk_bits_t *) luaL_checkstring(L, 5) :
    lua_type(L, 5) == LUA_TLIGHTUSERDATA ? (tk_bits_t *) lua_touserdata(L, 5) : NULL;

  unsigned int n_threads = tk_threads_getn(L, 6, "n_threads", NULL);

  tk_eval_t state;
  state.n_classes = n_classes;
  state.chunks = BITS_DIV(n_classes);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = mask;
  state.codes = codes;
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  double auc = _tm_auc(L, &state, pool);

  free(state.pl);
  tk_threads_destroy(pool);

  lua_pushnumber(L, auc);
  return 1;
}

static inline int tm_encoding_similarity (lua_State *L)
{
  lua_settop(L, 5);

  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "codes");
  tk_ivec_t *pos = tk_ivec_peek(L, 2);
  tk_ivec_t *neg = tk_ivec_peek(L, 3);
  uint64_t n_classes = tk_lua_checkunsigned(L, 4, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);
  uint64_t n_pos = pos->n / 2;
  uint64_t n_neg = neg->n / 2;

  tk_eval_t state;
  state.n_classes = n_classes;
  state.chunks = BITS_DIV(n_classes);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = NULL;
  state.codes = codes;
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.hist_pos = tk_malloc(L, (n_classes + 1) * sizeof(atomic_ulong));
  state.hist_neg = tk_malloc(L, (n_classes + 1) * sizeof(atomic_ulong));
  for (uint64_t i = 0; i < n_classes + 1; i ++) {
    atomic_init(state.hist_pos + i, 0);
    atomic_init(state.hist_neg + i, 0);
  }

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  double auc = _tm_auc(L, &state, pool);

  // Calculate total for best margin calculation via pool
  tk_threads_signal(pool, TK_EVAL_ENCODING_SIMILARITY);
  tk_threads_destroy(pool);

  // Find best margin for f1
  double best_f1 = -1.0, best_prec = 0.0, best_rec = 0.0;
  uint64_t best_margin = 0, cum_tp = 0, cum_fp = 0;
  for (int64_t m = (int64_t) n_classes; m >= 0; m --) {
    cum_tp += state.hist_pos[m];
    cum_fp += state.hist_neg[m];
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
  free(state.pl);
  free(state.hist_pos);
  free(state.hist_neg);

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
