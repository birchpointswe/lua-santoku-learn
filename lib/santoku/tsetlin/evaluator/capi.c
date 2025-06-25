#define _GNU_SOURCE

#include <santoku/tsetlin/conf.h>
#include <santoku/tsetlin/graph.h>
#include <santoku/tsetlin/cluster.h>
#include <santoku/threads.h>
#include <santoku/ivec.h>

typedef enum {
  TK_EVAL_CLASS_ACCURACY,
  TK_EVAL_CLUSTERING_ACCURACY,
  TK_EVAL_ENCODING_ACCURACY,
  TK_EVAL_ENCODING_SIMILARITY,
  TK_EVAL_ENCODING_AUC,
} tk_eval_stage_t;

typedef struct {
  tm_dl_t *pl;
  atomic_ulong *TP, *FP, *FN, *hist_pos, *hist_neg;
  tk_pvec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  double *f1, *precision, *recall;
  unsigned int *predicted, *expected;
  tk_bits_t *codes, *codes_predicted, *codes_expected, *mask;
  tk_iumap_t *id_code, *id_assignment;
  unsigned int n_classes, chunks;
  tk_ivec_t *assignments;
  tk_graph_t *graph;
} tk_eval_t;

typedef struct {
  tk_eval_t *state;
  uint64_t sfirst, slast;
  uint64_t pfirst, plast;
} tk_eval_thread_t;

typedef struct {
  double f1;
  double precision;
  double recall;
} tk_accuracy_t;

static void tk_eval_worker (void *dp, int sig)
{
  tk_eval_thread_t *data = (tk_eval_thread_t *) dp;
  tk_eval_t *state = data->state;
  khint_t khi;
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

    case TK_EVAL_CLUSTERING_ACCURACY:
      for (unsigned int i = data->pfirst; i <= data->plast; i ++) {
        tk_pvec_t *pairs = i < state->n_pos ? state->pos : state->neg;
        uint64_t offset = i < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(i - offset)].i;
        int64_t v = pairs->a[(i - offset)].p;
        khi = tk_iumap_get(state->id_assignment, u);
        if (khi == tk_iumap_end(state->id_assignment))
          continue;
        int64_t iu = tk_iumap_value(state->id_assignment, khi);
        khi = tk_iumap_get(state->id_assignment, v);
        if (khi == tk_iumap_end(state->id_assignment))
          continue;
        int64_t iv = tk_iumap_value(state->id_assignment, khi);
        int64_t cu = state->assignments->a[iu];
        int64_t cv = state->assignments->a[iv];
        if (i < state->n_pos) {
          if (cu == cv)
            atomic_fetch_add(state->TP, 1);
          else
            atomic_fetch_add(state->FN, 1);
        } else {
          if (cu == cv)
            atomic_fetch_add(state->FP, 1);
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
        if (state->pl[k].sim == -1)
          continue;
        int64_t s = state->pl[k].sim;
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
        tk_pvec_t *pairs = k < state->n_pos ? state->pos : state->neg;
        uint64_t offset = k < state->n_pos ? 0 : state->n_pos;
        int64_t u = pairs->a[(k - offset) * 2].i;
        int64_t v = pairs->a[(k - offset) * 2].p;
        int64_t iu, iv;
        if (state->id_code == NULL) {
          iu = u;
          iv = v;
        } else {
          khi = tk_iumap_get(state->id_code, u);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iu = tk_iumap_value(state->id_code, khi);
          khi = tk_iumap_get(state->id_code, v);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iv = tk_iumap_value(state->id_code, khi);
        }
        if (state->mask != NULL)
          state->pl[k].sim = (int64_t) (state->n_classes - hamming_mask(
            state->codes + (uint64_t) iu * state->chunks,
            state->codes + (uint64_t) iv * state->chunks, state->mask, state->chunks));
        else
          state->pl[k].sim = (int64_t) (state->n_classes - hamming(
            state->codes + (uint64_t) iu * state->chunks,
            state->codes + (uint64_t) iv * state->chunks, state->chunks));
        state->pl[k].label = k < state->n_pos ? 1 : 0;
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

static inline int tm_entropy_stats (lua_State *L)
{
  lua_settop(L, 4);
  tk_bits_t *codes;
  tk_cvec_t *cvec = tk_cvec_peekopt(L, 1);
  codes = cvec != NULL ? (tk_bits_t *) cvec->a : (tk_bits_t *) tk_lua_checkustring(L, 1, "codes");
  unsigned int n_samples = tk_lua_checkunsigned(L, 2, "n_samples");
  unsigned int n_classes = tk_lua_checkunsigned(L, 3, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 4, "n_threads", NULL);

  tk_dvec_t *entropies = tk_ivec_score_entropy(L, (char *) codes, n_samples, n_classes, n_threads);

  // Compute per-bit entropy
  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_classes; j ++) {
    double entropy = entropies->a[j];
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
    double entropy = entropies->a[j];
    variance += (entropy - mean) * (entropy - mean);
  }

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

static inline tk_accuracy_t tm_clustering_accuracy (
  lua_State *L,
  tk_ivec_t *ids,
  tk_ivec_t *assignments,
  tk_pvec_t *pos,
  tk_pvec_t *neg,
  unsigned int n_threads
) {
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  atomic_ulong TP, FP, FN;
  double precision, recall, f1;
  state.assignments = assignments;
  state.id_assignment = tk_iumap_from_ivec(ids);
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.TP = &TP;
  state.FP = &FP;
  state.FN = &FN;
  atomic_init(&TP, 0);
  atomic_init(&FP, 0);
  atomic_init(&FN, 0);

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    tk_thread_range(i, n_threads, n_pos + n_neg, &data[i].pfirst, &data[i].plast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_CLUSTERING_ACCURACY);
  tk_threads_destroy(pool);

  // Reduce
  uint64_t tp = atomic_load(&TP), fp = atomic_load(&FP), fn = atomic_load(&FN);
  precision = (tp + fp) > 0 ? (double) tp / (tp + fp) : 0.0;
  recall = (tp + fn) > 0 ? (double) tp / (tp + fn) : 0.0;
  f1 = (precision + recall) > 0 ?
    2.0 * precision * recall / (precision + recall) : 0.0;

  tk_accuracy_t acc;
  acc.f1 = f1;
  acc.precision = precision;
  acc.recall = recall;

  tk_iumap_destroy(state.id_assignment);
  return acc;
}

static inline int tm_clustering_accuracy_lua (lua_State *L)
{
  lua_settop(L, 5);
  tk_ivec_t *assignments = tk_ivec_peek(L, 1, "assignments");
  tk_ivec_t *ids = tk_ivec_peek(L, 2, "ids");
  tk_pvec_t *pos = tk_pvec_peek(L, 3, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 4, "neg");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  tk_accuracy_t acc = tm_clustering_accuracy(L, ids, assignments, pos, neg, n_threads);

  // Lua output
  lua_newtable(L);
  lua_pushnumber(L, acc.precision);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, acc.recall);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, acc.f1);
  lua_setfield(L, -2, "f1");

  return 1;
}

static inline int tm_encoding_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  tk_bits_t *codes_predicted, *codes_expected;
  tk_cvec_t *pvec = tk_cvec_peekopt(L, 1);
  tk_cvec_t *evec = tk_cvec_peekopt(L, 2);
  codes_predicted = pvec != NULL ? (tk_bits_t *) pvec->a : (tk_bits_t *) tk_lua_checkustring(L, 1, "predicted");
  codes_expected = evec != NULL ? (tk_bits_t *) evec->a : (tk_bits_t *) tk_lua_checkustring(L, 2, "expected");
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
  uint64_t n_pos_valid = 0, n_neg_valid = 0;
  for (uint64_t k = 0; k < state->n_pos + state->n_neg; k ++) {
    if (state->pl[k].sim == -1)
      continue;
    if (state->pl[k].label)
      sum_ranks += rank, n_pos_valid ++;
    else
      n_neg_valid ++;
    rank ++;
  }
  if (n_pos_valid == 0 || n_neg_valid == 0)
    return 0.0;
  double auc = (sum_ranks - ((double) n_pos_valid * (n_pos_valid + 1) / 2)) / ((double) n_pos_valid * n_neg_valid);
  return auc;
}

static inline int tm_auc (lua_State *L)
{
  lua_settop(L, 6);

  tk_bits_t *codes = (tk_bits_t *) tk_lua_checkustring(L, 1, "codes");
  tk_pvec_t *pos = tk_pvec_peek(L, 2, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 3, "neg");
  uint64_t n_classes = tk_lua_checkunsigned(L, 4, "n_hidden");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

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

static inline int tm_optimize_clustering (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "index");
  // tk_inv_t *inv = tk_inv_peekopt(L, i_index);
  // tk_ann_t *ann = tk_ann_peekopt(L, i_index);
  // tk_hbi_t *hbi = tk_hbi_peekopt(L, -1);

  // if (inv == NULL && ann == NULL && hbi == NULL)
  //   tk_lua_verror(L, 3, "graph", "index", "either tk_inv_t, tk_ann_t, or tk_inv_t must be provided");

  tk_hbi_t *hbi = tk_hbi_peek(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peek(L, -1, "pos");

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peek(L, -1, "neg");

  uint64_t min_margin = tk_lua_fcheckunsigned(L, 1, "optimize clustering", "min_margin");
  uint64_t max_margin = tk_lua_fcheckunsigned(L, 1, "optimize clustering", "max_margin");

  if (max_margin < min_margin)
    max_margin = min_margin;

  unsigned int n_threads = tk_threads_getn(L, 1, "optimize clustering", "threads");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  uint64_t n_clusters = 0;
  tk_ivec_t *ids = NULL;
  tk_ivec_t *assignments = NULL;
  tk_accuracy_t best = {0};
  uint64_t best_n = 0, best_m = 0;
  int i_ids = LUA_NOREF, i_assign = LUA_NOREF;
  for (uint64_t m = min_margin; m <= max_margin; m ++) {
    tk_cluster_dsu(L, hbi, m, &ids, &assignments, &n_clusters); // ids assignments
    tk_accuracy_t result = tm_clustering_accuracy(L, ids, assignments, pos, neg, n_threads);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, result.f1);
      lua_pushnumber(L, result.precision);
      lua_pushnumber(L, result.recall);
      lua_pushinteger(L, (int64_t) m);
      lua_pushinteger(L, (int64_t) n_clusters);
      lua_call(L, 5, 0);
    }
    if (result.f1 > best.f1) {
      best = result;
      best_n = n_clusters;
      best_m = m;
      luaL_unref(L, LUA_REGISTRYINDEX, i_assign);
      luaL_unref(L, LUA_REGISTRYINDEX, i_ids);
      i_assign = luaL_ref(L, LUA_REGISTRYINDEX); // assignments
      i_ids = luaL_ref(L, LUA_REGISTRYINDEX); // ids
      // empty
    } else {
      lua_pop(L, 2); // empty
    }
  }

  lua_newtable(L); // score
  lua_pushinteger(L, (int64_t) best_m);
  lua_setfield(L, -2, "margin");
  lua_pushinteger(L, (int64_t) best_n);
  lua_setfield(L, -2, "n_clusters");
  lua_pushnumber(L, best.precision);
  lua_setfield(L, -2, "precision");
  lua_pushnumber(L, best.recall);
  lua_setfield(L, -2, "recall");
  lua_pushnumber(L, best.f1);
  lua_setfield(L, -2, "f1");

  lua_rawgeti(L, LUA_REGISTRYINDEX, i_ids); // ids
  lua_rawgeti(L, LUA_REGISTRYINDEX, i_assign); // assign
  luaL_unref(L, LUA_REGISTRYINDEX, i_ids);
  luaL_unref(L, LUA_REGISTRYINDEX, i_assign);
  lua_pushinteger(L, (int64_t) best_n); // n

  return 4;
}

static inline int tm_optimize_retrieval (lua_State *L)
{
  lua_settop(L, 1);

  tk_bits_t *codes;
  lua_getfield(L, 1, "codes");
  tk_cvec_t *cvec = tk_cvec_peekopt(L, -1);
  codes = cvec != NULL
    ? (tk_bits_t *) cvec->a
    : (tk_bits_t *) tk_lua_checkustring(L, -1, "codes");

  lua_getfield(L, 1, "ids");
  tk_ivec_t *ids = tk_ivec_peekopt(L, -1);

  lua_getfield(L, 1, "pos");
  tk_pvec_t *pos = tk_pvec_peek(L, -1, "pos");

  lua_getfield(L, 1, "neg");
  tk_pvec_t *neg = tk_pvec_peek(L, -1, "neg");

  int i_each = -1;
  if (tk_lua_ftype(L, 1, "each") != LUA_TNIL) {
    lua_getfield(L, 1, "each");
    i_each = tk_lua_absindex(L, -1);
  }

  uint64_t n_classes = tk_lua_fcheckunsigned(L, 1, "optimize_retrieval", "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_retrieval", "threads");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  state.n_classes = n_classes;
  state.chunks = BITS_DIV(n_classes);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = NULL;
  state.codes = codes;
  state.id_code = ids == NULL ? NULL : tk_iumap_from_ivec(ids); // TODO: Pass id_code (aka uid_hood) in instead of recomputing
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

  // Calculate histogram
  uint64_t tail_tp[n_classes + 1], tail_fp[n_classes + 1];
  uint64_t running_tp = 0, running_fp = 0;
  for (int64_t s = (int64_t) n_classes; s >= 0; s--) {
    running_tp += atomic_load(state.hist_pos + s);
    running_fp += atomic_load(state.hist_neg + s);
    tail_tp[s] = running_tp;
    tail_fp[s] = running_fp;
  }

  // Find best margin for f1
  double best_f1 = -1.0, best_prec = 0.0, best_rec = 0.0;
  uint64_t best_margin = 0;
  for (uint64_t m = 0; m <= n_classes; m ++) {
    double prec = (tail_tp[m] + tail_fp[m]) > 0 ? (double) tail_tp[m] / (tail_tp[m] + tail_fp[m]) : 0.0;
    double rec = (double) tail_tp[m] / (double) n_pos;
    double f1 = (prec + rec) > 0 ? 2 * prec * rec / (prec + rec) : 0.0;
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, auc);
      lua_pushnumber(L, f1);
      lua_pushnumber(L, prec);
      lua_pushnumber(L, rec);
      lua_pushinteger(L, (int64_t) m);
      lua_call(L, 5, 1);
      if (lua_type(L, -1) == LUA_TBOOLEAN && lua_toboolean(L, -1) == 0) {
        lua_pop(L, 1);
        break;
      } else {
        lua_pop(L, 1);
      }
    }
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
  if (state.id_code != NULL)
    tk_iumap_destroy(state.id_code);

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
  { "clustering_accuracy", tm_clustering_accuracy_lua },
  { "optimize_retrieval", tm_optimize_retrieval },
  { "optimize_clustering", tm_optimize_clustering },
  { "entropy_stats", tm_entropy_stats },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}
