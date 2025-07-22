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
  atomic_ulong *ERR, *TP, *FP, *FN, *hist_pos, *hist_neg;
  tk_pvec_t *pos, *neg;
  uint64_t n_pos, n_neg;
  double *f1, *precision, *recall;
  unsigned int *predicted, *expected;
  tk_bits_t *codes, *codes_predicted, *codes_expected, *mask;
  double *dcodes;
  tk_iumap_t *id_code, *id_assignment;
  unsigned int n_visible, n_dims, chunks;
  tk_ivec_t *assignments;
  tk_graph_t *graph;
} tk_eval_t;

typedef struct {
  tk_eval_t *state;
  uint64_t diff;
  uint64_t *bdiff;
  uint64_t sfirst, slast;
  uint64_t pfirst, plast;
  uint64_t dfirst, dlast;
} tk_eval_thread_t;

typedef struct {
  double tpr;
  double tnr;
  double bacc;
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
        if (y_pred >= state->n_dims || y_true >= state->n_dims)
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
        for (uint64_t j = 0; j < state->n_dims; j ++) {
          uint64_t word = BITS_BYTE(j);
          uint64_t bit = BITS_BIT(j);
          bool y =
            (state->codes_expected[i * state->chunks + word] & ((tk_bits_t)1 << bit)) ==
            (state->codes_predicted[i * state->chunks + word] & ((tk_bits_t)1 << bit));
          if (y)
            continue;
          data->diff ++;
          data->bdiff[j] ++;
        }
      }
      break;

    case TK_EVAL_ENCODING_SIMILARITY:
      for (uint64_t k = data->pfirst; k <= data->plast; k ++) {
        if (state->pl[k].sim == -1)
          continue;
        int64_t s = state->n_dims - state->pl[k].sim;
        if (s < 0) s = 0;
        if (s > state->n_dims) s = state->n_dims;
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
        int64_t u = pairs->a[(k - offset)].i;
        int64_t v = pairs->a[(k - offset)].p;
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
          if (iu == -1)
            continue;
          khi = tk_iumap_get(state->id_code, v);
          if (khi == tk_iumap_end(state->id_code)) {
            state->pl[k].sim = -1;
            continue;
          }
          iv = tk_iumap_value(state->id_code, khi);
          if (iv == -1)
            continue;
        }
        if (state->dcodes != NULL) {
          // continuous embedding: dot-product
          double *X = state->dcodes;
          uint64_t K = state->n_dims;
          double dot = 0.0;
          double *xi = X + (uint64_t)iu * K;
          double *xj = X + (uint64_t)iv * K;
          for (uint64_t d = 0; d < K; d ++)
            dot += xi[d] * xj[d];
          // scale to retain precision in int64
          state->pl[k].sim = (int64_t)(dot * 1e3);
        } else if (state->codes != NULL) {
          if (state->mask != NULL) {
            state->pl[k].sim = (int64_t) (state->n_dims - tk_ann_hamming_mask(
              (const unsigned char *) state->codes + (uint64_t) iu * state->chunks,
              (const unsigned char *) state->codes + (uint64_t) iv * state->chunks, (const unsigned char *) state->mask, state->n_dims));
          } else {
            state->pl[k].sim = (int64_t) (state->n_dims - tk_ann_hamming(
              (const unsigned char *) state->codes + (uint64_t) iu * state->chunks,
              (const unsigned char *) state->codes + (uint64_t) iv * state->chunks, state->n_dims));
          }
        } else {
          state->pl[k].sim = -1;
        }
        state->pl[k].label = k < state->n_pos ? 1 : 0;
      }
      break;

    default:
      assert(false);

  }
}

static inline int tm_class_accuracy (lua_State *L)
{
  lua_settop(L, 5);
  unsigned int *predicted = (unsigned int *) tk_lua_checkustring(L, 1, "predicted");
  unsigned int *expected = (unsigned int *) tk_lua_checkustring(L, 2, "expected");
  unsigned int n_samples = tk_lua_checkunsigned(L, 3, "n_samples");
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_classes");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);

  if (n_dims == 0)
    tk_lua_verror(L, 3, "class_accuracy", "n_classes", "must be > 0");

  tk_eval_t state;
  state.n_dims = n_dims;
  state.expected = expected;
  state.predicted = predicted;
  state.TP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.FP = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.FN = tk_malloc(L, n_dims * sizeof(atomic_ulong));
  state.precision = tk_malloc(L, n_dims * sizeof(double));
  state.recall = tk_malloc(L, n_dims * sizeof(double));
  state.f1 = tk_malloc(L, n_dims * sizeof(double));
  for (uint64_t i = 0; i < n_dims; i ++) {
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
  for (unsigned int c = 0; c < n_dims; c ++) {
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
  precision_avg /= n_dims;
  recall_avg /= n_dims;
  f1_avg /= n_dims;
  lua_newtable(L);
  lua_newtable(L);
  for (uint64_t c = 0; c < n_dims; c ++) {
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
  unsigned int n_dims = tk_lua_checkunsigned(L, 3, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 4, "n_threads", NULL);

  tk_dvec_t *entropies = tk_ivec_score_entropy(L, (char *) codes, n_samples, n_dims, n_threads);

  // Compute per-bit entropy
  double min_entropy = 1.0, max_entropy = 0.0, sum_entropy = 0.0;
  lua_newtable(L); // result
  lua_newtable(L); // per-bit entropy table
  for (uint64_t j = 0; j < n_dims; j ++) {
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
  double mean = sum_entropy / n_dims;
  double variance = 0.0;
  for (uint64_t j = 0; j < n_dims; j ++) {
    double entropy = entropies->a[j];
    variance += (entropy - mean) * (entropy - mean);
  }

  variance /= n_dims;
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
  uint64_t tp = atomic_load(state.TP);
  uint64_t fp = atomic_load(state.FP);
  uint64_t tn = n_neg > fp ? (n_neg - fp) : 0;
  double tpr = n_pos > 0 ? (double) tp / n_pos : 0.0;
  double tnr = n_neg > 0 ? (double) tn / n_neg : 0.0;
  double bacc = 0.5 * (tpr + tnr);
  tk_accuracy_t acc;
  acc.tpr = tpr;
  acc.tnr = tnr;
  acc.bacc = bacc;

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
  lua_pushnumber(L, acc.bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, acc.tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, acc.tnr);
  lua_setfield(L, -2, "tnr");
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
  unsigned int n_dims = tk_lua_checkunsigned(L, 4, "n_hidden");
  unsigned int n_threads = tk_threads_getn(L, 5, "n_threads", NULL);
  uint64_t chunks = BITS_BYTES(n_dims);

  tk_eval_t state;
  state.n_dims = n_dims;
  state.chunks = chunks;
  state.codes_expected = codes_expected;
  state.codes_predicted = codes_predicted;

  // Setup pool
  tk_eval_thread_t data[n_threads];
  tk_threadpool_t *pool = tk_threads_create(L, n_threads, tk_eval_worker);
  for (unsigned int i = 0; i < n_threads; i ++) {
    pool->threads[i].data = data + i;
    data[i].state = &state;
    data[i].diff = 0;
    data[i].bdiff = tk_malloc(L, n_dims * sizeof(uint64_t));
    memset(data[i].bdiff, 0, sizeof(uint64_t) * n_dims);
    tk_thread_range(i, n_threads, n_samples, &data[i].sfirst, &data[i].slast);
  }

  // Run eval via pool
  tk_threads_signal(pool, TK_EVAL_ENCODING_ACCURACY);
  tk_threads_destroy(pool);

  // Reduce
  uint64_t diff_total = 0;
  uint64_t *bdiff_total = tk_malloc(L, n_dims * sizeof(uint64_t));
  memset(bdiff_total, 0, n_dims * sizeof(uint64_t));
  for (unsigned int i = 0; i < n_threads; i ++) {
    diff_total += data[i].diff;
    for (uint64_t j = 0; j < n_dims; j ++)
      bdiff_total[j] += data[i].bdiff[j];
  }

  // Lua output
  lua_newtable(L);
  lua_newtable(L); // dims
  double min_bdiff = 1.0, max_bdiff = 0.0;
  for (uint64_t j = 0; j < n_dims; j ++) {
    double t = (double) bdiff_total[j] / (double) n_samples;
    if (t < min_bdiff) min_bdiff = t;
    if (t > max_bdiff) max_bdiff = t;
    lua_pushinteger(L, (int64_t) j + 1);
    lua_pushnumber(L, t);
    lua_settable(L, -3);
  }

  double mean_bdiff = (double) diff_total / (n_samples * n_dims);  /* mean Hamming */
  double var = 0.0;
  for (uint64_t j = 0; j < n_dims; ++j) {
    double ber = (double) bdiff_total[j] / n_samples;
    var += (ber - mean_bdiff) * (ber - mean_bdiff);
  }
  double std_bdiff = n_dims > 1 ? sqrt(var / (n_dims - 1)) : 0.0;

  lua_setfield(L, -2, "bits");
  lua_pushnumber(L, 1.0 - mean_bdiff);
  lua_setfield(L, -2, "mean_hamming");
  lua_pushnumber(L, 1.0 - min_bdiff);
  lua_setfield(L, -2, "ber_min");
  lua_pushnumber(L, 1.0 - max_bdiff);
  lua_setfield(L, -2, "ber_max");
  lua_pushnumber(L, std_bdiff);
  lua_setfield(L, -2, "ber_std");

  free(bdiff_total);
  for (unsigned int i = 0; i < n_threads; i ++)
    free(data[i].bdiff);

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
  lua_settop(L, 7);

  tk_ivec_t *ids = tk_ivec_peek(L, 1, "ids");
  tk_dvec_t *dcodes = tk_dvec_peekopt(L, 2);
  tk_bits_t *codes = dcodes == NULL ? (tk_bits_t *) tk_lua_checkstring(L, 2, "codes") : NULL;
  if (!(dcodes != NULL || codes != NULL))
    tk_lua_verror(L, 3, "auc", "codes", "must be either a string or tk_dvec_t");

  tk_pvec_t *pos = tk_pvec_peek(L, 3, "pos");
  tk_pvec_t *neg = tk_pvec_peek(L, 4, "neg");
  uint64_t n_dims = tk_lua_checkunsigned(L, 5, "n_hidden");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_bits_t *mask =
    lua_type(L, 6) == LUA_TSTRING ? (tk_bits_t *) luaL_checkstring(L, 6) :
    lua_type(L, 6) == LUA_TLIGHTUSERDATA ? (tk_bits_t *) lua_touserdata(L, 6) : NULL;

  unsigned int n_threads = tk_threads_getn(L, 7, "n_threads", NULL);

  tk_eval_t state;
  memset(&state, 0, sizeof(tk_eval_t));
  state.n_dims = n_dims;
  state.chunks = BITS_BYTES(n_dims);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = mask;
  state.id_code = tk_iumap_from_ivec(ids);
  state.codes = codes;
  state.dcodes = dcodes != NULL ? dcodes->a : NULL;
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
  tk_iumap_destroy(state.id_code);

  lua_pushnumber(L, auc);
  return 1;
}

static inline int tm_optimize_clustering (lua_State *L)
{
  lua_settop(L, 1);

  lua_getfield(L, 1, "index");
  int i_index = tk_lua_absindex(L, -1);
  tk_inv_t *inv = tk_inv_peekopt(L, i_index);
  tk_ann_t *ann = tk_ann_peekopt(L, i_index);
  tk_hbi_t *hbi = tk_hbi_peekopt(L, i_index);

  if (inv == NULL && ann == NULL && hbi == NULL)
    tk_lua_verror(L, 3, "optimize_clustering", "index", "either tk_inv_t, tk_ann_t, or tk_inv_t must be provided");

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
  tk_accuracy_t best = { .bacc = -1.0, .tpr = 0.0, .tnr = 0.0 };
  uint64_t best_n = 0, best_m = 0;
  int i_ids = LUA_NOREF, i_assign = LUA_NOREF;
  for (uint64_t m = min_margin; m <= max_margin; ++m) {
    tk_cluster_dsu(L, hbi, ann, inv, m, &ids, &assignments, &n_clusters);
    tk_accuracy_t result =
      tm_clustering_accuracy(L, ids, assignments, pos, neg, n_threads);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, result.bacc);
      lua_pushnumber(L, result.tpr);
      lua_pushnumber(L, result.tnr);
      lua_pushinteger(L, (lua_Integer)m);
      lua_pushinteger(L, (lua_Integer)n_clusters);
      lua_call(L, 5, 0);
    }
    if (result.bacc > best.bacc) {
      best = result;
      best_n = n_clusters;
      best_m = m;
      luaL_unref(L, LUA_REGISTRYINDEX, i_assign);
      luaL_unref(L, LUA_REGISTRYINDEX, i_ids);
      i_assign = luaL_ref(L, LUA_REGISTRYINDEX); // assignments
      i_ids = luaL_ref(L, LUA_REGISTRYINDEX);  // ids
    } else {
      lua_pop(L, 2);
    }
  }

  lua_newtable(L);
  lua_pushinteger(L, (lua_Integer)best_m);
  lua_setfield(L, -2, "margin");
  lua_pushinteger(L, (lua_Integer)best_n);
  lua_setfield(L, -2, "n_clusters");
  lua_pushnumber(L, best.bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, best.tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, best.tnr);
  lua_setfield(L, -2, "tnr");

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

  uint64_t n_dims = tk_lua_fcheckunsigned(L, 1, "optimize_retrieval", "n_dims");
  unsigned int n_threads = tk_threads_getn(L, 1, "optimize_retrieval", "threads");
  uint64_t n_pos = pos->n;
  uint64_t n_neg = neg->n;

  tk_eval_t state;
  state.n_dims = n_dims;
  state.chunks = BITS_BYTES(n_dims);
  state.pl = malloc((n_pos + n_neg) * sizeof(tm_dl_t));
  state.mask = NULL;
  state.codes = codes;
  state.dcodes = NULL;
  state.id_code = ids == NULL ? NULL : tk_iumap_from_ivec(ids); // TODO: Pass id_code (aka uid_hood) in instead of recomputing
  state.pos = pos;
  state.neg = neg;
  state.n_pos = n_pos;
  state.n_neg = n_neg;
  state.hist_pos = tk_malloc(L, (n_dims + 1) * sizeof(atomic_ulong));
  state.hist_neg = tk_malloc(L, (n_dims + 1) * sizeof(atomic_ulong));
  for (uint64_t i = 0; i < n_dims + 1; i ++) {
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
  uint64_t pref_tp[n_dims + 1], pref_fp[n_dims + 1];
  uint64_t running_tp = 0, running_fp = 0;
  for (uint64_t d = 0; d <= n_dims; d ++) {
    running_tp += atomic_load(state.hist_pos + d);
    running_fp += atomic_load(state.hist_neg + d);
    pref_tp[d] = running_tp;
    pref_fp[d] = running_fp;
  }

  double best_bacc = -1.0;
  double best_tpr = -1.0;
  double best_tnr = -1.0;
  uint64_t best_margin = 0;
  for (uint64_t m = 0; m <= n_dims; ++m) {
    uint64_t tp = pref_tp[m];
    uint64_t fp = pref_fp[m];
    uint64_t tn = n_neg - fp;
    double tpr = n_pos > 0 ? (double)tp / (double)n_pos : 0;
    double tnr = n_neg > 0 ? (double)tn / (double)n_neg : 0;
    double bacc = 0.5 * (tpr + tnr);
    if (i_each > -1) {
      lua_pushvalue(L, i_each);
      lua_pushnumber(L, bacc);
      lua_pushnumber(L, tpr);
      lua_pushnumber(L, tnr);
      lua_pushinteger(L, (lua_Integer)m);
      lua_pushnumber(L, auc);
      lua_call(L, 5, 1);
      if (lua_isboolean(L, -1) && !lua_toboolean(L, -1)) {
        lua_pop(L, 1);
        break;
      }
      lua_pop(L, 1);
    }
    if (bacc > best_bacc || (bacc == best_bacc && tpr > best_tpr)) {
      best_bacc = bacc;
      best_tpr = tpr;
      best_tnr = tnr;
      best_margin = m;
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
  lua_pushnumber(L, best_bacc);
  lua_setfield(L, -2, "bacc");
  lua_pushnumber(L, best_tpr);
  lua_setfield(L, -2, "tpr");
  lua_pushnumber(L, best_tnr);
  lua_setfield(L, -2, "tnr");
  lua_pushnumber(L, auc);
  lua_setfield(L, -2, "auc");
  return 1;
}

static luaL_Reg tm_evaluator_fns[] =
{
  { "class_accuracy", tm_class_accuracy },
  { "encoding_accuracy", tm_encoding_accuracy },
  { "clustering_accuracy", tm_clustering_accuracy_lua },
  { "optimize_retrieval", tm_optimize_retrieval },
  { "optimize_clustering", tm_optimize_clustering },
  { "entropy_stats", tm_entropy_stats },
  { "auc", tm_auc },
  { NULL, NULL }
};

int luaopen_santoku_tsetlin_evaluator_capi (lua_State *L)
{
  lua_newtable(L); // t
  tk_lua_register(L, tm_evaluator_fns, 0); // t
  return 1;
}
